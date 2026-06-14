"""Extract per-frame VLM features from a π0.5 (PaliGemma) backbone for a SCENE-AWARE
RECAP value function.

Motivation: our default RoboCasa VF (robocasa_vf.StateTaskVF) sees ONLY the 16-dim
PandaOmron proprio state — it is blind to the scene (drawer/door/object state lives in
the 3 cameras, which the VF never reads). For a method whose entire signal is advantage
QUALITY, a proprio-only critic is the bottleneck. The paper's V_pre is fine-tuned FROM the
VLM backbone. Here we approximate that faithfully and cheaply: run the (frozen, already
RoboCasa-adapted) mt_sft PI05 backbone over each frame, masked-mean-pool the contextual
PaliGemma *prefix* hidden states, and emit a (N, H) feature matrix. Downstream the VF is
just StateTaskVF(state_dim=H) trained on these features — so the only new/expensive piece
is this one-time forward pass, which we fan out over 4 GPUs.

Alignment: the features are written in MultiLeRobotDataset(repo_ids) frame order (combined
position 0..N-1), IDENTICAL to build_combined()'s order, so the existing positionally
aligned advantage/indicator pipeline consumes them unchanged (just `--features`).

Usage (4-GPU DDP via torchrun):
  torchrun --nproc_per_node=4 recap/scripts/extract_vlm_features_robocasa.py \
      --root .lerobot --base_ckpt outputs/robocasa/multi_task/sft \
      --demos pepijn223/robocasa_PickPlaceCounterToCabinet \
      --rollouts local/robocasa_rollouts_PickPlaceCounterToCabinet ... \
      --out outputs/robocasa/specialist_v2/PickPlaceCounterToCabinet/vlmvf
"""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _local_version(repo_id, version=None, *a, **k):  # noqa: ANN001
    return version or "main"
import lerobot.datasets.utils as _du            # noqa: E402
import lerobot.datasets.dataset_metadata as _dm  # noqa: E402
import lerobot.datasets.lerobot_dataset as _dl   # noqa: E402
for _m in (_du, _dm, _dl):
    if hasattr(_m, "get_safe_version"):
        _m.get_safe_version = _local_version

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: E402
from lerobot.datasets.multi_dataset import MultiLeRobotDataset  # noqa: E402
from lerobot.policies.pi05.modeling_pi05 import (  # noqa: E402
    PI05Policy,
    make_att_2d_masks,
    OBS_LANGUAGE_TOKENS,
    OBS_LANGUAGE_ATTENTION_MASK,
)
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors  # noqa: E402


@torch.no_grad()
def prefix_features(policy, batch):
    """Contextual PaliGemma prefix hidden states, masked-mean-pooled -> (B, H).

    Mirrors PI05Pytorch.sample_actions' prefix encode (embed_prefix -> 2d masks ->
    position ids -> 4d masks -> paligemma_with_expert.forward([prefix, None])), but keeps
    the prefix OUTPUT embeddings instead of the KV cache.
    """
    model = policy.model
    images, img_masks = policy._preprocess_images(batch)
    tokens, masks = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]

    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, tokens, masks)
    prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_pos_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    prefix_att_2d_4d = model._prepare_attention_masks_4d(prefix_att_2d)
    model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001

    (prefix_out, _), _ = model.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_4d,
        position_ids=prefix_pos_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=False,
    )
    # masked mean-pool over valid prefix tokens (image + language)
    m = prefix_pad_masks[..., None].to(prefix_out.dtype)
    feat = (prefix_out * m).sum(1) / m.sum(1).clamp(min=1.0)
    return feat.float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".lerobot")
    ap.add_argument("--base_ckpt", required=True, help="mt_sft (RoboCasa-adapted π0.5) dir")
    ap.add_argument("--demos", nargs="+", required=True, help="demo repo_ids, in fixed order")
    ap.add_argument("--rollouts", nargs="*", default=[], help="rollout repo_ids, AFTER demos")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=48)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--video_backend", default="pyav")
    ap.add_argument("--max_frames", type=int, default=0,
                    help="debug: cap per-rank frames (login smoke, keep <=100). 0 = all.")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    is_ddp = "LOCAL_RANK" in os.environ
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    is_main = rank == 0
    if is_ddp:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cuda"

    def log0(*a, **k):
        if is_main:
            print(*a, **k, flush=True)

    repo_ids = list(args.demos) + list(args.rollouts)
    meta0 = LeRobotDatasetMetadata(repo_ids[0], root=f"{args.root}/{repo_ids[0]}")
    fps = meta0.fps
    # keep an action chunk in the batch so the π0.5 pre-processor pipeline runs unchanged
    # (it normalizes action); frame ORDER is independent of delta_timestamps.
    import glob as _glob, json as _json
    _cfgs = _glob.glob(os.path.join(args.base_ckpt, "config.json"))
    chunk_size = 50
    if _cfgs:
        try:
            chunk_size = int(_json.load(open(_cfgs[0])).get("chunk_size", 50))
        except Exception:
            pass
    delta_timestamps = {"action": [i / fps for i in range(chunk_size)]}

    base = MultiLeRobotDataset(repo_ids, root=Path(args.root),
                               delta_timestamps=delta_timestamps,
                               video_backend=args.video_backend)
    N = len(base)
    log0(f"[data] repos={len(repo_ids)} frames={N} per-repo={[len(d) for d in base._datasets]}")

    # ---- policy: frozen mt_sft, RoboCasa layout (mirror the trainer) ----
    t0 = time.time()
    policy = PI05Policy.from_pretrained(args.base_ckpt)
    from lerobot.utils.feature_utils import dataset_to_policy_features
    from lerobot.configs.types import FeatureType, NormalizationMode
    meta_features = base._datasets[0].meta.features
    meta_stats = base.stats
    feats = dataset_to_policy_features(meta_features)
    in_feats = {k: v for k, v in feats.items() if v.type is not FeatureType.ACTION}
    policy.config.empty_cameras = 0
    policy.config.input_features = in_feats
    policy.config.output_features = {k: v for k, v in feats.items() if v.type is FeatureType.ACTION}
    policy.config.normalization_mapping = {
        "VISUAL": NormalizationMode.IDENTITY,
        "STATE": NormalizationMode.MEAN_STD,
        "ACTION": NormalizationMode.MEAN_STD,
    }
    policy.config.device = device
    policy.to(device).eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    pre, _ = make_pi05_pre_post_processors(policy.config, meta_stats)
    cam_keys = [k for k in meta_features if k.startswith("observation.images")]
    log0(f"[policy] loaded {args.base_ckpt} in {time.time()-t0:.1f}s | cams={cam_keys}")

    # ---- strided shard: rank r handles positions r, r+W, r+2W, ... ----
    my_positions = list(range(rank, N, world_size))
    if args.max_frames > 0:
        my_positions = my_positions[:args.max_frames]
    shard = os.path.join(args.out, f"feat_shard_{rank}.npz")

    # RESUME: positions are processed in deterministic order, so a prior shard's feats are
    # exactly the first len(prev) of my_positions — skip them and continue. Periodic flush
    # below means a timeout/preempt never loses more than one flush interval.
    feats_out = []
    n_done = 0
    if args.max_frames == 0 and os.path.exists(shard):
        prev = np.load(shard)["feats"]
        if len(prev) > 0:
            feats_out.append(prev)
            n_done = len(prev)
            print(f"[resume] rank {rank} found {n_done} cached feats -> skipping ahead", flush=True)
    remaining = my_positions[n_done:]

    def save_shard(nproc):
        arr = np.concatenate(feats_out, 0) if feats_out else np.zeros((0, 1), np.float16)
        pos = np.asarray(my_positions[:n_done + nproc], dtype=np.int64)
        np.savez(shard, positions=pos, feats=arr)

    done = 0  # newly processed THIS run
    FLUSH = 100  # batches (~4.8k frames, ~7min) between shard checkpoints — short enough that a
                 # preempt on volatile extra/share pools loses < one interval of progress
    t_log = time.time()
    if remaining:
        dl = DataLoader(Subset(base, remaining), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, drop_last=False,
                        persistent_workers=args.num_workers > 0)
        bi = 0
        for batch in dl:
            for cam in cam_keys:
                if cam in batch and batch[cam].dtype == torch.uint8:
                    batch[cam] = batch[cam].float() / 255.0
            batch.pop("dataset_index", None)
            batch = pre(batch)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                f = prefix_features(policy, batch)
            feats_out.append(f.cpu().to(torch.float16).numpy())
            done += f.shape[0]
            bi += 1
            if bi % FLUSH == 0:
                save_shard(done)
            if is_main and bi % 20 == 0:
                dt = (time.time() - t_log)
                rate = done / max(dt, 1e-6)
                eta = (len(remaining) - done) / max(rate, 1e-6)
                log0(f"[extract] rank0 {n_done+done}/{len(my_positions)}  {rate:.0f} fr/s  eta {eta/60:.1f}min")

    save_shard(done)
    feats_arr = np.concatenate(feats_out, 0) if feats_out else np.zeros((0, 1), np.float16)
    print(f"[shard] rank {rank} wrote {feats_arr.shape} -> {shard}", flush=True)

    if is_ddp:
        dist.barrier()
    # ---- rank0 merges shards into features.npy aligned to combined position 0..N-1 ----
    if is_main and args.max_frames > 0:
        print(f"[smoke] max_frames={args.max_frames}: feats {feats_arr.shape} dtype={feats_arr.dtype} "
              f"mean={float(feats_arr.astype('float32').mean()):.4f} — tap OK, skipping full merge",
              flush=True)
    elif is_main:
        H = feats_arr.shape[1]
        full = np.zeros((N, H), dtype=np.float16)
        filled = np.zeros(N, dtype=bool)
        for r in range(world_size):
            z = np.load(os.path.join(args.out, f"feat_shard_{r}.npz"))
            full[z["positions"]] = z["feats"]
            filled[z["positions"]] = True
        assert filled.all(), f"missing {int((~filled).sum())} positions after merge"
        outp = os.path.join(args.out, "features.npy")
        np.save(outp, full)
        print(f"[merge] features {full.shape} dtype={full.dtype} -> {outp}", flush=True)
        for r in range(world_size):
            os.remove(os.path.join(args.out, f"feat_shard_{r}.npz"))

    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
