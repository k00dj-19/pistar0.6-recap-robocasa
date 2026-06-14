"""Finetune π0.5 with RECAP advantage conditioning on RoboCasa CloseFridge — RoboCasa
mirror of train_pi05_recap.py (the proven LIBERO finetune).

Why this exists: the public RoboCasa pi05 checkpoints don't give a usable baseline on
CloseFridge (the quantized 25k ckpt is too weak → 0%; ruiname/10tasks-200k doesn't include
CloseFridge in its training config + has a camera-key mismatch). So we train OUR OWN π0.5
on the 513 CloseFridge demos as the honest baseline, then add RECAP advantage conditioning
on top — exactly the LIBERO recipe, generalized.

Key differences from the LIBERO script:
  * dataset = pepijn223/robocasa_CloseFridge (VIDEO format) — loaded with video_backend=pyav
    because .venv_robocasa's torchcodec is broken (ffmpeg ABI mismatch). pyav decodes fine.
  * 3 REAL cameras (robot0_eye_in_hand / robot0_agentview_left / robot0_agentview_right) —
    no empty_camera synthesis needed (LIBERO had 2 real + 1 empty).
  * state dim 16, action dim 12 (vs LIBERO 8 / 7) — handled automatically by
    dataset_to_policy_features.
  * episode/return source = recap.robocasa_data (success can be False for rollouts; demos
    default success=True).

The env (lerobot-eval RoboCasaEnv) emits the SAME raw camera names as the dataset, so the
policy trained here needs NO --rename_map at eval time.

Usage (smoke):
  PYTHONPATH=recap python recap/scripts/train_pi05_recap_robocasa.py \
     --root .lerobot/pepijn223/robocasa_CloseFridge \
     --indicators outputs/vf_robocasa/indicators_p30.npz --mode sft \
     --episodes_per_task 30 --steps 30 --batch_size 8 --out outputs/pi05_robocasa_sft_smoke
"""

from __future__ import annotations

import argparse
import os
import time

# Fully offline against local HF caches (see train_pi05_recap.py header for the rationale).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Skip the online dataset refs-check so the populated local copy is used offline. Imported
# into multiple modules, so patch every binding.
def _local_version(repo_id, version=None, *a, **k):  # noqa: ANN001
    return version or "main"
import lerobot.datasets.utils as _du            # noqa: E402
import lerobot.datasets.dataset_metadata as _dm  # noqa: E402
import lerobot.datasets.lerobot_dataset as _dl   # noqa: E402
for _m in (_du, _dm, _dl):
    if hasattr(_m, "get_safe_version"):
        _m.get_safe_version = _local_version

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.multi_dataset import MultiLeRobotDataset
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors

from recap.robocasa_data import load_episode_table, subsample_episodes
from recap.advantage_dataset import AdvantageConditionedDataset

REPO_ID = "pepijn223/robocasa_CloseFridge"

# modules kept trainable when --train_expert_only (freeze the VLM backbone). Fine for the
# SFT baseline (plain BC), but NOT for recap: with the VLM frozen the "Advantage:
# positive/negative" tokens barely move the action expert (LIBERO probe: 0.9%).
EXPERT_KEYS = ("gemma_expert", "action_in_proj", "action_out_proj",
               "action_time_mlp", "state_proj", "time_mlp")


def set_trainable(policy, train_expert_only: bool):
    if not train_expert_only:
        return sum(p.numel() for p in policy.parameters() if p.requires_grad)
    trainable = 0
    for name, p in policy.named_parameters():
        keep = any(k in name for k in EXPERT_KEYS)
        p.requires_grad_(keep)
        if keep:
            trainable += p.numel()
    return trainable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="single-task: local dataset dir. multi-task (--tasks): the .lerobot home "
                         "(parent of pepijn223/), e.g. /home/.../.venv? -> .lerobot")
    ap.add_argument("--tasks", default="",
                    help="comma-separated RoboCasa task names for MULTI-TASK training "
                         "(e.g. 'CloseFridge,OpenDrawer,PickPlaceCounterToCabinet'). "
                         "Empty = single-task via --repo_id/--root.")
    ap.add_argument("--rollouts", default="",
                    help="comma-separated rollout repo_ids to APPEND after the demo repos for "
                         "RECAP iteration (e.g. 'local/robocasa_rollouts_CloseFridge,...'). "
                         "MUST match the exact order used to build --indicators. multi-task only.")
    ap.add_argument("--repo_id", default=REPO_ID)
    ap.add_argument("--base_ckpt", default="lerobot/pi05_base",
                    help="generic pi05 weights (max RECAP headroom); config is overridden to "
                         "the RoboCasa 3-camera / state16 / action12 layout below")
    ap.add_argument("--video_backend", default="pyav",
                    help="pyav — .venv_robocasa torchcodec is broken (ffmpeg ABI mismatch)")
    ap.add_argument("--indicators", default="outputs/vf_robocasa/indicators_p30.npz")
    ap.add_argument("--mode", choices=["recap", "sft", "awr", "filtered"], default="sft")
    ap.add_argument("--episodes_per_task", type=int, default=0,
                    help="data-limited subset per task; 0 = use ALL episodes")
    ap.add_argument("--dropout", type=float, default=0.30)
    ap.add_argument("--awr_temp", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2.5e-5)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--train_expert_only", action="store_true",
                    help="freeze VLM (fast). OK for sft baseline; recap MUST be full finetune")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--save_every", type=int, default=1000)
    args = ap.parse_args()
    root = args.root.rstrip("/")
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)

    # ---- DDP setup (no-op unless launched via torchrun) ----
    # Multi-GPU data-parallel: torchrun --nproc_per_node=N sets LOCAL_RANK/RANK/WORLD_SIZE.
    # Each rank holds a full pi05 replica on its own GPU; effective batch = batch_size x N,
    # so wall-clock for the same data coverage drops ~N-fold. Single-GPU path is untouched.
    is_ddp = "LOCAL_RANK" in os.environ
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    is_main = rank == 0
    if is_ddp:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        if is_main:
            print(f"[ddp] world_size={world_size} local_rank={local_rank} device={device} "
                  f"effective_batch={args.batch_size * world_size}", flush=True)
    else:
        device = args.device

    def log0(*a, **k):
        if is_main:
            print(*a, **k)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    # The "multitask" (MultiLeRobotDataset) branch is also required for a SINGLE task when
    # rollout repos are appended (per-skill specialist RECAP: D_ℓ = task demo + its rollouts).
    multitask = len(tasks) > 1 or (len(tasks) == 1 and bool(args.rollouts.strip()))

    # chunk_size from the base ckpt config (pi05 suffix length for the action chunk).
    import glob as _glob, json as _json
    _cfgs = _glob.glob(os.path.expanduser(
        f"~/pi_06_star/.hf/hub/models--{args.base_ckpt.replace('/','--')}/snapshots/*/config.json"))
    chunk_size = 50
    if _cfgs:
        try:
            chunk_size = int(_json.load(open(_cfgs[0])).get("chunk_size", 50))
        except Exception:
            pass

    if multitask:
        # ---- MULTI-TASK: concatenate N RoboCasa task repos (MultiLeRobotDataset aggregates
        # normalization stats across tasks and preserves each frame's `task` string for pi05's
        # language conditioning). Global frame index is deterministic in repo_ids order, which
        # lets the recap indicators (built later from the multi-task VF) align by offset. ----
        demo_repos = [f"pepijn223/robocasa_{t}" for t in tasks]
        # RECAP iteration: APPEND accumulated rollout repos AFTER the demos, in the SAME order
        # used to build --indicators (demos first, then rollouts), so the positionally-aligned
        # i_dense/adv_dense line up with MultiLeRobotDataset's combined frame order.
        rollout_repos = [r.strip() for r in args.rollouts.split(",") if r.strip()]
        repo_ids = demo_repos + rollout_repos
        from pathlib import Path
        meta0 = LeRobotDatasetMetadata(demo_repos[0], root=f"{root}/{demo_repos[0]}")
        fps = meta0.fps
        delta_timestamps = {"action": [i / fps for i in range(chunk_size)]}
        print(f"[data] MULTI-TASK demos={len(demo_repos)} rollouts={len(rollout_repos)} "
              f"tasks={tasks} fps={fps} chunk_size={chunk_size} backend={args.video_backend}")
        base = MultiLeRobotDataset(repo_ids, root=Path(root),
                                   delta_timestamps=delta_timestamps,
                                   video_backend=args.video_backend)
        meta_features = base._datasets[0].meta.features
        meta_stats = base.stats
        print(f"[data] total frames={len(base)} across {len(repo_ids)} repos "
              f"(per-repo: {[len(d) for d in base._datasets]})")
    else:
        # ---- SINGLE-TASK (original path, unchanged behavior) ----
        meta = LeRobotDatasetMetadata(args.repo_id, root=root)
        table = load_episode_table(meta)
        n_per_task = args.episodes_per_task if args.episodes_per_task > 0 else None
        keep_eps = subsample_episodes(table, n_per_task=n_per_task, seed=args.seed)
        print(f"[data] {len(keep_eps)}/{len(table)} episodes "
              f"({'all' if n_per_task is None else n_per_task}/task)")
        fps = meta.fps
        delta_timestamps = {"action": [i / fps for i in range(chunk_size)]}
        print(f"[data] action chunk_size={chunk_size} fps={fps} video_backend={args.video_backend}")
        base = LeRobotDataset(args.repo_id, root=root, episodes=keep_eps,
                              delta_timestamps=delta_timestamps, video_backend=args.video_backend)
        meta_features = base.meta.features
        meta_stats = base.meta.stats

    # ---- indicators (only for advantage-using modes; plain SFT needs none) ----
    i_dense = adv_dense = fail_dense = None
    if args.mode != "sft" and args.indicators and os.path.exists(args.indicators):
        npz = np.load(args.indicators)
        i_dense, adv_dense = npz["I_dense"], npz["adv_dense"]
        fail_dense = npz["fail_dense"] if "fail_dense" in npz.files else None
        if fail_dense is not None:
            print(f"[ind] fail_dense present: {int(fail_dense.sum())} failure frames "
                  f"(always 'Advantage: negative', never unconditional)")
        print(f"[ind] frames={len(i_dense)} positive_frac={float(i_dense.mean()):.3f}")
        if multitask:
            # indicators are positionally aligned to MultiLeRobotDataset(demos+rollouts) order;
            # the lengths MUST match or the conditioning is misaligned -> fail loudly.
            assert len(i_dense) == len(base), (
                f"indicator length {len(i_dense)} != dataset length {len(base)} — "
                f"--rollouts must match the repos used to build --indicators (same order)")

    if args.mode == "sft" or i_dense is None:
        ds = base  # plain BC: no advantage-conditioning wrapper needed
    else:
        ds = AdvantageConditionedDataset(base, i_dense, adv_dense, fail_dense=fail_dense, mode=args.mode,
                                         dropout=args.dropout, seed=args.seed,
                                         index_by_position=multitask)

    # filtered-BC: restrict sampling to positive-advantage frames present in the subset
    sampler = None
    shuffle = True
    if args.mode == "filtered" and multitask:
        # SUCCESS-filtered BC over the combined demos+rollouts: plain BC (no conditioning;
        # the wrapper attaches no phrase in this mode) sampled ONLY from non-failure frames
        # (all demo frames + successful-rollout frames). fail_dense comes from the indicator
        # npz and is positionally aligned to the combined order.
        assert fail_dense is not None and len(fail_dense) == len(base), \
            "filtered (multitask) needs fail_dense in --indicators aligned to the combined dataset"
        w = (~fail_dense).astype(np.float64)
        print(f"[filtered] non-failure frames: {int(w.sum())}/{len(base)}")
        sampler = WeightedRandomSampler(weights=w.tolist(), num_samples=len(base), replacement=True)
        shuffle = False
    elif args.mode == "filtered" and not multitask:
        idx_in_subset = []
        for pos in range(len(base)):
            g = int(base.hf_dataset[pos]["index"]) if hasattr(base, "hf_dataset") else None
            if g is not None and g < len(i_dense) and i_dense[g]:
                idx_in_subset.append(pos)
        print(f"[filtered] positive frames in subset: {len(idx_in_subset)}/{len(base)}")
        w = np.zeros(len(base)); w[idx_in_subset] = 1.0
        sampler = WeightedRandomSampler(weights=w.tolist(), num_samples=len(base), replacement=True)
        shuffle = False

    # DDP: shard the dataset across ranks with a DistributedSampler (overrides shuffle).
    if is_ddp and sampler is None:
        sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
        shuffle = False
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler,
                    num_workers=args.num_workers, drop_last=True, persistent_workers=args.num_workers > 0)

    # ---- policy ----
    # Load generic pi05_base weights and apply the RoboCasa feature layout derived from the
    # dataset itself (3 native cameras, state16, action12). FULL finetune (VLM included) so the
    # "Advantage: positive/negative" tokens actually influence actions — train_expert_only made
    # conditioning a near no-op on the LIBERO track (see docs/CRITIC_REVIEW_LIBERO.md).
    t0 = time.time()
    # RESUME: if a prior (preempted) run left a checkpoint + resume_state in --out, init from
    # there and continue; else start from base_ckpt. Enables safe extra/share preemption.
    _resume_state = os.path.join(args.out, "resume_state.pt")
    _resume_ckpt = os.path.join(args.out, "model.safetensors")
    _resuming = os.path.exists(_resume_state) and os.path.exists(_resume_ckpt)
    _init_from = args.out if _resuming else args.base_ckpt
    policy = PI05Policy.from_pretrained(_init_from)
    from lerobot.utils.feature_utils import dataset_to_policy_features
    from lerobot.configs.types import FeatureType, NormalizationMode

    feats = dataset_to_policy_features(meta_features)
    in_feats = {k: v for k, v in feats.items() if v.type is not FeatureType.ACTION}
    policy.config.empty_cameras = 0  # RoboCasa already supplies 3 real cameras
    policy.config.input_features = in_feats
    policy.config.output_features = {k: v for k, v in feats.items() if v.type is FeatureType.ACTION}
    policy.config.normalization_mapping = {
        "VISUAL": NormalizationMode.IDENTITY,
        "STATE": NormalizationMode.MEAN_STD,
        "ACTION": NormalizationMode.MEAN_STD,
    }
    cam_in = [k for k in in_feats if "image" in k]
    print(f"[cfg] RoboCasa layout: cameras={cam_in} "
          f"state={in_feats.get('observation.state').shape if 'observation.state' in in_feats else '?'} "
          f"action={policy.config.output_features['action'].shape}")

    if args.mode == "recap" and args.train_expert_only:
        raise SystemExit("[fatal] recap mode requires full finetune (VLM must see the advantage "
                         "tokens) — drop --train_expert_only. See LIBERO probe (0.9% no-op).")
    policy.config.device = device
    policy.to(device)
    ntr = set_trainable(policy, args.train_expert_only)
    # Full finetune of the 3B model needs gradient checkpointing to fit on one B200;
    # expert-only (693M) fits without it.
    if not args.train_expert_only:
        policy.config.gradient_checkpointing = True
    log0(f"[policy] loaded in {time.time()-t0:.1f}s | trainable={ntr/1e6:.0f}M "
         f"| train_expert_only={args.train_expert_only} "
         f"| grad_ckpt={getattr(policy.config,'gradient_checkpointing',False)}")

    # Wrap in DDP. find_unused_parameters=True is safe for full-finetune + gradient
    # checkpointing (some VLM params can be skipped on a given step); static_graph would be
    # faster but is fragile with grad-ckpt, so we accept the small overhead.
    fwd = policy
    if is_ddp:
        fwd = DDP(policy, device_ids=[local_rank], output_device=local_rank,
                  find_unused_parameters=True)

    pre, post = make_pi05_pre_post_processors(policy.config, meta_stats)

    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    # RESUME: restore step + optimizer/scheduler so a preempted run continues mid-training.
    start_step = 0
    if _resuming:
        try:
            rs = torch.load(_resume_state, map_location=device)
            start_step = int(rs.get("step", 0))
            opt.load_state_dict(rs["opt"]); sched.load_state_dict(rs["sched"])
            log0(f"[resume] continuing from step {start_step} (loaded {args.out})")
        except Exception as e:
            log0(f"[resume] state reload failed ({e}); restarting optimizer from step {start_step}")

    cam_keys = [k for k in meta_features if k.startswith("observation.images")]
    policy.train()
    epoch = 0
    if isinstance(sampler, DistributedSampler):
        sampler.set_epoch(epoch)
    di = iter(dl)
    t_log = time.time()
    for step in range(start_step + 1, args.steps + 1):
        try:
            batch = next(di)
        except StopIteration:
            epoch += 1
            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(epoch)  # reshuffle shards each epoch
            di = iter(dl); batch = next(di)
        for cam in cam_keys:
            if cam in batch and batch[cam].dtype == torch.uint8:
                batch[cam] = batch[cam].float() / 255.0
        adv = batch.pop("advantage", None)
        batch.pop("is_positive", None)
        batch.pop("dataset_index", None)  # MultiLeRobotDataset adds this; not a policy input
        batch = pre(batch)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            if args.mode == "awr":
                per_sample, d = fwd(batch, reduction="none")
                w = torch.exp((adv.to(per_sample.device) / args.awr_temp)).clamp(max=20.0)
                loss = (per_sample * w).sum() / (w.sum() + 1e-6)
            else:
                loss, d = fwd(batch)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], 1.0)
        opt.step(); sched.step()

        if step % args.log_every == 0:
            dt = (time.time() - t_log) / args.log_every
            log0(f"[{args.mode}] step {step:5d}/{args.steps}  loss={float(loss):.4f}  "
                 f"lr={sched.get_last_lr()[0]:.2e}  {dt:.2f}s/it", flush=True)
            t_log = time.time()
        if (step % args.save_every == 0 or step == args.steps) and is_main:
            policy.save_pretrained(args.out)  # save the underlying module, not the DDP wrapper
            pre.save_pretrained(args.out); post.save_pretrained(args.out)
            torch.save({"step": step, "opt": opt.state_dict(), "sched": sched.state_dict()},
                       os.path.join(args.out, "resume_state.pt"))
            print(f"[save] {args.out} @ step {step}", flush=True)

    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()
    if is_main:
        # completion marker for the output-driven manager (distinguishes done vs preempted).
        with open(os.path.join(args.out, "TRAIN_DONE"), "w") as _f:
            _f.write(f"steps={args.steps}\n")
    log0("[done]", args.out)


if __name__ == "__main__":
    main()
