"""Compute per-frame advantage + binarized improvement indicator I_t for the
MULTI-TASK RECAP combination of DEMO + ROLLOUT datasets, using the multi-task VF
(outputs/vf_robocasa_mt/vf.pt).

The saved I_dense / adv_dense are arrays of length == len(MultiLeRobotDataset(repo
list)) and POSITIONALLY ALIGNED to that exact order (demos in fixed order, then
rollouts), so the advantage-conditioned policy's AdvantageConditionedDataset can
look them up by combined position. See robocasa_multitask_data for why concatenating
per-repo parquet states (sorted by global frame index) reproduces the
MultiLeRobotDataset frame order.

Usage:
  PYTHONPATH=recap python recap/scripts/compute_indicators_robocasa_multitask.py \
      --root .lerobot --vf outputs/vf_robocasa_mt/vf.pt \
      --rollouts local/robocasa_rollouts_CloseFridge_s0 \
      --out outputs/vf_robocasa_mt --positive_fraction 0.30 --n_step 50

CPU-friendly (states only, no image/video decode).
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from recap.robocasa_multitask_data import (
    RepoSpec,
    build_combined,
    default_demo_repos,
    verify_frame_count,
)
from recap.robocasa_vf import StateTaskVF
from recap.advantage_robocasa import compute_advantages, build_indicators, build_indicators_per_task


def build_repo_list(root: str, demos: list[str] | None, rollouts: list[str]) -> tuple[list[RepoSpec], int]:
    if demos is None:
        demo_specs = default_demo_repos(root)
    else:
        demo_specs = [RepoSpec(repo_id=r, root=root, kind="demo") for r in demos]
    rollout_specs = [RepoSpec(repo_id=r, root=root, kind="rollout") for r in rollouts]
    return demo_specs + rollout_specs, len(demo_specs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".lerobot",
                    help="base root; each dataset dir is <root>/<repo_id>")
    ap.add_argument("--demos", nargs="*", default=None,
                    help="override demo repo_ids (default: the 7 RECAP demo tasks)")
    ap.add_argument("--rollouts", nargs="*", default=[],
                    help="rollout repo_ids (local/robocasa_rollouts_<task>_s<seed>)")
    ap.add_argument("--vf", default="outputs/vf_robocasa_mt/vf.pt")
    ap.add_argument("--out", default="outputs/vf_robocasa_mt")
    ap.add_argument("--c_fail", type=float, default=1.0e3)
    ap.add_argument("--positive_fraction", type=float, default=0.30)
    ap.add_argument("--n_step", type=int, default=50)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--demo_positive", action="store_true",
                    help="corrections-style labeling: force I=True on ALL demo frames (demos are "
                         "expert data, analogous to the paper forcing human corrections positive). "
                         "Rollout frames keep advantage-based labels.")
    ap.add_argument("--tag_suffix", default="", help="suffix for the output npz tag (e.g. 'corr')")
    ap.add_argument("--features", default="",
                    help="optional .npy of precomputed per-frame VLM features (N,H), positionally "
                         "aligned to the combined order. When set (with a VLM-feature --vf), the "
                         "advantage is read off the scene-aware critic instead of proprio state.")
    ap.add_argument("--per_task_threshold", action="store_true", default=True,
                    help="paper-faithful per-task ε_ℓ (top positive_fraction WITHIN each task). Default on.")
    ap.add_argument("--global_threshold", dest="per_task_threshold", action="store_false",
                    help="use a single global ε across all tasks instead of per-task ε_ℓ.")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    repos, n_demos = build_repo_list(args.root, args.demos, args.rollouts)
    combined = build_combined(repos, c_fail=args.c_fail)
    total = verify_frame_count(combined)

    ckpt = torch.load(args.vf, map_location=args.device)
    vf = StateTaskVF(state_dim=ckpt["state_dim"], n_tasks=ckpt["n_tasks"], n_bins=201)
    vf.load_state_dict(ckpt["state_dict"])

    # state matrix fed to the VF: proprio (default) OR precomputed VLM features.
    if args.features:
        state_mat = np.load(args.features).astype(np.float32)
        # iter-K subset = prefix of full repo order -> first `total` feature rows.
        if len(state_mat) > total:
            print(f"[features] slicing full features {len(state_mat)} -> first {total} "
                  f"(iter-subset prefix)", flush=True)
            state_mat = state_mat[:total]
        assert len(state_mat) == total, f"features {len(state_mat)} < frames {total}"
        assert state_mat.shape[1] == ckpt["state_dim"], (
            f"feature dim {state_mat.shape[1]} != VF state_dim {ckpt['state_dim']}")
        print(f"[features] advantage from VLM-feature VF {state_mat.shape}", flush=True)
    else:
        state_mat = combined.state

    # gindex == combined position (np.arange(N)); table holds COMBINED from/to.
    adv = compute_advantages(
        vf, state_mat, combined.task_index, combined.gindex,
        combined.table, n_step=args.n_step, device=args.device,
        norm_lengths=combined.norm_lengths,
    )
    valid = ~np.isnan(adv)
    print(f"[data] repos={len(repos)} frames={total} tasks={combined.n_tasks}", flush=True)
    if args.per_task_threshold:
        I, eps_map = build_indicators_per_task(
            adv, combined.task_index, positive_fraction=args.positive_fraction)
        eps = float(np.mean(list(eps_map.values()))) if eps_map else float("nan")  # representative
        print(f"[adv] PER-TASK ε_ℓ: " +
              ", ".join(f"t{t}={e:.4f}" for t, e in sorted(eps_map.items())), flush=True)
    else:
        I, eps = build_indicators(adv, positive_fraction=args.positive_fraction)
        eps_map = {}
        print(f"[adv] GLOBAL eps={eps:.4f}", flush=True)
    print(f"[adv] valid={valid.sum()}  positive_frac(target {args.positive_fraction})="
          f"{I[valid].mean():.3f}  mode={'per_task' if args.per_task_threshold else 'global'}", flush=True)

    # I_dense / adv_dense are aligned to the COMBINED position (== MultiLeRobotDataset order),
    # so they are length-N arrays indexed directly by combined position (no gather).
    I_dense = I.astype(bool)
    adv_dense = adv.astype(np.float32)
    assert len(I_dense) == total == len(adv_dense)

    # fail_dense: frames belonging to FAILURE episodes. The policy trainer uses this to
    # always attach "Advantage: negative" to failure frames (never train them as an
    # unconditional sample -> no unlabeled BC on bad actions).
    fail_dense = np.zeros(total, dtype=bool)
    for e in combined.table:
        if not e.success:
            fail_dense[e.from_index:e.to_index] = True
    print(f"[fail] failure frames: {int(fail_dense.sum())}/{total} ({100*fail_dense.mean():.0f}%)", flush=True)

    if args.demo_positive:
        demo_end = int(combined.offsets[n_demos])
        I[:demo_end] = True  # demos = expert data, forced positive (corrections-style)
        print(f"[demo_positive] forced I=True on demo span [0,{demo_end}) "
              f"-> overall positive_frac={I[valid].mean():.3f}", flush=True)

    tag = f"p{int(round(args.positive_fraction*100)):02d}{args.tag_suffix}"
    out_npz = os.path.join(args.out, f"indicators_{tag}.npz")
    np.savez(
        out_npz,
        I_dense=I_dense,
        adv_dense=adv_dense,
        adv=adv,
        indicator=I,
        gindex=combined.gindex,
        eps=eps,
        eps_per_task=np.asarray(sorted(eps_map.items()), dtype=object) if eps_map else np.asarray([], dtype=object),
        per_task_threshold=bool(args.per_task_threshold),
        fail_dense=fail_dense,
        demo_positive=bool(args.demo_positive),
        positive_fraction=args.positive_fraction,
        n_step=args.n_step,
        repo_ids=np.asarray(combined.repo_ids, dtype=object),
        offsets=combined.offsets,
        task_index=combined.task_index,
        n_frames=total,
    )
    print(f"[save] {out_npz}", flush=True)

    # diagnostic figure: advantage histogram + eps; per-task positive fraction
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    a = adv[valid]
    ax[0].hist(a, bins=80, color="steelblue", alpha=0.85)
    if args.per_task_threshold and eps_map:
        for t, e in sorted(eps_map.items()):
            ax[0].axvline(e, color="r", ls="--", alpha=0.5)
        ax[0].axvline(eps, color="darkred", ls="-", label=f"per-task ε_ℓ (mean {eps:.3f}, top {int(args.positive_fraction*100)}%)")
    else:
        ax[0].axvline(eps, color="r", ls="--", label=f"global ε={eps:.3f} (top {int(args.positive_fraction*100)}%)")
    ax[0].set_xlabel("advantage A(o_t)"); ax[0].set_ylabel("count")
    ax[0].set_title(f"Advantage distribution (N={args.n_step}-step)\npositive->'Advantage: positive'")
    ax[0].legend()
    tk = combined.task_index[valid]
    ntk = int(combined.task_index.max()) + 1
    fr = [I[valid][tk == t].mean() if (tk == t).any() else 0.0 for t in range(ntk)]
    ax[1].bar(range(len(fr)), fr, color="seagreen")
    ax[1].axhline(args.positive_fraction, color="r", ls="--", label="target")
    ax[1].set_xlabel("task index"); ax[1].set_ylabel("fraction positive")
    ax[1].set_title("Per-task positive fraction"); ax[1].legend()
    fig.tight_layout()
    figp = os.path.join(args.out, f"advantage_{tag}.png")
    fig.savefig(figp, dpi=130); print(f"[save] {figp}", flush=True)


if __name__ == "__main__":
    main()
