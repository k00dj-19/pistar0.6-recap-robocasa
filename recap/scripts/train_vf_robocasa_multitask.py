"""Train the RECAP distributional value function on the MULTI-TASK combination of
DEMO datasets + autonomous ROLLOUT datasets (which contain failures).

This is the step that makes RECAP work: by feeding the VF failure episodes (whose
returns collapse to -1 under Eq.5), the learned value -- and therefore the
downstream advantage -- encodes *action quality*, not just speed.

Usage:
  PYTHONPATH=recap python recap/scripts/train_vf_robocasa_multitask.py \
      --root .lerobot --out outputs/vf_robocasa_mt \
      --rollouts local/robocasa_rollouts_CloseFridge_s0 local/robocasa_rollouts_OpenDrawer_s0

The 7 demo repos (pepijn223/robocasa_<Task>) are always included in the fixed
RECAP order; --rollouts appends rollout repos (resolved under the same --root, so
a single root covers demos + rollouts -- see robocasa_multitask_data). Rollout
repos share each task's id with its demo via the canonical task name.

CPU-friendly (states only, no image decode); pass --device cpu or cuda.
"""

from __future__ import annotations

import argparse
import json
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
from recap.robocasa_vf import StateTaskVF, train_state_vf


def build_repo_list(root: str, demos: list[str] | None, rollouts: list[str]) -> list[RepoSpec]:
    if demos is None:
        demo_specs = default_demo_repos(root)
    else:
        demo_specs = [RepoSpec(repo_id=r, root=root, kind="demo") for r in demos]
    rollout_specs = [RepoSpec(repo_id=r, root=root, kind="rollout") for r in rollouts]
    return demo_specs + rollout_specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".lerobot",
                    help="base root; each dataset dir is <root>/<repo_id>")
    ap.add_argument("--demos", nargs="*", default=None,
                    help="override demo repo_ids (default: the 7 RECAP demo tasks)")
    ap.add_argument("--rollouts", nargs="*", default=[],
                    help="rollout repo_ids (local/robocasa_rollouts_<task>_s<seed>)")
    ap.add_argument("--out", default="outputs/vf_robocasa_mt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--c_fail", type=float, default=1.0e3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=None,
                    help="cap gradient steps (login dry-run: keep <=100)")
    ap.add_argument("--features", default="",
                    help="optional .npy of precomputed per-frame features (N,H), positionally "
                         "aligned to the combined demos+rollouts order (extract_vlm_features_*). "
                         "When set, the VF is trained over these VLM features instead of the "
                         "16-dim proprio state -> scene-aware critic.")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    repos = build_repo_list(args.root, args.demos, args.rollouts)
    combined = build_combined(repos, c_fail=args.c_fail)
    total = verify_frame_count(combined)

    # state matrix: proprio (default) OR precomputed VLM features (scene-aware VF).
    if args.features:
        state_mat = np.load(args.features).astype(np.float32)
        # iteration K=k uses a PREFIX of the full repo order, so its combined frames are the
        # first `total` rows of the full feature matrix -> slice (prefix-aligned by construction).
        if len(state_mat) > total:
            print(f"[features] slicing full features {len(state_mat)} -> first {total} "
                  f"(iter-subset prefix)", flush=True)
            state_mat = state_mat[:total]
        assert len(state_mat) == total, f"features {len(state_mat)} < frames {total}"
        print(f"[features] using VLM features {state_mat.shape} instead of proprio "
              f"state {combined.state.shape}", flush=True)
    else:
        state_mat = combined.state

    y = combined.returns
    n_tasks = combined.n_tasks
    n_fail_eps = sum(1 for e in combined.table if not e.success)
    print(f"[data] repos={len(repos)} episodes={len(combined.table)} "
          f"frames={total} state_dim={state_mat.shape[1]} tasks={n_tasks} "
          f"fail_episodes={n_fail_eps}", flush=True)
    print(f"[data] task_name_to_id={combined.task_name_to_id}", flush=True)
    print(f"[data] offsets={combined.offsets.tolist()}", flush=True)
    print(f"[data] target return min/mean/max = "
          f"{y.min():.3f}/{y.mean():.3f}/{y.max():.3f}", flush=True)

    vf = StateTaskVF(state_dim=state_mat.shape[1], n_tasks=n_tasks, n_bins=201)
    vf, hist = train_state_vf(
        vf, state_mat, combined.task_index, y,
        epochs=args.epochs, batch_size=args.batch_size, device=args.device,
        seed=args.seed, max_steps=args.max_steps,
    )

    ckpt = os.path.join(args.out, "vf.pt")
    torch.save({"state_dict": vf.state_dict(), "n_tasks": n_tasks,
                "state_dim": state_mat.shape[1], "history": hist,
                "repo_ids": combined.repo_ids,
                "task_name_to_id": combined.task_name_to_id}, ckpt)
    print(f"[save] {ckpt}", flush=True)

    # ---- value-over-time + calibration (Fig.4 style) ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    rng = np.random.default_rng(0)
    sample_eps = rng.choice(len(combined.table), size=min(8, len(combined.table)),
                            replace=False)
    for ei in sample_eps:
        e = combined.table[ei]
        pos = np.arange(e.from_index, e.to_index)
        S = torch.as_tensor(state_mat[pos], device=args.device)
        Tk = torch.as_tensor(combined.task_index[pos], device=args.device)
        with torch.no_grad():
            v = vf(S, Tk).cpu().numpy()
        t = np.linspace(0, 1, len(v))
        ls = "-" if e.success else "--"
        axes[0].plot(t, v, alpha=0.8, ls=ls,
                     label=f"{e.task}{'' if e.success else ' FAIL'}")
    axes[0].axhline(0.0, color="k", ls="--", lw=0.8)
    axes[0].set_xlabel("normalized time")
    axes[0].set_ylabel("V(o)")
    axes[0].set_title("Multi-task RECAP value function\n(success rises to 0; failures stay near -1)")
    axes[0].legend(fontsize=7)

    idx = rng.choice(total, size=min(4000, total), replace=False)
    S = torch.as_tensor(state_mat[idx], device=args.device)
    Tk = torch.as_tensor(combined.task_index[idx], device=args.device)
    with torch.no_grad():
        vpred = vf(S, Tk).cpu().numpy()
    axes[1].scatter(y[idx], vpred, s=3, alpha=0.25)
    axes[1].plot([-1, 0], [-1, 0], "r--", lw=1)
    axes[1].set_xlabel("MC return target (Eq.5, normalized)")
    axes[1].set_ylabel("predicted V(o)")
    axes[1].set_title("Value calibration")
    fig.tight_layout()
    figpath = os.path.join(args.out, "fig4_value_function.png")
    fig.savefig(figpath, dpi=130)
    print(f"[save] {figpath}", flush=True)

    with open(os.path.join(args.out, "history.json"), "w") as f:
        json.dump(hist, f, indent=1)


if __name__ == "__main__":
    main()
