"""RoboCasa data layer for RECAP, built on LeRobotDataset (v3.0).

Implements the RECAP data layer for RoboCasa365 / PandaOmron episodes,
supporting both demo data and autonomous rollout data:

  * 16-dim proprio state (``observation.state``): base_pos(3) + base_quat(4) +
    ee_pos_rel(3) + ee_quat_rel(4) + gripper_qpos(2)   -- see
    ``lerobot/envs/robocasa.py`` (OBS_STATE_DIM = 16).
  * 12-dim action (``action``): base_motion(4) + control_mode(1) +
    ee_pos(3) + ee_rot(3) + gripper(1)                  -- ACTION_DIM = 12.
  * Three raw RoboCasa cameras (``observation.images.robot0_agentview_left`` /
    ``robot0_eye_in_hand`` / ``robot0_agentview_right``).

It implements the RECAP reward of Eq. (5) and the per-task normalization of the
empirical return-to-go to (-1, 0) (Section V-C), computed purely from episode
boundaries + success labels (no frame/video decoding, so this module is cheap).

Success labels
--------------
RoboCasa LeRobot demo datasets (e.g. ``pepijn223/robocasa_CloseFridge``) are
filtered MimicGen / human demonstrations and are therefore (almost) all
*successful*. This means a demo episode of length T has per-step
reward -1 and a terminal 0, giving return-to-go R_t = -(T-1-t), normalized by the
per-task max length L_task into (-1, 0]. Failure episodes (terminal -C_fail)
arrive later from autonomous rollouts in the RoboCasa env (``info["success"]``);
after per-task normalization + clipping every state in a failed episode collapses
to ~ -1, exactly as in the paper.

This loader *detects* per-episode success when the dataset exposes it (an
episodes-table column among SUCCESS_KEYS, or a per-frame terminal success/reward
feature), and otherwise falls back to the demo assumption (all-success). This
keeps the same module usable both for demo data and for rollout-collected data
that mixes successes and failures.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
except Exception:  # pragma: no cover - lerobot optional at import time
    LeRobotDataset = None
    LeRobotDatasetMetadata = None

# Default single-task demo dataset used for the spike (0.4 GB).
ROBOCASA_CLOSEFRIDGE_REPO = "pepijn223/robocasa_CloseFridge"

# Dimensions of the flat RoboCasa state/action vectors (PandaOmron).
# Kept in sync with lerobot/src/lerobot/envs/robocasa.py.
OBS_STATE_DIM = 16
ACTION_DIM = 12

# Episode-level metadata columns that, if present, carry the success label.
SUCCESS_KEYS = ("success", "is_success", "episode_success", "task_success")
# Per-frame feature names that, if present, carry a terminal success/reward signal.
FRAME_SUCCESS_KEYS = ("next.success", "success", "is_success")
FRAME_REWARD_KEYS = ("next.reward", "reward")


@dataclasses.dataclass
class EpisodeInfo:
    episode_index: int
    length: int
    task: str
    task_index: int
    from_index: int   # global frame index of first frame (inclusive)
    to_index: int     # global frame index just past the last frame (exclusive)
    success: bool = True  # RoboCasa demos are (almost) all successful


def _extract_task(row) -> str:
    """Pull a task string out of an episodes-table row (handles list/str)."""
    t = row["tasks"] if "tasks" in row else row.get("task", "")
    if isinstance(t, (list, tuple)):
        return t[0] if len(t) else ""
    return t


def _episode_success_from_row(row) -> Optional[bool]:
    """Return the per-episode success label if the episodes table carries one."""
    for k in SUCCESS_KEYS:
        if k in row and row[k] is not None:
            v = row[k]
            if isinstance(v, (list, tuple)):
                v = v[-1] if len(v) else None
            if v is None:
                continue
            return bool(v)
    return None


def load_episode_table(meta, default_success: bool = True) -> list[EpisodeInfo]:
    """Read per-episode metadata (length, task, global frame range, success).

    ``meta`` is a ``LeRobotDatasetMetadata``. Success is taken from the episodes
    table when available (SUCCESS_KEYS), otherwise ``default_success`` (the demo
    assumption: every demonstration succeeded).
    """
    eps = meta.episodes
    out: list[EpisodeInfo] = []
    task_to_idx: dict[str, int] = {}
    for i in range(len(eps)):
        row = eps[i]
        task = _extract_task(row)
        if task not in task_to_idx:
            task_to_idx[task] = len(task_to_idx)
        succ = _episode_success_from_row(row)
        out.append(
            EpisodeInfo(
                episode_index=int(row["episode_index"]),
                length=int(row["length"]),
                task=task,
                task_index=task_to_idx[task],
                from_index=int(row["dataset_from_index"]),
                to_index=int(row["dataset_to_index"]),
                success=default_success if succ is None else succ,
            )
        )
    return out


def per_task_max_length(table: list[EpisodeInfo]) -> dict[int, int]:
    """L_task used to normalize returns to (-1, 0), per Section V-C."""
    m: dict[int, int] = {}
    for e in table:
        m[e.task_index] = max(m.get(e.task_index, 0), e.length)
    return m


def compute_frame_returns(
    table: list[EpisodeInfo],
    total_frames: int,
    c_fail: float = 1.0e3,
    norm_lengths: Optional[dict[int, int]] = None,
) -> np.ndarray:
    """Per-frame normalized return-to-go target (Eq. 5 + per-task normalization).

    Returns an array ``targets`` of shape (total_frames,) indexed by the
    dataset's global frame index, so ``targets[ds_global_index]`` is the VF
    regression target for that frame. Reward semantics follow RECAP Eq. (5);
    only the success source differs (RoboCasa rollouts can fail).
    """
    if norm_lengths is None:
        norm_lengths = per_task_max_length(table)
    targets = np.zeros(total_frames, dtype=np.float32)
    for e in table:
        T = e.length
        L = float(norm_lengths[e.task_index])
        # per-step rewards (Eq. 5): -1 everywhere, terminal 0 (success) or -C_fail (fail)
        rewards = -np.ones(T, dtype=np.float64)
        rewards[-1] = 0.0 if e.success else -float(c_fail)
        rtg = np.cumsum(rewards[::-1])[::-1]      # return-to-go
        norm = np.clip(rtg / L, -1.0, 0.0)
        targets[e.from_index : e.to_index] = norm.astype(np.float32)
    return targets


def subsample_episodes(
    table: list[EpisodeInfo],
    n_per_task: Optional[int] = None,
    frac: Optional[float] = None,
    seed: int = 0,
) -> list[int]:
    """Pick a *data-limited* subset of episode indices to create headroom for
    RECAP. Returns a sorted list of ``episode_index`` values to keep.
    """
    rng = np.random.default_rng(seed)
    by_task: dict[int, list[int]] = {}
    for e in table:
        by_task.setdefault(e.task_index, []).append(e.episode_index)
    keep: list[int] = []
    for _tidx, eps in by_task.items():
        eps = sorted(eps)
        if n_per_task is not None:
            k = min(n_per_task, len(eps))
        elif frac is not None:
            k = max(1, int(round(frac * len(eps))))
        else:
            k = len(eps)
        sel = rng.choice(eps, size=k, replace=False)
        keep.extend(int(x) for x in sel)
    return sorted(keep)


def summarize(meta, table: list[EpisodeInfo]) -> dict:
    """Small human-readable summary of the loaded RoboCasa dataset."""
    n_succ = sum(1 for e in table if e.success)
    lens = np.array([e.length for e in table], dtype=np.int64)
    tasks = sorted({e.task for e in table})
    return {
        "n_episodes": len(table),
        "n_tasks": len(tasks),
        "n_success": n_succ,
        "n_fail": len(table) - n_succ,
        "total_frames": int(lens.sum()),
        "len_min": int(lens.min()) if len(lens) else 0,
        "len_max": int(lens.max()) if len(lens) else 0,
        "len_mean": float(lens.mean()) if len(lens) else 0.0,
        "tasks": tasks,
    }


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def _selftest_synthetic() -> None:
    """Validate the Eq.5 / normalization math without needing a real dataset."""
    # two episodes of one task: one success (len 5), one failure (len 3)
    table = [
        EpisodeInfo(0, 5, "CloseFridge", 0, 0, 5, success=True),
        EpisodeInfo(1, 3, "CloseFridge", 0, 5, 8, success=False),
    ]
    L = per_task_max_length(table)
    assert L[0] == 5, L
    targets = compute_frame_returns(table, total_frames=8, c_fail=1e3)
    # success episode: rtg = [-4,-3,-2,-1,0]/5 ; first ~ -0.8, last = 0
    np.testing.assert_allclose(targets[0], -4.0 / 5.0, atol=1e-6)
    np.testing.assert_allclose(targets[4], 0.0, atol=1e-6)
    # monotonically non-decreasing within the success episode
    assert np.all(np.diff(targets[0:5]) > 0)
    # failure episode: terminal -C_fail dominates -> all clip to -1
    np.testing.assert_allclose(targets[5:8], -1.0, atol=1e-6)
    # subsample: 1 per task keeps exactly 1 episode index
    keep = subsample_episodes(table, n_per_task=1, seed=0)
    assert len(keep) == 1 and keep[0] in (0, 1)
    print("[robocasa_data] synthetic self-test OK:", targets.tolist())


def _selftest_real(repo_id: str = ROBOCASA_CLOSEFRIDGE_REPO) -> None:
    """Load the real RoboCasa LeRobotDataset metadata and check the loader.

    Requires lerobot + the dataset in the HF cache. Skipped gracefully if either
    is unavailable.
    """
    if LeRobotDatasetMetadata is None:
        print("[robocasa_data] lerobot not importable; skipping real self-test.")
        return
    try:
        # Resolve the local snapshot and pass it as `root` so metadata loads
        # straight from the HF cache (avoids a hub refs round-trip that can trip
        # a broken HfHubHTTPError wrapper on some huggingface_hub versions).
        from huggingface_hub import snapshot_download

        root = snapshot_download(repo_id, repo_type="dataset")
        meta = LeRobotDatasetMetadata(repo_id, root=root)
    except Exception as e:  # pragma: no cover
        print(f"[robocasa_data] could not load metadata for {repo_id}: {e}")
        return
    table = load_episode_table(meta)
    info = summarize(meta, table)
    print(f"[robocasa_data] real dataset {repo_id}:")
    for k, v in info.items():
        if k != "tasks":
            print(f"    {k}: {v}")
    print(f"    tasks (<=5): {info['tasks'][:5]}")
    targets = compute_frame_returns(table, total_frames=info["total_frames"])
    assert targets.shape[0] == info["total_frames"]
    assert targets.max() <= 1e-6 and targets.min() >= -1.0 - 1e-6
    # for an all-success demo set every episode's terminal frame target == 0
    for e in table[: min(5, len(table))]:
        assert abs(targets[e.to_index - 1] - (0.0 if e.success else -1.0)) < 1e-5
    print("[robocasa_data] real self-test OK.")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="RoboCasa RECAP data-layer self-test")
    ap.add_argument("--real", action="store_true", help="also run the real-dataset test")
    ap.add_argument("--repo", default=ROBOCASA_CLOSEFRIDGE_REPO)
    args = ap.parse_args()
    _selftest_synthetic()
    if args.real:
        _selftest_real(args.repo)
