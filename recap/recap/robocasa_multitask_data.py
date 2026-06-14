"""Multi-task RECAP data layer: combine DEMO + autonomous ROLLOUT LeRobotDatasets
into a single positionally-aligned (state, task_index, return) view.

This is the data backbone that lets the RECAP value function learn from *failure*
episodes. Demos (``pepijn223/robocasa_<Task>``) are all successful; autonomous
rollouts (``local/robocasa_rollouts_<Task>_s<seed>``) mix successes and failures
(success labels live in a side file ``episode_success.npz``). Training the VF on
both makes the advantage encode *action quality*, not just speed.

Reuse, don't reimplement
------------------------
The reward / normalization math (Eq. 5, per-task normalization) comes straight
from ``robocasa_data`` (``compute_frame_returns`` / ``per_task_max_length``) and
the parquet state reader from ``robocasa_vf`` (``load_states_from_parquet``). This
module only does the *combination + global task-id assignment + positional
alignment* on top of those primitives.

Positional alignment (the whole point)
--------------------------------------
The downstream RECAP policy trains a ``MultiLeRobotDataset`` over a FIXED ordered
list of repos = ``[demo repos...] + [rollout repos...]``. Its ``__getitem__``
concatenates frames in repo-list order (repo0 frames ``[0..n0)``, repo1
``[n0..n0+n1)``, ...) and, within a repo, in ``LeRobotDataset`` frame order
(episodes contiguous, ascending global frame ``index``).

``load_states_from_parquet`` returns each repo's rows sorted by that same
per-repo global frame ``index``. So concatenating the per-repo state arrays in
repo-list order produces an array whose row ``p`` corresponds *exactly* to
``MultiLeRobotDataset[p]``. Every combined array we build (``state``,
``task_index``, ``returns``, ``gindex``) and every advantage/indicator computed
from them is therefore positionally aligned to ``MultiLeRobotDataset(repos)`` by
construction. ``offsets`` makes the per-repo block boundaries explicit.
"""

from __future__ import annotations

import dataclasses
import os
import re
from typing import Optional

import numpy as np

from .robocasa_data import (
    EpisodeInfo,
    compute_frame_returns,
    load_episode_table,
    per_task_max_length,
)
from .robocasa_vf import load_states_from_parquet

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
except Exception:  # pragma: no cover - lerobot optional at import time
    LeRobotDatasetMetadata = None


# ----------------------------------------------------------------------------- spec
@dataclasses.dataclass
class RepoSpec:
    """One dataset in the ordered combined list.

    ``root`` is the *base* root such that the dataset dir is ``root/repo_id`` --
    this matches ``MultiLeRobotDataset(repo_ids, root=root)`` which resolves each
    sub-dataset at ``root/repo_id``. Demos and rollouts must share one base root
    (the project convention is ``.lerobot``) for that single-root resolution to
    cover both.

    ``kind``: ``"demo"`` -> every episode success=True; ``"rollout"`` -> success
    read from ``episode_success.npz`` (keyed by ``episode_index``).

    ``task_name``: canonical task key used to assign a *global* task id shared by
    a task's demo and its rollouts. If None it is derived from ``repo_id``.

    ``success_npz``: override path to the success side-file (default
    ``<dataset_dir>/episode_success.npz``).
    """

    repo_id: str
    root: str
    kind: str = "demo"
    task_name: Optional[str] = None
    success_npz: Optional[str] = None

    @property
    def dataset_dir(self) -> str:
        return os.path.join(self.root, self.repo_id)


# Robocasa task names appearing in repo_ids -- demos are pepijn223/robocasa_<Task>,
# rollouts local/robocasa_rollouts_<Task>_s<seed> (or _test fixture suffix).
_ROLLOUT_PREFIX = "robocasa_rollouts_"
_DEMO_PREFIX = "robocasa_"
# Strip any trailing _iter<N>, _s<seed>, and/or _sp (specialist) bookkeeping suffixes in
# any combination (e.g. _sp_iter2_s8000, _iter1_s3000, _s2000). Repeated application peels
# stacked suffixes so a task's demo and ALL its rollouts (any policy/iteration/seed) share
# one canonical task id.
_SUFFIX = re.compile(r"(_iter\d+|_s\d+|_sp)$")


def canonical_task_name(repo_id: str) -> str:
    """Extract the canonical task name from a demo or rollout repo_id.

    pepijn223/robocasa_CloseFridge                  -> CloseFridge
    local/robocasa_rollouts_CloseFridge_s2000       -> CloseFridge
    local/robocasa_rollouts_CloseFridge_iter1_s3000 -> CloseFridge
    local/robocasa_rollouts_CloseFridge_test        -> CloseFridge

    The canonical name is what makes a task's demo and its rollouts share a task
    id, so the VF and advantage are computed per *task*, not per *dataset*.
    """
    name = repo_id.split("/")[-1]
    if name.startswith(_ROLLOUT_PREFIX):
        name = name[len(_ROLLOUT_PREFIX):]
    elif name.startswith(_DEMO_PREFIX):
        name = name[len(_DEMO_PREFIX):]
    if name.endswith("_test"):
        name = name[: -len("_test")]
    prev = None
    while prev != name:  # peel stacked suffixes: _iter1_s3000 -> _iter1 -> ""
        prev = name
        name = _SUFFIX.sub("", name)
    if name.endswith("_test"):
        name = name[: -len("_test")]
    return name


def _load_success_map(spec: RepoSpec) -> dict[int, bool]:
    """episode_index -> success bool, from a rollout's episode_success.npz."""
    path = spec.success_npz or os.path.join(spec.dataset_dir, "episode_success.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"rollout repo {spec.repo_id!r} (kind=rollout) needs a success file; "
            f"not found: {path}"
        )
    d = np.load(path)
    ei = np.asarray(d["episode_index"]).reshape(-1).astype(np.int64)
    su = np.asarray(d["success"]).reshape(-1).astype(bool)
    return {int(e): bool(s) for e, s in zip(ei, su)}


# --------------------------------------------------------------------------- result
@dataclasses.dataclass
class CombinedData:
    """Positionally-aligned combined view over the ordered repo list.

    All arrays have length ``N == sum(num_frames)`` and row ``p`` corresponds to
    ``MultiLeRobotDataset(repo_ids, root=...)[p]`` (see module docstring).
    """

    state: np.ndarray          # (N, state_dim) float32
    task_index: np.ndarray     # (N,) int64  GLOBAL task id (shared demo<->rollout)
    returns: np.ndarray        # (N,) float32  normalized MC return target (Eq.5)
    gindex: np.ndarray         # (N,) int64  combined position == np.arange(N)
    table: list[EpisodeInfo]   # episodes with COMBINED from_index/to_index + global task_index
    offsets: np.ndarray        # (n_repos + 1,) cumulative frame offsets per repo
    repo_ids: list[str]        # ordered repo list (alignment order)
    repo_num_frames: list[int] # per-repo frame counts
    task_name_to_id: dict[str, int]
    norm_lengths: dict[int, int]

    @property
    def n_frames(self) -> int:
        return len(self.state)

    @property
    def n_tasks(self) -> int:
        return len(self.task_name_to_id)


def build_combined(
    repos: list[RepoSpec],
    c_fail: float = 1.0e3,
    task_name_to_id: Optional[dict[str, int]] = None,
) -> CombinedData:
    """Build the combined frame table + states + normalized returns.

    Task ids are assigned per canonical task name (``task_name_to_id``; built in
    repo order if not supplied), so a task's demo and its rollouts share an id.
    Per-task normalization length ``L`` is the max episode length per task id over
    the COMBINED data. Returns reuse ``compute_frame_returns`` semantics: success
    frames land in (-1, 0]; any failure episode collapses to -1 after clipping.
    """
    if LeRobotDatasetMetadata is None:
        raise RuntimeError("lerobot is required to build the combined dataset")

    task_name_to_id = dict(task_name_to_id) if task_name_to_id else {}

    states: list[np.ndarray] = []
    task_idx_blocks: list[np.ndarray] = []
    combined_table: list[EpisodeInfo] = []
    repo_num_frames: list[int] = []
    offsets = [0]

    for spec in repos:
        tname = spec.task_name or canonical_task_name(spec.repo_id)
        if tname not in task_name_to_id:
            task_name_to_id[tname] = len(task_name_to_id)
        gtid = task_name_to_id[tname]

        d = spec.dataset_dir
        meta = LeRobotDatasetMetadata(spec.repo_id, root=d)
        table = load_episode_table(meta)
        data = load_states_from_parquet(f"{d}/data/chunk-*/file-*.parquet")

        n_frames = int(meta.total_frames)
        if len(data["state"]) != n_frames:
            raise AssertionError(
                f"{spec.repo_id}: parquet rows {len(data['state'])} != "
                f"meta.total_frames {n_frames}"
            )
        offset = offsets[-1]

        succ_map = _load_success_map(spec) if spec.kind == "rollout" else None

        # Per-row global task id for this repo's block (overrides parquet's local id).
        block_tidx = np.full(n_frames, gtid, dtype=np.int64)

        for e in table:
            if succ_map is not None:
                success = succ_map.get(int(e.episode_index), e.success)
            else:
                success = True  # demos are all-success
            combined_table.append(
                EpisodeInfo(
                    episode_index=e.episode_index,
                    length=e.length,
                    task=tname,
                    task_index=gtid,
                    from_index=offset + e.from_index,
                    to_index=offset + e.to_index,
                    success=success,
                )
            )

        states.append(data["state"])
        task_idx_blocks.append(block_tidx)
        repo_num_frames.append(n_frames)
        offsets.append(offset + n_frames)

    state = np.concatenate(states, 0)
    task_index = np.concatenate(task_idx_blocks, 0)
    N = len(state)
    assert N == offsets[-1], (N, offsets[-1])

    norm_lengths = per_task_max_length(combined_table)
    returns = compute_frame_returns(
        combined_table, total_frames=N, c_fail=c_fail, norm_lengths=norm_lengths
    )
    gindex = np.arange(N, dtype=np.int64)

    return CombinedData(
        state=state,
        task_index=task_index,
        returns=returns,
        gindex=gindex,
        table=combined_table,
        offsets=np.asarray(offsets, dtype=np.int64),
        repo_ids=[s.repo_id for s in repos],
        repo_num_frames=repo_num_frames,
        task_name_to_id=task_name_to_id,
        norm_lengths=norm_lengths,
    )


def verify_frame_count(combined: CombinedData) -> int:
    """Assert combined frame count == sum of per-repo num_frames.

    This sum equals ``len(MultiLeRobotDataset(combined.repo_ids, root=...))``
    (``MultiLeRobotDataset.num_frames == sum(d.num_frames)``), so this is the
    positional-alignment length contract. Returns the verified frame count.
    """
    total = int(sum(combined.repo_num_frames))
    assert combined.n_frames == total, (combined.n_frames, total)
    assert combined.n_frames == int(combined.offsets[-1])
    assert len(combined.task_index) == total
    assert len(combined.returns) == total
    assert len(combined.gindex) == total
    return total


# default demo repo list (order is part of the alignment contract)
DEMO_TASKS = [
    "CloseFridge",
    "OpenDrawer",
    "TurnOnMicrowave",
    "OpenCabinet",
    "CoffeeSetupMug",
    "PickPlaceCounterToCabinet",
    "PickPlaceSinkToCounter",
]


def default_demo_repos(root: str) -> list[RepoSpec]:
    """The 7 demo repos in the fixed RECAP order, resolved under ``root``."""
    return [
        RepoSpec(repo_id=f"pepijn223/robocasa_{t}", root=root, kind="demo")
        for t in DEMO_TASKS
    ]
