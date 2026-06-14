"""Advantage estimation + binarized improvement indicator I_t for RoboCasa (App. F).

Self-contained advantage module for RoboCasa (the RoboCasa track only
adds robocasa_* files; it reuses the read-only shared modules only by pattern).

Given the trained state value function V (`robocasa_vf.StateTaskVF`), compute for
every frame the N-step-lookahead advantage
    A(o_t) = sum_{t'=t}^{t+N-1} r_t'(norm) + V(o_{t+N}) - V(o_t)
(episode-aware: lookahead clipped to the episode end, where V of the terminal
frame is the bootstrap). Rewards are the Eq.5 rewards normalized by the per-task
length L, on the same (-1,0) scale as V.

Then pick a threshold epsilon by percentile so that a target fraction of frames
have positive advantage (paper: ~30% pre-train), and set the binarized indicator
    I_t = 1[A(o_t) > epsilon].
The output is a per-(global frame index) indicator consumed by the
advantage-conditioned policy (factor 5) to inject "Advantage: positive/negative".
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .robocasa_data import EpisodeInfo, per_task_max_length
from .robocasa_vf import StateTaskVF


@torch.no_grad()
def value_all(vf: StateTaskVF, state: np.ndarray, task_index: np.ndarray,
              device: str = "cuda", batch: int = 16384) -> np.ndarray:
    vf = vf.to(device).eval()
    out = np.empty(len(state), dtype=np.float32)
    for i in range(0, len(state), batch):
        s = torch.as_tensor(state[i:i + batch], device=device)
        t = torch.as_tensor(task_index[i:i + batch], device=device)
        out[i:i + batch] = vf(s, t).cpu().numpy()
    return out


def compute_advantages(
    vf: StateTaskVF,
    state: np.ndarray,
    task_index: np.ndarray,
    gindex: np.ndarray,             # global frame index per row (sorted)
    table: list[EpisodeInfo],
    n_step: int = 50,
    device: str = "cuda",
    norm_lengths: Optional[dict[int, int]] = None,
) -> np.ndarray:
    """Per-row N-step advantage. Returns array aligned to the rows of `state`."""
    if norm_lengths is None:
        norm_lengths = per_task_max_length(table)
    V = value_all(vf, state, task_index, device=device)             # value per row
    gpos = {int(g): i for i, g in enumerate(gindex)}                 # global idx -> row
    adv = np.full(len(state), np.nan, dtype=np.float32)
    for e in table:
        L = float(norm_lengths[e.task_index])
        rows = [gpos.get(g) for g in range(e.from_index, e.to_index)]
        rows = [r for r in rows if r is not None]
        if len(rows) < 2:
            continue
        rows = np.array(rows)
        Ve = V[rows]
        T = len(rows)
        # normalized per-step reward = -1/L for every step (success terminal adds 0)
        r = -1.0 / L
        a = np.empty(T, dtype=np.float32)
        for t in range(T):
            end = min(t + n_step, T - 1)        # bootstrap index within episode
            nsteps = end - t                    # number of -1/L rewards accumulated
            disc_sum = r * nsteps
            a[t] = disc_sum + Ve[end] - Ve[t]
        # Terminal frame: lookahead is empty -> A = 0 exactly, which the top-quantile
        # threshold would label "positive" (even at the moment of failure). Exclude it
        # (NaN -> indicator False) rather than mislabel. (Code-review LOW finding.)
        a[T - 1] = np.nan
        adv[rows] = a
    return adv


def select_threshold(adv: np.ndarray, positive_fraction: float) -> float:
    """epsilon s.t. approximately `positive_fraction` of (valid) advantages exceed it."""
    valid = adv[~np.isnan(adv)]
    return float(np.quantile(valid, 1.0 - positive_fraction))


def build_indicators(
    adv: np.ndarray,
    positive_fraction: float = 0.30,
    correction_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, float]:
    """Binarized indicator I_t = 1[A > eps]; corrections forced True. NaN rows -> False."""
    eps = select_threshold(adv, positive_fraction)
    I = np.zeros(len(adv), dtype=bool)
    valid = ~np.isnan(adv)
    I[valid] = adv[valid] > eps
    if correction_mask is not None:
        I = np.logical_or(I, correction_mask)
    return I, eps


def build_indicators_per_task(
    adv: np.ndarray,
    task_index: np.ndarray,
    positive_fraction: float = 0.30,
    correction_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, dict[int, float]]:
    """Paper-faithful PER-TASK threshold ε_ℓ (App. F): label the top `positive_fraction`
    of advantages WITHIN EACH task ℓ as positive, so every task contributes the same
    fraction of positive frames regardless of its advantage scale. Returns (I, {task_id: ε_ℓ}).

    Why per-task (vs a single global ε): a high-advantage-spread task (e.g. hard PnP) would
    otherwise dominate the positive set while an easy/saturated task gets almost none.
    Per-task ε_ℓ keeps the improvement signal balanced across tasks and matches the paper's
    ε_ℓ definition. NaN rows -> False; corrections forced True.
    """
    I = np.zeros(len(adv), dtype=bool)
    eps_per_task: dict[int, float] = {}
    for t in np.unique(task_index):
        sel = (task_index == int(t)) & ~np.isnan(adv)
        if not sel.any():
            continue
        eps_t = float(np.quantile(adv[sel], 1.0 - positive_fraction))
        eps_per_task[int(t)] = eps_t
        I[sel] = adv[sel] > eps_t
    if correction_mask is not None:
        I = np.logical_or(I, correction_mask)
    return I, eps_per_task
