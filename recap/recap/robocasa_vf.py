"""Distributional value function for RoboCasa (RECAP Eq.1), state-based variant.

State-based variant for RoboCasa's 16-dim PandaOmron
proprio state. RECAP trains V^{pi_ref} as a categorical distribution over B=201
return bins with a cross-entropy loss against the two-hot discretized
Monte-Carlo return, and reads a scalar value V(o)=sum_b p(b) v(b) (Section IV-A).
For the demos-only pre-training stage the advantage is essentially "relative
speed / progress", so a proprio-state + task-embedding MLP is a faithful and
*cheap* critic — no image decoding over the 155k CloseFridge frames.

Self-contained on purpose (the RoboCasa track only adds robocasa_* files); it
reuses just the bin/two-hot machinery from the shared, read-only
`value_function.py`.
"""

from __future__ import annotations

import glob
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .value_function import two_hot


# ----------------------------------------------------------------------------- data
def load_states_from_parquet(data_glob: str) -> dict:
    """Read proprio state + indices directly from LeRobot parquet shards (no image
    decode). Returns dict of arrays sorted by global frame `index`.

    Columns used: observation.state, episode_index, task_index, index.
    """
    import pyarrow.parquet as pq

    files = sorted(glob.glob(data_glob))
    if not files:
        raise FileNotFoundError(f"no parquet shards matched: {data_glob}")
    states, ep_idx, task_idx, gidx = [], [], [], []
    cols = ["observation.state", "episode_index", "task_index", "index"]
    for f in files:
        t = pq.read_table(f, columns=cols).to_pydict()
        states.append(np.asarray(t["observation.state"], dtype=np.float32))
        ep_idx.append(np.asarray(t["episode_index"], dtype=np.int64))
        task_idx.append(np.asarray(t["task_index"], dtype=np.int64))
        gidx.append(np.asarray(t["index"], dtype=np.int64))
    states = np.concatenate(states, 0)
    ep_idx = np.concatenate(ep_idx, 0)
    task_idx = np.concatenate(task_idx, 0)
    gidx = np.concatenate(gidx, 0)
    order = np.argsort(gidx)
    return {
        "state": states[order],
        "episode_index": ep_idx[order],
        "task_index": task_idx[order],
        "index": gidx[order],
    }


# ---------------------------------------------------------------------------- model
class StateTaskVF(nn.Module):
    """Categorical distributional VF over [state, task_embedding] (Eq. 1)."""

    def __init__(self, state_dim: int, n_tasks: int, n_bins: int = 201,
                 v_min: float = -1.0, v_max: float = 0.0,
                 task_emb_dim: int = 32, hidden: int = 256):
        super().__init__()
        self.n_bins = n_bins
        self.register_buffer("bin_centers", torch.linspace(v_min, v_max, n_bins))
        # max(1, n_tasks) so a single-task dataset (CloseFridge) still embeds.
        self.task_emb = nn.Embedding(max(1, n_tasks), task_emb_dim)
        self.net = nn.Sequential(
            nn.Linear(state_dim + task_emb_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, n_bins),
        )

    def logits(self, state: torch.Tensor, task_index: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, self.task_emb(task_index)], dim=-1)
        return self.net(x)

    def forward(self, state: torch.Tensor, task_index: torch.Tensor) -> torch.Tensor:
        """Scalar value V(o) = sum_b p(b) v(b)."""
        probs = F.softmax(self.logits(state, task_index), dim=-1)
        return (probs * self.bin_centers).sum(-1)

    def cross_entropy_loss(self, state, task_index, target_value) -> torch.Tensor:
        """H(R^B_t, p_phi) — Eq. (1)."""
        log_probs = F.log_softmax(self.logits(state, task_index), dim=-1)
        target = two_hot(target_value, self.bin_centers)
        return -(target * log_probs).sum(-1).mean()


def train_state_vf(
    vf: StateTaskVF,
    state: np.ndarray,
    task_index: np.ndarray,
    target_value: np.ndarray,
    epochs: int = 20,
    batch_size: int = 2048,
    lr: float = 1e-3,
    device: str = "cuda",
    seed: int = 0,
    val_frac: float = 0.1,
    max_steps: Optional[int] = None,
    verbose: bool = True,
):
    """Train the distributional VF by cross-entropy (Eq. 1). Reports train/val CE
    and a regression sanity metric (mean |V - target|).

    `max_steps` caps total gradient steps (used for a quick login dry-run that
    stays under the cluster debug threshold).
    """
    rng = np.random.default_rng(seed)
    n = len(state)
    perm = rng.permutation(n)
    n_val = int(val_frac * n)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    S = torch.as_tensor(state, device=device)
    Tk = torch.as_tensor(task_index, device=device)
    Y = torch.as_tensor(target_value, dtype=torch.float32, device=device)
    vf = vf.to(device)
    opt = torch.optim.Adam(vf.parameters(), lr=lr)

    def evaluate(idx):
        vf.eval()
        with torch.no_grad():
            ce = vf.cross_entropy_loss(S[idx], Tk[idx], Y[idx]).item()
            v = vf(S[idx], Tk[idx])
            mae = (v - Y[idx]).abs().mean().item()
        return ce, mae

    hist = []
    steps = 0
    for ep in range(epochs):
        vf.train()
        p = rng.permutation(len(tr_idx))
        tot = 0.0
        for i in range(0, len(tr_idx), batch_size):
            b = tr_idx[p[i:i + batch_size]]
            loss = vf.cross_entropy_loss(S[b], Tk[b], Y[b])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
        if verbose and (ep % 2 == 0 or ep == epochs - 1):
            vce, vmae = evaluate(val_idx)
            print(f"  [VF] ep {ep:3d} step {steps:5d}  train_CE={tot/len(tr_idx):.4f}  "
                  f"val_CE={vce:.4f}  val_MAE={vmae:.4f}", flush=True)
            hist.append({"epoch": ep, "step": steps, "train_ce": tot / len(tr_idx),
                         "val_ce": vce, "val_mae": vmae})
        if max_steps is not None and steps >= max_steps:
            break
    vf.eval()
    return vf, hist
