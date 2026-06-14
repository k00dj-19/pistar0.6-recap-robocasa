"""Shared utilities for the distributional value function (Eq.1 of the RECAP paper).

The VF discretizes normalized Monte-Carlo returns into B bins and regresses them with
cross-entropy against a two-hot target (HL-Gauss style). Only the encoding lives here;
the model itself is in `robocasa_vf.StateTaskVF`.
"""
from __future__ import annotations

import torch


def two_hot(values: torch.Tensor, bin_centers: torch.Tensor) -> torch.Tensor:
    """Two-hot encoding of scalar `values` onto `bin_centers` (sorted, uniform).

    Each scalar is distributed across its two nearest bins so that the encoded
    distribution has the exact expected value -- the standard way to "discretize
    the empirical return into B bins" for a cross-entropy target (HL-Gauss style).
    """
    v = values.clamp(bin_centers[0], bin_centers[-1])
    B = bin_centers.numel()
    # locate the right bin index
    idx = torch.bucketize(v, bin_centers)  # in [0, B]
    idx = idx.clamp(1, B - 1)
    lo = idx - 1
    hi = idx
    lo_c = bin_centers[lo]
    hi_c = bin_centers[hi]
    w_hi = (v - lo_c) / (hi_c - lo_c + 1e-12)
    w_lo = 1.0 - w_hi
    out = torch.zeros(v.shape[0], B, device=v.device, dtype=v.dtype)
    out.scatter_(1, lo.unsqueeze(1), w_lo.unsqueeze(1))
    out.scatter_(1, hi.unsqueeze(1), w_hi.unsqueeze(1))
    return out
