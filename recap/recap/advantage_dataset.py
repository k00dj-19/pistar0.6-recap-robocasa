"""Advantage-conditioned dataset wrapper for π0.5 (RECAP factor 5).

RECAP conditions the policy on a binarized advantage indicator by adding the text
"Advantage: positive" / "Advantage: negative" to the model input (Section V-B). π0.5
already consumes a language prompt built from the per-frame `task` string
(`processor_pi05.Pi05PrepareStateTokenizerProcessorStep`), so the cleanest, surgery-free
injection is to append the advantage phrase to each frame's `task` here, at the dataset
level. This flows through collation -> prompt -> tokens automatically.

Modes:
  - "recap":   inject "Advantage: positive/negative" per the precomputed indicator,
               with `dropout` probability of dropping the phrase (unconditional) so the
               model learns both conditional and unconditional distributions -> enables
               β=1 sampling (condition positive) and CFG β>1 (Appendix E).
  - "sft":     no injection (plain supervised finetuning baseline = behavior cloning).
  - "awr" / "filtered": no injection (handled by the trainer via sample weighting /
               filtering); wrapper still exposes the per-frame advantage for them.

At inference the policy is prompted with "Advantage: positive" (β=1) or CFG-combined.
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch


POS_PHRASE = " Advantage: positive"
NEG_PHRASE = " Advantage: negative"
# Matches one-or-more trailing advantage phrases. Early rollout collectors BAKED
# " Advantage: positive" into the stored task string (code-review CRITICAL finding);
# stripping here repairs those datasets at training time without re-collection.
import re as _re
_ADV_SUFFIX = _re.compile(r"(\s*Advantage:\s*(positive|negative))+\s*$")


def strip_advantage_phrase(task: str) -> str:
    """Remove any trailing 'Advantage: positive/negative' phrase(s) from a task string."""
    return _ADV_SUFFIX.sub("", task)


class AdvantageConditionedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base,                       # a LeRobotDataset (already episode-subset if desired)
        i_dense: np.ndarray,        # bool indicator per global frame index
        adv_dense: Optional[np.ndarray] = None,  # float advantage per global frame index
        mode: str = "recap",
        dropout: float = 0.30,
        task_key: str = "task",
        index_key: str = "index",
        index_by_position: bool = False,
        fail_dense: Optional[np.ndarray] = None,  # bool per frame: belongs to a FAILURE episode
        seed: int = 0,
    ):
        assert mode in ("recap", "sft", "awr", "filtered")
        self.base = base
        self.i_dense = i_dense
        self.adv_dense = adv_dense
        self.mode = mode
        self.dropout = dropout
        self.task_key = task_key
        self.index_key = index_key
        # index_by_position: look up indicators by the dataset POSITION i (0..len-1) instead
        # of item["index"]. Required for MultiLeRobotDataset, whose per-sub-dataset "index"
        # fields collide — there i_dense/adv_dense are built positionally aligned to the
        # combined frame order, so i is the correct key. Single LeRobotDataset uses item["index"].
        self.index_by_position = index_by_position
        # fail_dense: when provided, frames of FAILURE episodes are ALWAYS conditioned
        # "Advantage: negative" and never fall into the unconditional dropout branch —
        # otherwise 30% of failure-frame gradients are plain unlabeled BC on bad actions,
        # polluting the marginal the policy samples from at inference.
        self.fail_dense = fail_dense
        self._rng = random.Random(seed)

    def __len__(self):
        return len(self.base)

    def positive_indices(self) -> list[int]:
        """Global-frame positions whose advantage is positive (for filtered-BC)."""
        return np.nonzero(self.i_dense)[0].tolist()

    def __getitem__(self, i):
        item = self.base[i]
        if self.index_by_position:
            g = int(i)  # combined position == indicator index (MultiLeRobotDataset)
        else:
            g = int(item[self.index_key]) if self.index_key in item else None
        pos = bool(self.i_dense[g]) if (g is not None and g < len(self.i_dense)) else False

        # ALWAYS strip any baked-in advantage phrase first, in EVERY mode (repairs rollout
        # datasets collected with the phrase baked into the stored task string): sft/filtered/
        # awr must train on raw task strings; recap re-attaches exactly ONE phrase below.
        raw_task = strip_advantage_phrase(item[self.task_key])
        item[self.task_key] = raw_task
        if self.mode == "recap":
            is_fail = bool(self.fail_dense[g]) if (
                self.fail_dense is not None and g is not None and g < len(self.fail_dense)) else False
            if is_fail:
                # failure frames: always labeled negative, never unconditional
                item[self.task_key] = raw_task + NEG_PHRASE
            elif self._rng.random() >= self.dropout:  # keep conditioning (1 - dropout)
                item[self.task_key] = raw_task + (POS_PHRASE if pos else NEG_PHRASE)
            else:
                item[self.task_key] = raw_task  # unconditional sample
        # expose advantage scalar for weighting baselines
        if self.adv_dense is not None and g is not None and g < len(self.adv_dense):
            a = self.adv_dense[g]
            item["advantage"] = torch.tensor(0.0 if np.isnan(a) else float(a))
            item["is_positive"] = torch.tensor(float(pos))
        return item


def make_inference_prompt(task: str, positive: bool = True) -> str:
    """Build the conditioned task string used at eval time (β=1 -> positive)."""
    return task + (POS_PHRASE if positive else NEG_PHRASE)
