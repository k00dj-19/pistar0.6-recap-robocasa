"""Final K=3 specialist curve figure for the submission report.

Per task: RECAP specialist success rate over iterations (1->2->3) vs the fixed SFT
specialist baseline (horizontal line). All points n=50, held-out seed 5000.
Eval dirs are resolved by Slurm job id (robust to renames).

  PYTHONPATH=recap python recap/scripts/plot_specialist_curves.py
  -> outputs/robocasa/specialist_v2/curves.png + curves_table.md
"""
from __future__ import annotations
import glob, json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TASKS = ["CloseFridge", "OpenDrawer", "OpenCabinet", "PickPlaceCounterToCabinet"]
SHORT = {"PickPlaceCounterToCabinet": "PnP(CounterToCab)"}
# eval job ids (n=50, seed5000): task -> {sft, iter1, iter2, iter3}
JOBS = {
    "CloseFridge":               {"sft": 14940, "iter1": 14941, "iter2": 14953, "iter3": 14987},
    "OpenDrawer":                {"sft": 14942, "iter1": 14943, "iter2": 14956, "iter3": 14990},
    "OpenCabinet":               {"sft": 14944, "iter1": 14945, "iter2": 14959, "iter3": 14993},
    "PickPlaceCounterToCabinet": {"sft": 14946, "iter1": 14947, "iter2": 14962, "iter3": 14996},
}


def pc(jobid: int):
    hits = glob.glob(os.path.join(ROOT, f"outputs/eval/robocasa_*_{jobid}/eval_info.json"))
    if not hits:
        return None
    return json.load(open(hits[0]))["overall"]["pc_success"]


def main():
    data = {t: {k: pc(j) for k, j in JOBS[t].items()} for t in TASKS}
    iters = [1, 2, 3]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), sharey=True)
    for ax, t in zip(axes, TASKS):
        d = data[t]
        ys = [d[f"iter{k}"] for k in iters]
        ax.plot(iters, ys, "o-", color="crimson", lw=2, ms=7, label="RECAP specialist")
        if d["sft"] is not None:
            ax.axhline(d["sft"], color="steelblue", ls="--", lw=2, label=f"SFT specialist ({d['sft']:.0f}%)")
        ax.set_title(SHORT.get(t, t)); ax.set_xticks(iters)
        ax.set_xlabel("RECAP iteration k"); ax.grid(alpha=0.3); ax.set_ylim(0, 100)
    axes[0].set_ylabel("success rate (%)  [n=50, held-out seed]")
    axes[0].legend(loc="upper left", fontsize=9)
    fig.suptitle("Per-task specialist: RECAP iteration curve vs SFT baseline (RoboCasa, π0.5)", y=1.02)
    fig.tight_layout()
    out = os.path.join(ROOT, "outputs/robocasa/specialist_v2/curves.png")
    fig.savefig(out, dpi=140, bbox_inches="tight"); print("wrote", out)

    # markdown table for the report
    rows = ["| Task | SFT | RECAP iter1 | iter2 | iter3 |", "|---|---:|---:|---:|---:|"]
    means = {k: [] for k in ["sft", "iter1", "iter2", "iter3"]}
    for t in TASKS:
        d = data[t]
        cells = [f"{d[k]:.0f}" if d[k] is not None else "—" for k in ["sft", "iter1", "iter2", "iter3"]]
        for k in means:
            if d[k] is not None:
                means[k].append(d[k])
        rows.append(f"| {SHORT.get(t, t)} | " + " | ".join(cells) + " |")
    rows.append("| **mean** | " + " | ".join(
        f"**{sum(v)/len(v):.1f}**" if v else "—" for v in means.values()) + " |")
    tbl = "\n".join(rows)
    out_md = os.path.join(ROOT, "outputs/robocasa/specialist_v2/curves_table.md")
    open(out_md, "w").write(tbl + "\n"); print("wrote", out_md); print(tbl)


if __name__ == "__main__":
    main()
