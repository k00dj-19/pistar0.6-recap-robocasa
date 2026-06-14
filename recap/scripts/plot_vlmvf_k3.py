"""Final figure: VLM-VF (scene-aware critic) vs orig-RECAP (proprio critic) vs SFT, per task,
across K=1..3 iterations. RoboCasa per-task specialists, n=50, seed5000. Hardcoded from the
completed 12/12 run (see docs/PROGRESS_ROBOCASA_ITERATION.md)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TASKS = ["PickPlaceCounterToCabinet", "OpenDrawer", "OpenCabinet", "CloseFridge"]
SHORT = {"PickPlaceCounterToCabinet": "PnP(CtoCab)"}
SFT = {"PickPlaceCounterToCabinet": 36, "OpenDrawer": 58, "OpenCabinet": 58, "CloseFridge": 80}
ORIG = {"PickPlaceCounterToCabinet": [32, 28, 28], "OpenDrawer": [48, 42, 46],
        "OpenCabinet": [36, 42, 26], "CloseFridge": [40, 52, 50]}
VLM = {"PickPlaceCounterToCabinet": [28, 20, 32], "OpenDrawer": [50, 40, 46],
       "OpenCabinet": [38, 44, 30], "CloseFridge": [56, 46, 54]}
K = [1, 2, 3]

fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), sharey=True)
for ax, t in zip(axes, TASKS):
    ax.axhline(SFT[t], color="k", ls="--", lw=1.4, label=f"SFT={SFT[t]}")
    ax.plot(K, ORIG[t], "o-", color="#999999", label="RECAP (proprio VF)")
    ax.plot(K, VLM[t], "s-", color="#2b7bba", lw=2, label="RECAP (VLM VF)")
    ax.set_title(SHORT.get(t, t)); ax.set_xlabel("iteration K"); ax.set_xticks(K)
    ax.set_ylim(0, 90); ax.grid(alpha=0.3)
axes[0].set_ylabel("success rate (%, n=50)")
axes[0].legend(fontsize=8, loc="upper left")
fig.suptitle("RoboCasa per-task specialists: scene-aware (VLM) VF improves RECAP over proprio VF, "
             "but neither beats SFT (demo-purity) at this scale", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = "outputs/robocasa/specialist_v2/vlmvf_k3_curves.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=140)
print(f"[save] {out}")

# means
for name, D in [("SFT", {t: [SFT[t]] for t in TASKS}), ("orig-RECAP", ORIG), ("VLM-VF", VLM)]:
    best = sum(max(D[t]) for t in TASKS) / 4
    print(f"{name}: best-per-task mean = {best:.1f}")
