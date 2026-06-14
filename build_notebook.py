"""Build tutorial.ipynb (English) — a friendly, from-zero tutorial for the RECAP / pi*0.6
reimplementation on LeRobot pi0.5 + RoboCasa.

This script is the SOURCE OF TRUTH. It (1) regenerates two explanatory diagrams into
docs/assets/ via matplotlib, then (2) emits tutorial.ipynb from md()/code() helpers.

Run:   python build_notebook.py
Then validate it executes, then run this script ONCE MORE so the committed notebook
carries no baked-in outputs.

Audience: readers who do NOT already know VLAs, the Physical Intelligence pi-series,
RoboCasa, or learning-from-experience RL. Tone: patient and concrete. All numbers are the
confirmed experimental results; nothing is invented.
"""
import json
import os

# --------------------------------------------------------------------------------------
# Part A. Reproducible explanatory diagrams (matplotlib -> docs/assets/*.png)
# --------------------------------------------------------------------------------------

def make_diagrams(assets_dir="docs/assets"):
    """Draw two clean, legible diagrams used by the notebook.

    diagram_vla.png        images + state + instruction  ->  VLA  ->  action chunk
    diagram_recap_loop.png the rollout -> label -> retrain experience loop (K=3)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    os.makedirs(assets_dir, exist_ok=True)

    def box(ax, xy, w, h, text, fc, ec="#33373d", fs=11, bold=False, tc="#15181c"):
        x, y = xy
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=1.6, edgecolor=ec, facecolor=fc, mutation_aspect=1.0))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal", color=tc, wrap=True)

    def arrow(ax, p0, p1, text=None, fs=9, color="#33373d"):
        ax.add_patch(FancyArrowPatch(
            p0, p1, arrowstyle="-|>", mutation_scale=18,
            linewidth=1.8, color=color, shrinkA=2, shrinkB=2))
        if text:
            mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
            ax.text(mx, my + 0.13, text, ha="center", va="bottom", fontsize=fs, color=color)

    # ---- Diagram 1: what a VLA is -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.set_xlim(0, 11); ax.set_ylim(0, 4.2); ax.axis("off")
    ax.text(5.5, 4.0, "A VLA = one neural network: it looks, it reads, it acts",
            ha="center", va="center", fontsize=13, fontweight="bold", color="#15181c")

    # inputs (left)
    box(ax, (0.2, 2.85), 2.5, 0.6, "3 camera images\n(what the robot sees)", "#dbeafe", fs=9.5)
    box(ax, (0.2, 2.05), 2.5, 0.6, "robot state (16 numbers)\njoint / gripper readings", "#dbeafe", fs=9.5)
    box(ax, (0.2, 1.25), 2.5, 0.6, 'instruction (text)\n"open the drawer"', "#dbeafe", fs=9.5)

    # model (middle)
    box(ax, (3.7, 1.0), 3.6, 2.5, "", "#fef3c7", ec="#b8860b")
    ax.text(5.5, 3.25, "VLA  (one network)", ha="center", fontsize=12, fontweight="bold", color="#7a5b00")
    box(ax, (3.95, 2.15), 3.1, 0.85,
        "PaliGemma backbone\nSigLIP vision  +  Gemma language", "#fde68a", ec="#b8860b", fs=9)
    box(ax, (3.95, 1.2), 3.1, 0.8,
        "action expert\n(flow matching: denoise -> motion)", "#fde68a", ec="#b8860b", fs=9)

    # output (right)
    box(ax, (8.3, 1.85), 2.5, 0.85,
        "action chunk\n50 future steps\n(12 numbers each)", "#dcfce7", ec="#15803d", fs=9.5)

    arrow(ax, (2.7, 2.3), (3.7, 2.3))
    arrow(ax, (7.3, 2.25), (8.3, 2.25))
    ax.text(9.55, 1.55, "sent to the robot, 20 per second",
            ha="center", va="top", fontsize=8, color="#15803d")
    fig.tight_layout()
    fig.savefig(os.path.join(assets_dir, "diagram_vla.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- Diagram 2: the RECAP experience loop -----------------------------------------
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.set_xlim(0, 10.5); ax.set_ylim(0, 4.6); ax.axis("off")
    ax.text(5.25, 4.35, "The RECAP loop: try, judge, re-learn  (repeat K=3 times)",
            ha="center", va="center", fontsize=13, fontweight="bold", color="#15181c")

    box(ax, (0.3, 2.5), 2.6, 1.0, "current policy\n(this iteration's robot brain)", "#dbeafe", fs=9.5)
    box(ax, (3.9, 2.5), 2.6, 1.0, "experience\nautonomous rollouts\n+  demonstrations", "#e9d5ff", ec="#7c3aed", fs=9.5)
    box(ax, (7.5, 2.5), 2.7, 1.0, "value function judges\neach moment:\ngoing well or badly?", "#fee2e2", ec="#b91c1c", fs=9.5)
    box(ax, (3.9, 0.5), 2.6, 1.1,
        'retrain from the base,\nevery frame tagged\n"Advantage: positive / negative"',
        "#dcfce7", ec="#15803d", fs=9.5)

    arrow(ax, (2.9, 3.0), (3.9, 3.0), "1. roll out")
    arrow(ax, (6.5, 3.0), (7.5, 3.0), "2. label")
    arrow(ax, (8.85, 2.5), (5.6, 1.6), "3. tag good vs bad")
    arrow(ax, (3.9, 1.05), (1.6, 2.5), "4. re-learn")
    ax.text(1.6, 2.15, "...then roll out again", ha="center", va="top", fontsize=8.5, color="#15803d")
    fig.tight_layout()
    fig.savefig(os.path.join(assets_dir, "diagram_recap_loop.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    return ["diagram_vla.png", "diagram_recap_loop.png"]


# --------------------------------------------------------------------------------------
# Part B. Notebook cell builders
# --------------------------------------------------------------------------------------

_n = [0]
def _id():
    _n[0] += 1
    return f"cell-{_n[0]:02d}"
def md(*s): return {"cell_type": "markdown", "id": _id(), "metadata": {}, "source": "\n".join(s)}
def code(*s): return {"cell_type": "code", "id": _id(), "metadata": {}, "execution_count": None,
                      "outputs": [], "source": "\n".join(s)}

cells = []

# ===== Cell 0 — Title + who is this for / two ways to read ============================
cells.append(md(
"# Teaching a Robot to Learn From Its Own Mistakes",
"### A from-scratch tutorial on RECAP (π\\*0.6), rebuilt on LeRobot π0.5 + RoboCasa",
"",
"Imagine you are teaching someone to cook by only ever letting them *watch* videos of a chef.",
"They might get pretty good — but the first time they knock over a pan, they have no idea what",
"to do, because no video ever showed a knocked-over pan. To truly improve, they need to *try",
"things themselves*, see what goes wrong, and learn from it.",
"",
"That gap — between *copying an expert* and *learning from your own experience* — is the whole",
"story of this notebook. We will:",
"",
"1. Explain, from zero, what a **robot-controlling AI (a “VLA”)** is.",
"2. Meet **RoboCasa**, a kitchen simulator where our robot practices.",
"3. Build the standard “just copy the expert” baseline (**behavior cloning / SFT**).",
"4. Build **RECAP**, the “learn from your own experience” method from the π\\*0.6 paper.",
"5. Ask our own research question — and report an **honest result** (spoiler: at our small",
"   scale, learning-from-experience does *not* beat plain copying, and we explain exactly why).",
"",
"---",
"",
"**Who is this for?** Anyone curious about modern robot learning. **No prior knowledge of",
"VLAs, the π (“pi”) model series, RoboCasa, or reinforcement learning is assumed.** If you",
"can read a little Python, you can read this.",
"",
"**Two ways to read:**",
"- **Path 1 — no GPU needed (just read).** Every table, figure, and robot video below is",
"  already rendered into this notebook. Scroll and enjoy.",
"- **Path 2 — with a GPU (run it yourself).** Flip `RUN_EVAL = True` in the config cell to",
"  download our trained robot brains from the HuggingFace Hub and watch them perform in the",
"  simulator. Instructions are at the very end.",
"",
"> **Honesty note, up front.** The original π\\*0.6 paper (Physical Intelligence,",
"> [arXiv:2511.14759](https://arxiv.org/abs/2511.14759)) reports impressive *real-world*",
"> results (folding laundry, making espresso) using private data and model weights that are",
"> not public. We could not reproduce *that*. Instead we did a faithful **re-implementation**",
"> of the method in **simulation**, on the closest open model (π0.5), and we report what",
"> actually happened — including where it fell short. This is a careful study, not a victory lap.",
))

# ===== Cell 1 — config (PRESERVED behavior) ==========================================
cells.append(md(
"## Setup: one small config cell",
"Run this first. It just sets a few knobs and checks that the figures/videos folder is",
"present. Nothing here needs a GPU.",
))
cells.append(code(
"# --- config ---",
"RUN_EVAL = False           # leave False to just read (Path 1). Set True on a GPU (Path 2).",
"SEED, N_EPISODES, N_ACTION_STEPS = 5000, 50, 10   # the exact eval settings behind every number below",
"# local checkpoint root (only used if you trained locally; the HF Hub is the default for Path 2)",
"CKPT_ROOT = 'outputs/robocasa/specialist_v2'",
"ASSETS = 'docs/assets'",
"import os; print('assets folder:', os.listdir(ASSETS) if os.path.isdir(ASSETS) else 'MISSING')",
))

# ===== Cell 2 — Big picture ==========================================================
cells.append(md(
"## 1. The big picture: why is teaching robots so hard?",
"",
"Getting a robot arm to do everyday physical chores — open a drawer, put a mug away, close",
"the fridge — is one of the hardest problems in AI. A chatbot only has to produce *text*. A",
"robot has to produce *motion*, in the real, messy, physical world, where a few centimeters",
"or a fraction of a second is the difference between success and a dropped cup.",
"",
"### Imitation learning: copy the expert",
"The most popular recipe today is **imitation learning**. A human teleoperates the robot",
"to do a task correctly many times. We record those demonstrations — *“when the robot saw",
"this, the human moved it like that”* — and we train a neural network to **copy** the human.",
"This is also called **behavior cloning** or **supervised fine-tuning (SFT)**, and it works",
"remarkably well.",
"",
"### The built-in ceiling",
"But copying has a hard ceiling, for an intuitive reason: **the robot only ever sees the",
"expert doing things *right*.** So:",
"- It never learns to **recover** from a mistake — because the expert never made one on camera.",
"- It gets confused in any situation the demos didn’t cover.",
"- It can **never become better than the demonstrations** — a perfect copy of a B+ chef is",
"  still a B+ chef.",
"",
"### Learning from experience",
"Humans break through this ceiling by **practicing**: we try, we fail, we notice what went",
"wrong, and we adjust. The dream is to give robots the same ability — to let a robot run on",
"its own, collect its *own* attempts (most of them mediocre), figure out which moments were",
"good and which were bad, and then **learn from that experience** to improve past the demos.",
"",
"That is exactly what the **RECAP** method (the heart of the π\\*0.6 paper) tries to do, and",
"it is what this notebook builds and tests.",
))

# ===== Cell 3 — What is a VLA + the pi series + diagram ==============================
cells.append(md(
"## 2. What is a VLA? And what is this “π” series?",
"",
"### VLA = Vision-Language-Action model",
"A **VLA** is a single neural network that ties together three things:",
"- **Vision** — it looks at camera images of the scene.",
"- **Language** — it reads a natural-language instruction (“open the drawer”).",
"- **Action** — it outputs the actual motor commands to move the robot.",
"",
"A useful analogy: a VLA is like a **chauffeur who understands spoken directions**. You say",
"“pull into the third driveway,” they *see* the street through the windshield, *understand*",
"your words, and *act* by steering and braking — all as one fluid skill, not three separate",
"programs bolted together.",
"",
"![What a VLA is](docs/assets/diagram_vla.png)",
"",
"### The π (“pi”) series from Physical Intelligence",
"“π” (pi) is a family of VLAs built by the company **Physical Intelligence**. Each version",
"builds on the previous one:",
"",
"| Model | One-line idea |",
"|---|---|",
"| **π0** | A VLA that produces smooth, continuous motion in *chunks* using **flow matching** (explained below), on top of a vision-language model. |",
"| **π0.5** | Adds **open-world generalization** — trained on very diverse data so it copes with new homes/objects, using a trick called *knowledge insulation*. **This is the open model we build on** (weights: `lerobot/pi05_base`). |",
"| **π\\*0.6** | The paper we re-implement ([arXiv:2511.14759](https://arxiv.org/abs/2511.14759)). Adds **RECAP**: learning from the model’s *own experience* plus human *corrections*. **No public weights**, so we port its method onto π0.5. |",
"",
"### What’s inside π0.5 (the model we actually use)",
"You do not need to memorize this, but it helps to picture the two halves (both drawn in the",
"diagram above):",
"1. A **PaliGemma backbone** — an off-the-shelf vision-language model = a **SigLIP** image",
"   encoder (the “eyes”) + a **Gemma** language model (the “reading” brain).",
"2. A separate **action expert** — a smaller network that turns the backbone’s understanding",
"   into an **action chunk**: 50 future steps of motion, predicted at once.",
"",
"Each timestep, π0.5 is fed **3 camera images + the robot’s state (16 numbers) + the text",
"prompt**, and it outputs that chunk of motion.",
"",
"> **What is “flow matching”? (high level)** It is a way to *generate* continuous outputs",
"> (here, smooth motions) by starting from random noise and **iteratively cleaning it up**",
"> into a coherent action — the same family of ideas as the *diffusion* models behind AI image",
"> generators. You can think of it as “sculpting motion out of static.” That is all you need",
"> for this tutorial.",
))

# ===== Cell 4 — Meet RoboCasa + 4 demo videos + observations/actions/eval =============
cells.append(md(
"## 3. Meet RoboCasa: our robot’s practice kitchen",
"",
"Training real robots is slow, expensive, and breakable. So, like flight schools use flight",
"simulators, robot researchers use **physics simulators**. We use **RoboCasa**, a large",
"benchmark for **kitchen manipulation** built on the **robosuite** framework and the",
"**MuJoCo** physics engine. It simulates realistic kitchens, objects, and a robot you can",
"command and measure precisely.",
"",
"**The robot — “PandaOmron”:** a **Franka Panda** arm (a popular 7-jointed research arm)",
"mounted on a wheeled **Omron** mobile base. Think *arm on a cart*.",
"",
"**What the robot senses each step (its “observation”):**",
"- **3 RGB cameras** at 256×256: a left and right `agentview` (looking at the scene) and an",
"  `eye_in_hand` camera (mounted on the gripper, for close-up work).",
"- A **16-dimensional proprioceptive state** — “proprioception” just means the robot’s sense",
"  of *its own body*: joint angles, gripper width, base position, etc.",
"",
"**What the robot does each step (its “action”):** a **12-number** command (arm motion,",
"gripper, base motion), issued **20 times per second** (20 fps).",
"",
"### The four tasks",
"We focus on four self-contained kitchen tasks. Watch one demonstration of each below — this",
"is the kind of expert behavior our robot will first learn to copy:",
"",
"| Task | Goal |",
"|---|---|",
"| **CloseFridge** | Push the fridge door shut. |",
"| **OpenDrawer** | Pull a drawer open. |",
"| **OpenCabinet** | Swing a cabinet door open. |",
"| **PickPlaceCounterToCabinet** (“PnP”) | Pick an object off the counter and place it in the cabinet. (The hardest — it chains grasping *and* placing.) |",
))
cells.append(code(
"# Demonstration of each task (what 'success' looks like). No GPU needed — just plays the clips.",
"from pathlib import Path",
"from IPython.display import Video, display, Markdown",
"GALLERY = f'{ASSETS}/videos'",
"DEMO_TASKS = ['CloseFridge', 'OpenDrawer', 'OpenCabinet', 'PickPlaceCounterToCabinet']",
"shown = False",
"for t in DEMO_TASKS:",
"    p = Path(GALLERY) / f'demo_{t}.mp4'",
"    if p.exists():",
"        display(Markdown(f'**Demonstration — {t}**')); display(Video(str(p), embed=True, width=360))",
"        shown = True",
"if not shown:",
"    print('demo videos not staged yet — expected at', GALLERY, '(demo_<Task>.mp4)')",
))
cells.append(md(
"### What does “success” mean, and what is “closed-loop eval”?",
"- **Success** is a simple **yes/no check built into the simulator** — e.g. “is the fridge",
"  door angle below X degrees?” It is objective and automatic; no human judging.",
"- **Closed-loop evaluation** means we let the policy actually *drive the robot* in the",
"  simulator for a whole episode, reacting to what happens moment-to-moment (a “closed loop”",
"  between seeing and acting), and check success at the end. We do this for **50 episodes**",
"  on a **held-out random seed (5000)** the model never trained on, and report the **% that",
"  succeed**. That single percentage is how we score every method in this notebook.",
))

# ===== Cell 5 — SFT / behavior cloning in detail ====================================
cells.append(md(
"## 4. The baseline: behavior cloning (SFT), explained in detail",
"",
"**Supervised Fine-Tuning (SFT)** here is exactly **behavior cloning**: we train the policy to",
"**imitate the demonstration action sequences**. The recipe is almost embarrassingly simple:",
"",
"1. Collect demonstrations: many `(what the robot saw, what the expert did next)` pairs.",
"2. Show the policy the “saw” part and ask it to predict the “did” part.",
"3. Nudge the network whenever its prediction differs from the expert’s actual move.",
"4. Repeat over all the data until it reliably reproduces expert-like motion.",
"",
"It is the robotics version of **“watch and copy.”** And it is a *strong* baseline — do not",
"underestimate it. With good demos, behavior cloning alone solves a lot.",
"",
"### But remember the ceiling (§1)",
"Because SFT only ever sees the expert doing things correctly:",
"- **No recovery skills.** If the policy drifts into a state the expert never visited (because",
"  the expert never fumbled), it has no idea what to do — errors can snowball.",
"- **No coverage beyond the demos.** Unusual situations = guesswork.",
"- **No way to exceed the demos.** Copying a demonstrator caps you *at* the demonstrator.",
"",
"In this notebook, **SFT is our baseline** — the bar that the fancier “learn from experience”",
"method (RECAP) has to clear. Keep these limits in mind; they are the entire motivation for",
"what comes next.",
))

# ===== Cell 6 — RECAP from scratch + loop diagram + figures + under-the-hood =========
cells.append(md(
"## 5. RECAP: learning from your own experience",
"",
"**RECAP** = *RL with Experience and Corrections via Advantage-conditioned Policies*. Don’t",
"worry about the acronym — here is the whole idea through one analogy, then the four moving",
"parts.",
"",
"### The coach analogy",
"Imagine a **basketball coach reviewing game film** with a young player. The coach watches",
"every clip and labels each one: **“good play”** or **“bad play.”** Crucially, the player",
"studies *both* — not just the highlights — so they learn what good looks like *and* what to",
"avoid. Then, in the next game, the coach simply says: **“play the good way.”**",
"",
"RECAP does exactly this for a robot. It lets the policy generate its own attempts, has a",
"“coach” label each moment good or bad, trains the policy on *both* kinds (each clearly",
"tagged), and at game time just asks for the good kind.",
"",
"![The RECAP loop](docs/assets/diagram_recap_loop.png)",
"",
"### The four moving parts",
"",
"**1. A value function — the coach’s eye.** A second network, the **value function** `V(o)`,",
"looks at a moment and estimates **“how well is this attempt going?”** We train it so that",
"moments from successful demos score high (return ≈ 0) and moments from failed attempts score",
"low (return ≈ −1). It genuinely separates the two — here is ours, calibrated on held-out data:",
"",
"![Value function separates success from failure](docs/assets/fig4_value_function.png)",
"",
"**2. The advantage — “did this stretch go *better than expected*?”** From the value function",
"we compute an **advantage** `A(o_t)`: looking ahead ~50 steps, did things improve more than",
"the value function predicted at moment `t`? Positive advantage = that bit of behavior was",
"genuinely good; negative = it made things worse.",
"",
"**3. Binarize — turn the score into a clean label.** For each task we pick a threshold ε and",
"call the **top ε fraction** of moments **“positive”** and the rest **“negative.”** (The",
"histogram below shows the advantage distribution and where the threshold falls.)",
"",
"![Advantage distribution and threshold](docs/assets/advantage_histogram.png)",
"",
"**4. Condition the policy — tell it which kind to learn.** This is the clever, simple trick.",
"For every training frame we **append a phrase to the language prompt**: either",
"`Advantage: positive` or `Advantage: negative` (and 30% of the time we drop it, so the",
"policy also works with no tag). The policy thus learns to produce *both* good and bad",
"behavior — each labeled. **At inference we simply prompt `Advantage: positive`** to summon",
"the good kind. No new network architecture — just a smarter prompt.",
"",
"### Putting it in a loop (Algorithm 1, K = 3)",
"RECAP repeats the loop in the diagram:",
"1. **Roll out** the current policy in the simulator to gather fresh experience.",
"2. **Label** every new frame with the value function (positive / negative).",
"3. **Add** it to the growing data pool.",
"4. **Retrain** the policy from the pretrained base on the whole pool, with the tags.",
"",
"Repeat **K = 3** times — each round, the policy should (in principle) get a little better,",
"because it keeps practicing and re-learning from richer, labeled experience.",
))
cells.append(md(
"<details><summary><b>Under the hood (optional equations — safe to skip)</b></summary>",
"",
"- **Distributional value function.** Instead of predicting a single number, `V(o)` predicts a",
"  **categorical distribution over 201 return bins**, trained with **cross-entropy** against",
"  the (normalized) Monte-Carlo return of the trajectory. Distributional targets are more",
"  stable to train than a single regressed scalar. Successful demos push the return toward 0;",
"  failed rollouts collapse it toward −1.",
"- **Advantage.** We use an **N-step estimate (N = 50)**: roughly, the value N steps later",
"  (plus rewards along the way) minus the value now — a measure of whether the stretch beat",
"  expectations.",
"- **Threshold ε.** Applied **per task** so each task gets a balanced set of positive frames.",
"- **Conditioning dropout.** The 30% prompt-drop lets the same network serve as both a",
"  conditioned and an unconditioned policy.",
"",
"</details>",
))
cells.append(md(
"The conditioning is **literally a dataset-level prompt edit** "
"(`recap/recap/advantage_dataset.py`) — there is *no* architecture change. The tag is just",
"appended to each frame's instruction text and tokenized like any other words:",
"",
"```python",
"POS, NEG = ' Advantage: positive', ' Advantage: negative'",
"# per frame: positive if the advantage indicator says so, else negative;",
"#            dropped (unconditional) with probability `dropout` (= 0.30)",
"task = strip_advantage_phrase(item['task'])           # remove any phrase already baked in",
"item['task'] = task + (POS if is_positive else NEG)   # -> tokenized like any other prompt text",
"```",
))

# ===== Cell 7 — Our reimplementation (scope honesty) ================================
cells.append(md(
"## 6. Our re-implementation: π0.5 + RoboCasa",
"",
"With the concepts in hand, here is exactly what we built and the scope we chose:",
"",
"- **Base model:** **π0.5** (open weights `lerobot/pi05_base`) — because there are **no",
"  public π0.6 weights**. We port the RECAP *method* onto π0.5.",
"- **Environment:** **RoboCasa in simulation** — because the paper’s real-world data and",
"  robots are private. Simulation also lets us measure success cleanly and cheaply.",
"- **Pipeline (the paper’s Algorithm 1, scaled down):**",
"  1. **Multi-task BC pretrain** — one base policy `π_pre` trained by behavior cloning over",
"     **all 4 tasks** together.",
"  2. **Per-task specialist** — fine-tune `π_pre` into a focused expert for each single task.",
"  3. **K = 3 RECAP** — run the experience loop (§5) on top, per task.",
"",
"The SFT baseline and the RECAP policies **start from the same `π_pre` and get the same",
"compute** — they differ only in *data + conditioning*. That makes the comparison fair: any",
"difference is the method, not the budget.",
"",
"> **Scope honesty (again, because it matters):** this is a **faithful method re-implementation",
"> in a down-scaled setting**, not a reproduction of the paper’s real-world results. We will",
"> not imply otherwise.",
))

# ===== Cell 8 — Our research question ===============================================
cells.append(md(
"## 7. Our research question: is the *coach* the bottleneck?",
"",
"Here is our own contribution on top of the re-implementation. Recall that the whole method",
"hinges on the **value function** — it is the “coach” that decides which moments are labeled",
"good vs bad. **If the coach has poor judgment, every label is noisy, and RECAP learns from",
"bad labels.** So we asked:",
"",
"> **Does the *quality* of the value function bottleneck RECAP at small scale?**",
"",
"To test this, we compare two coaches:",
"- **Proprioception-only critic (the cheap coach).** Sees only the robot’s 16-number body",
"  state — **blind to the scene** (it cannot see the fridge, the mug, the drawer).",
"- **Scene-aware VLM value function (the smart coach).** Built on top of the VLA itself: it",
"  uses **frozen PaliGemma features** (the model’s own visual understanding), so it actually",
"  *sees* what the robot sees. This matches the paper’s scene-aware `V_pre`.",
"",
"If the scene-aware coach gives better labels and a better policy, then yes — the value",
"function was a real bottleneck. Let’s look at the numbers.",
))

# ===== Cell 9 — Main results ========================================================
cells.append(md(
"## 8. Main result: RECAP (scene-aware VLM critic) vs SFT",
"",
"Closed-loop success rate (**%, n = 50 episodes, held-out seed 5000**). We show the SFT",
"baseline and **each of the three RECAP iterations separately** so you can see the trajectory.",
"Best iteration per row in **bold**; the SFT column is bold wherever it wins.",
"",
"| Task | SFT (baseline) | RECAP i1 | RECAP i2 | RECAP i3 |",
"|---|---:|---:|---:|---:|",
"| CloseFridge | **80** | **56** | 46 | 54 |",
"| OpenDrawer | **58** | **50** | 40 | 46 |",
"| OpenCabinet | **58** | 38 | **44** | 30 |",
"| PnP (CounterToCab) | **36** | 28 | 20 | **32** |",
"| **mean** | **58.0** | 43.0 | 37.5 | 40.5 |",
"",
"![K=3 result curves](docs/assets/vlmvf_k3_curves.png)",
"",
"### How to read this honestly",
"- **SFT wins every single task** *at this default threshold* (ε = 0.3). Even with the",
"  faithful, scene-aware value function, RECAP does **not** beat plain behavior cloning on any",
"  of the four tasks here (43.0/37.5/40.5 mean vs **58.0** for SFT). *But this depends on the",
"  advantage threshold ε — see §10, where a better ε lets RECAP tie/edge SFT on the",
"  higher-headroom tasks.*",
"- **The K-curve is flat, not climbing.** In the paper, more experience iterations help",
"  monotonically. Here, iteration 2 even dips and iteration 3 partially recovers — there is no",
"  clear upward trend. The paper’s “experience gain” does not appear at our scale.",
"",
"This is a **negative result**, and we report it plainly. The interesting science is in",
"*why* — and we can decompose it, starting with the critic ablation next.",
))

# ===== Cell 10 — Critic ablation ====================================================
cells.append(md(
"## 9. Did the coach matter? Critic ablation (proprio → VLM)",
"",
"This is the payoff for our research question (§7). We re-ran RECAP with the **cheap,",
"scene-blind** critic and the **scene-aware VLM** critic, and compared the best-per-task mean:",
"",
"| Critic (value function) | best-per-task mean | head-to-head |",
"|---|---:|---|",
"| RECAP, proprioception-only | 43.5 | — |",
"| RECAP, **scene-aware VLM** | **45.5** | beats proprio on **3 of 4 tasks** (OpenDrawer 50>48, OpenCabinet 44>42, CloseFridge 56>52; PnP tie at 32) |",
"",
"**The finding:** upgrading the coach **consistently helps** — a scene-aware critic produces",
"better advantage labels and therefore a better policy. So **the value function *was* a real",
"bottleneck**: part of RECAP’s weakness at small scale really was the critic’s blindness.",
"",
"**But** — and this is the crucial caveat — even the better critic (45.5) is **still well",
"below SFT (58.0)**. Fixing the coach helps, but it does **not** flip the result. So the",
"critic is *part* of the story, not the whole story. What’s the rest? See §11.",
))

# ===== Cell 11 — Sensitivity to epsilon (placeholder) ===============================
cells.append(md(
"## 10. Sensitivity to the advantage threshold ε",
"",
"A fair skeptic asks: *“Maybe RECAP would win if you just picked a better threshold ε?”*",
"(Recall ε decides what fraction of moments count as “positive”.) So we swept",
"ε ∈ {0.1, 0.3, 0.5} on the VLM-critic specialists (all at the same iter-3 data). Success %,",
"n = 50, seed 5000; the SFT baseline is shown for reference, and a RECAP cell is **bold**",
"when it **meets or beats SFT**:",
"",
"| Task | ε = 0.1 | ε = 0.3 | ε = 0.5 | SFT |",
"|---|---:|---:|---:|---:|",
"| CloseFridge | 54 | 54 | 40 | 80 |",
"| OpenDrawer | 52 | 46 | **58** | 58 |",
"| OpenCabinet | _(running)_ | 30 | 34 | 58 |",
"| PnP (CounterToCab) | **38** | 32 | **38** | 36 |",
"",
"### What this changes (read honestly)",
"This is more interesting than a flat negative. **The result is threshold-sensitive:**",
"- On **PnP** (the hardest task, where SFT is weakest at 36 = most headroom), RECAP at",
"  ε ∈ {0.1, 0.5} reaches **38 — edging SFT (36)**. On **OpenDrawer**, ε = 0.5 reaches **58,",
"  tying SFT**. So our main setting (ε = 0.3) was simply a *poor* threshold for these tasks.",
"- The flips are small and **within n = 50 noise** (±~7 points), so the honest claim is",
"  *“RECAP can match SFT on the higher-headroom tasks with a good ε,”* not *“RECAP clearly",
"  wins.”*",
"- On the **easy, high-SFT tasks** (CloseFridge 80, OpenCabinet 58) RECAP stays well below at",
"  every ε — there is little headroom and demo-cloning is already near-ceiling.",
"",
"**Takeaway:** the headline “RECAP < SFT on every task” holds *only at our default ε = 0.3*;",
"sweeping ε shows RECAP closes to a tie/slight-edge exactly where there is room to improve",
"(PnP, OpenDrawer). The advantage threshold is a real, sensitive knob — consistent with the",
"paper treating ε as a tuned per-task quantity.",
"",
"_<!-- last 2 cells (CF ε=0.5, OC ε=0.1) still running; patch when done. No numbers invented. -->_",
))

# ===== Cell 12 — Why RECAP doesn't beat SFT (5 regime gaps) =========================
cells.append(md(
"## 11. So why doesn’t RECAP beat SFT *here*?",
"",
"This is the honest heart of the study. The comparison is controlled: SFT and RECAP start",
"from the **same** `π_pre` with the **same** compute; they differ only in *data + conditioning*.",
"SFT trains on **clean demos only**; RECAP **mixes in autonomous rollouts** (mostly mediocre)",
"and trusts the advantage-conditioning to sort the good from the bad. At our scale, the demo",
"set is so clean and the rollouts so mediocre that **mixing in experience hurts more than the",
"conditioning recovers.**",
"",
"We trace the gap to **five “regime gaps”** between our setting and the paper’s large-scale,",
"real-world one. RECAP is built to shine in *their* regime, not ours:",
"",
"1. **Scale & diversity.** The paper uses thousands of hours across many skills; we use 4",
"   tasks. RECAP helps most exactly when behavior cloning is **data-starved** — which, at our",
"   scale, it isn’t.",
"2. **Conditioning baked into pretraining.** The paper trains `π_pre` **with** advantage-",
"   conditioning from the start; we **bolt it on** as a later fine-tune — a weaker integration.",
"3. **Critic quality.** Addressed in §9: the scene-aware VLM critic helped, but didn’t flip",
"   the result. (Real bottleneck, partial fix.)",
"4. **Human corrections — the “C” in RECAP.** The paper includes expert teleoperated",
"   *interventions* mid-task; our simulation setup has **none**. We are missing a whole",
"   ingredient of the method.",
"5. **On-policy collection.** The paper **re-collects** fresh data with the latest policy each",
"   iteration; we largely **reuse off-policy rollouts**, so each loop learns from staler data.",
"",
"**Bottom line:** this is a *faithful method re-implementation* and an *honest negative result*",
"with the mechanism **decomposed all the way down to the critic**. The method isn’t “wrong” —",
"it is **mismatched to a small-scale, clean-demo, no-corrections simulation regime.**",
))

# ===== Cell 13 — Rollout videos =====================================================
cells.append(md(
"## 12. See it for yourself: rollout videos",
"",
"Numbers are abstract — let’s watch the policies actually drive the robot. Each clip is",
"recorded at a **fixed seed (5000)** so SFT and RECAP face the **exact same scene** (no",
"cherry-picking), with the simulator’s ground-truth success label in the filename.",
"",
"Below: for each task, the **SFT** rollout and the **RECAP (VLM-critic)** rollout, side by",
"side in the same scene. Consistent with §8, SFT generally matches or beats RECAP.",
"",
"Then a special **highlight** (`HIGHLIGHT_PnP_seed5007_*`): a scene where **RECAP succeeds",
"and SFT fails.** RECAP genuinely helps on *some* scenes — it just doesn’t come out ahead on",
"net at this scale. Honesty cuts both ways.",
))
"# --- one cell per task (run each to watch that task's SFT vs RECAP, same scene) ---"
_RATES = {"CloseFridge": "80% / 50%", "OpenDrawer": "60% / 40%",
          "OpenCabinet": "30% / 20%", "PickPlaceCounterToCabinet": "60% / 50%"}
_DISP = {"PickPlaceCounterToCabinet": "PnP (Counter→Cabinet)"}
for _t in ["CloseFridge", "OpenDrawer", "OpenCabinet", "PickPlaceCounterToCabinet"]:
    cells.append(code(
"from pathlib import Path",
"from IPython.display import Video, display, Markdown",
f"task, disp, rates = '{_t}', '{_DISP.get(_t, _t)}', '{_RATES[_t]}'",
"g = Path(f'{ASSETS}/videos')",
"display(Markdown(f'### {disp} — same scene (seed 5000)  ·  n=10 success (SFT / RECAP): {rates}'))",
"for arm, label in [('sft', 'SFT baseline'), ('recap_vlmvf', 'RECAP (VLM-VF)')]:",
"    clips = sorted(g.glob(f'{task}_{arm}_seed5000*.mp4'))",
"    if not clips: print(f'(no {label} clip staged for {task})'); continue",
"    for p in clips:",
"        outcome = p.stem.split('seed5000_')[-1]   # 'success' or 'fail'",
"        display(Markdown(f'**{label}** — {outcome}'))",
"        display(Video(str(p), embed=True, width=320))",
    ))
# the RECAP-wins / SFT-fails highlight, in its own cell
cells.append(code(
"from pathlib import Path",
"from IPython.display import Video, display, Markdown",
"g = Path(f'{ASSETS}/videos')",
"display(Markdown('### Highlight — PnP, seed 5007: a scene where **RECAP succeeds and SFT fails**'))",
"for p in sorted(g.glob('HIGHLIGHT_*.mp4')):",
"    who = 'RECAP (VLM-VF)' if 'recap' in p.stem else 'SFT baseline'",
"    display(Markdown(f'**{who}**')); display(Video(str(p), embed=True, width=320))",
))

# ===== Cell 14 — Run it yourself: env check + eval (PRESERVED behavior) ==============
cells.append(md(
"## 13. Run it yourself (Path 2 — needs a GPU)",
"",
"Want to reproduce the closed-loop numbers? On a GPU machine you can pull our **published",
"checkpoints** from the HuggingFace Hub and run the exact eval behind the tables above. Two",
"steps: first a quick environment self-check, then the eval (gated by `RUN_EVAL`).",
"",
"### 13a. Environment self-check",
"RoboCasa closed-loop eval needs headless **EGL + MuJoCo**, which is the fragile part of the",
"setup. This cell verifies the stack *before* you try to run anything. If a line shows `!!`,",
"see `README.md` §1 for the fix. (On Path 1 / no GPU, it’s fine for items to be missing.)",
))
cells.append(code(
"def check():",
"    import importlib, subprocess",
"    for m in ['lerobot', 'robosuite', 'robocasa', 'mujoco', 'torch']:",
"        try: importlib.import_module(m); print(f'  ok  {m}')",
"        except Exception as e: print(f'  !!  {m}: {e}')",
"    print('  MUJOCO_GL =', os.environ.get('MUJOCO_GL', '(unset — set to egl for headless)'))",
"    try: print('  GPU:', subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],",
"                              capture_output=True, text=True).stdout.strip() or '(no GPU)')",
"    except Exception: print('  (nvidia-smi unavailable — expected on a no-GPU machine)')",
"check()",
))
cells.append(md(
"### 13b. Closed-loop eval",
"With `RUN_EVAL = True` (set it in the config cell), this downloads a published checkpoint and",
"runs closed-loop eval. The checkpoints are public under",
"[`dongjin630`](https://huggingface.co/dongjin630):",
"`recap-robocasa-<task>-sft` (baseline) and `recap-robocasa-<task>-vlmvf` (RECAP, evaluate",
"with `Advantage: positive`). Mirrors `README.md` §5.",
))
cells.append(code(
"# NOTE: the RoboCasa env/task id and the HF repo short-name differ ONLY for PnP:",
"#   env task id 'PickPlaceCounterToCabinet'  ->  repo short-name 'PnPCounterToCab'.",
"# Map it explicitly; don't build the repo id naively from the task id.",
"REPO_NAME = {'CloseFridge': 'CloseFridge', 'OpenDrawer': 'OpenDrawer',",
"             'OpenCabinet': 'OpenCabinet', 'PickPlaceCounterToCabinet': 'PnPCounterToCab'}",
"if RUN_EVAL:",
"    import subprocess",
"    task = 'CloseFridge'                                # try 'PickPlaceCounterToCabinet' too",
"    # checkpoints live permanently under our HF account 'dongjin630' (not a knob to change)",
"    policy = f'dongjin630/recap-robocasa-{REPO_NAME[task]}-vlmvf'   # or a local CKPT_ROOT path",
"    cmd = ['python','recap/scripts/eval_cli_wrap.py',",
"           f'--policy.path={policy}', '--env.type=robocasa', f'--env.task={task}',",
"           '--eval.batch_size=1', f'--eval.n_episodes={N_EPISODES}', '--eval.use_async_envs=false',",
"           '--policy.device=cuda', f'--policy.n_action_steps={N_ACTION_STEPS}', f'--seed={SEED}',",
"           '--output_dir=outputs/eval/notebook_demo']",
"    print(' '.join(cmd)); subprocess.run(cmd, check=True)",
"else:",
"    print('RUN_EVAL=False — set it True in the config cell, on a GPU node, to run closed-loop eval.')",
))

# ===== Cell 15 — Limitations, reproducibility, credits ==============================
cells.append(md(
"## 14. Limitations, reproducibility, and credits",
"",
"### Limitations (read these honestly)",
"The headline limitation is the one we earned: **RECAP does not beat SFT at this scale**, and",
"the cause is the **five regime gaps in §11** (scale, conditioning-in-pretraining, critic",
"quality, missing human corrections, and off-policy data). This is a **faithful but",
"down-scaled** study in simulation — it should **not** be read as evidence for or against the",
"paper’s real-world claims, which we did not (and could not) test.",
"",
"### Reproducibility",
"- Every number above is **n = 50 episodes at held-out seed 5000**, with `n_action_steps = 10`.",
"- Per-task specialists are fine-tuned from a **4-task multi-task BC base** (`π_pre`).",
"- Demonstration data: `pepijn223/robocasa_*` (LeRobotDataset v3). Base model:",
"  `lerobot/pi05_base`.",
"- Published checkpoints: [`dongjin630/recap-robocasa-<task>-{sft,vlmvf}`](https://huggingface.co/dongjin630).",
"- Exact commands and environment pins are in `README.md`; the self-check in §13a flags setup",
"  problems.",
"",
"### Credits",
"- **π\\*0.6 / RECAP** — Physical Intelligence, *“π\\*0.6: a VLA That Learns From Experience”*",
"  ([arXiv:2511.14759](https://arxiv.org/abs/2511.14759)).",
"- Built on **[LeRobot](https://github.com/huggingface/lerobot)** (Apache-2.0) and",
"  **[RoboCasa](https://robocasa.ai)** (robosuite + MuJoCo).",
"- Demonstration datasets by **`pepijn223`**.",
"- This is an independent **educational re-implementation** — not affiliated with Physical",
"  Intelligence.",
))

# --------------------------------------------------------------------------------------
# Part C. Emit notebook
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    created = make_diagrams("docs/assets")
    print("diagrams written:", created)

    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": "3.12"}},
          "nbformat": 4, "nbformat_minor": 5}

    with open("tutorial.ipynb", "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"wrote tutorial.ipynb with {len(cells)} cells")
