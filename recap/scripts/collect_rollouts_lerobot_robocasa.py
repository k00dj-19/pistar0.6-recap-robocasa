"""RECAP factor 5 — collect autonomous RoboCasa rollouts into a LeRobotDataset.

Sibling of `collect_rollouts_robocasa.py`. That script saves only states/actions/
success to a flat `rollouts.npz`. THIS script additionally records the three camera
images per frame and writes a full **autonomous LeRobotDataset** (images + state +
action + task string), so the failure frames (with pixels) can later be concatenated
with the human demos via `MultiLeRobotDataset` and used to train an
advantage-conditioned policy on RECAP experience data.

The created dataset's `meta/info.json` features mirror the demo schema
(`.lerobot/pepijn223/robocasa_CloseFridge`): three video cameras
`observation.images.robot0_eye_in_hand` / `robot0_agentview_left` /
`robot0_agentview_right` (each 256x256x3), `observation.state` (16),
`action` (12), task string, fps=20.

Because LeRobotDataset has no native per-episode success field, we ALSO write a side
file `<out>/episode_success.npz` mapping `episode_index -> bool success`, auto-labeled
from the env exactly as the flat collector does. Overall success_rate is printed.

Everything else (the offline get_safe_version monkeypatch, policy load, env build,
env/policy pre/post processors, the rollout loop, and the Gymnasium>=1.0
final_info success-extraction block) is copied verbatim from
`collect_rollouts_robocasa.py` and must stay in lockstep with it.
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch


def _local_version(repo_id, version=None, *a, **k):
    return version or "main"
import lerobot.datasets.utils as _du
import lerobot.datasets.dataset_metadata as _dm
import lerobot.datasets.lerobot_dataset as _dl
for _m in (_du, _dm, _dl):
    if hasattr(_m, "get_safe_version"):
        _m.get_safe_version = _local_version

from lerobot.envs import make_env, make_env_pre_post_processors, preprocess_observation
from lerobot.envs.factory import make_env_config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

ADV_POS = " Advantage: positive"
FPS = 20
IMG_HW = 256
STATE_DIM = 16
ACTION_DIM = 12


def build_features(camera_keys):
    """LeRobotDataset feature spec matching the CloseFridge demo schema.

    Shapes are tuples (NOT lists): add_frame validates the frame's numpy
    `shape` (a tuple) against `feature["shape"]` with `!=`, which would always
    fail against a JSON-style list. State/action are float64 to match the demo
    so the rollout dataset concatenates cleanly with the demos.
    """
    feats = {}
    for cam in camera_keys:
        feats[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": (IMG_HW, IMG_HW, 3),
            "names": ["height", "width", "channel"],
        }
    feats["observation.state"] = {"dtype": "float64", "shape": (STATE_DIM,), "names": None}
    feats["action"] = {"dtype": "float64", "shape": (ACTION_DIM,), "names": None}
    return feats


@torch.no_grad()
def collect_task(env, policy, env_pre, env_post, pre, post, condition,
                 n_episodes, seed, dataset, camera_keys):
    """Roll out one task's env; write each accepted episode to `dataset`.

    Returns the per-episode success labels (in dataset episode order) for this task.
    The proprio-state / action / success bookkeeping is identical to the flat
    collector; we additionally buffer the three camera frames per step and, once an
    episode finishes, push it into the LeRobotDataset via add_frame + save_episode.
    """
    ep_success = []
    ep = 0
    base_proprio_key = "observation.state"
    while ep < n_episodes:
        policy.reset()
        obs, info = env.reset(seed=seed + ep)
        n = env.num_envs
        done = np.array([False] * n)
        success = np.zeros(n, dtype=bool)
        max_steps = env.call("_max_episode_steps")[0]
        buf_s = [[] for _ in range(n)]
        buf_a = [[] for _ in range(n)]
        buf_img = [{cam: [] for cam in camera_keys} for _ in range(n)]
        buf_task = [[] for _ in range(n)]
        step = 0
        while not np.all(done) and step < max_steps:
            o = preprocess_observation(obs)
            try:
                tasks = list(env.call("task_description"))
            except Exception:
                tasks = [""] * n
            # CRITICAL: condition the POLICY INPUT only — the dataset must store the RAW
            # task string. Baking " Advantage: positive" into the stored task corrupted the
            # iter1/iter2 rollout sets: downstream RECAP training appended a 2nd phrase on
            # top ("... Advantage: positive Advantage: negative"), and on the 30% dropout
            # branch failure frames were trained as positive. (Code-review CRITICAL finding.)
            policy_tasks = [t + ADV_POS for t in tasks] if condition == "positive" else tasks
            o["task"] = policy_tasks
            # record raw proprio state before policy processors
            st = o.get(base_proprio_key)
            # raw env images (HWC uint8, one array per camera, batched over envs)
            pixels = obs["pixels"]
            o2 = env_pre(o)
            o2 = pre(o2)
            action = policy.select_action(o2)
            action = post(action)
            action = env_post({"action": action})["action"]
            a_np = action.cpu().numpy()
            st_np = st.cpu().numpy() if hasattr(st, "cpu") else np.asarray(st)
            for i in range(n):
                if not done[i]:
                    buf_s[i].append(st_np[i]); buf_a[i].append(a_np[i])
                    buf_task[i].append(tasks[i])
                    for cam in camera_keys:
                        buf_img[i][cam].append(np.asarray(pixels[cam][i]))
            obs, reward, terminated, truncated, info = env.step(a_np)
            step_done = np.logical_or(terminated, truncated)
            # Success extraction mirrors lerobot_eval.py:202-218. In Gymnasium >= 1.0
            # vector envs, info["final_info"] is a DICT of arrays (not a list of per-env
            # dicts), valid only for envs that just terminated (masked by "_final_info").
            # Iterating the dict yields key strings -> the old fi.get(...) crashed with
            # "'str' object has no attribute 'get'".
            succ_step = np.zeros(n, dtype=bool)
            if "final_info" in info and isinstance(info["final_info"], dict) \
                    and "is_success" in info["final_info"]:
                arr = np.asarray(info["final_info"]["is_success"]).reshape(-1)
                mask = np.asarray(info["_final_info"]).reshape(-1) if "_final_info" in info \
                    else np.ones(n, dtype=bool)
                succ_step = arr.astype(bool) & mask.astype(bool)
            elif "is_success" in info:
                succ_step = np.asarray(info["is_success"]).reshape(-1).astype(bool)
            for i in range(n):
                if not done[i] and i < len(succ_step) and succ_step[i]:
                    success[i] = True
            done = np.logical_or(done, step_done)
            step += 1
        for i in range(n):
            if ep >= n_episodes:
                break
            T = len(buf_s[i])
            if T == 0:
                continue
            for t in range(T):
                frame = {
                    "observation.state": np.asarray(buf_s[i][t], dtype=np.float64).reshape(STATE_DIM),
                    "action": np.asarray(buf_a[i][t], dtype=np.float64).reshape(ACTION_DIM),
                    "task": buf_task[i][t],
                }
                for cam in camera_keys:
                    img = np.asarray(buf_img[i][cam][t])
                    if img.dtype != np.uint8:
                        img = img.astype(np.uint8)
                    frame[f"observation.images.{cam}"] = img
                dataset.add_frame(frame)
            dataset.save_episode()
            ep_success.append(bool(success[i]))
            ep += 1
    return ep_success


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="policy dir (RECAP-finetuned pi05 or patched baseline)")
    ap.add_argument("--task", default="CloseFridge", help="RoboCasa --env.task (name / comma-list / group)")
    ap.add_argument("--n_episodes", type=int, default=20, help="per task")
    ap.add_argument("--condition", choices=["none", "positive"], default="positive")
    ap.add_argument("--n_envs", type=int, default=1, help="RoboCasa kitchens are RAM-heavy; keep small")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--dataset_repo", default="pepijn223/robocasa_CloseFridge",
                    help="dataset whose normalization stats rebuild the pi05 processors")
    ap.add_argument("--dataset_root", default=None,
                    help="local snapshot dir for --dataset_repo (offline); else snapshot_download")
    ap.add_argument("--repo_id", default=None,
                    help="LeRobotDataset repo_id; default local/robocasa_rollouts_<task>")
    ap.add_argument("--out", required=True, help="output dir (must not already exist) for the LeRobotDataset")
    ap.add_argument("--n_action_steps", type=int, default=10,
                    help="actions executed per chunk before replanning. Default 10 to MATCH the "
                         "eval-time setting that produced our success numbers (pi05 config default "
                         "is 50, which changes closed-loop behavior).")
    args = ap.parse_args()

    repo_id = args.repo_id or f"local/robocasa_rollouts_{args.task}"
    if os.path.exists(args.out):
        raise SystemExit(f"--out already exists: {args.out} (LeRobotDataset.create needs a fresh dir)")

    policy = PI05Policy.from_pretrained(args.ckpt)
    policy.config.n_action_steps = args.n_action_steps  # match eval-time replanning cadence
    policy.config.device = "cuda"; policy.to("cuda").eval()
    print(f"[collect] n_action_steps={policy.config.n_action_steps}", flush=True)

    env_cfg = make_env_config("robocasa", task=args.task)
    envs = make_env(env_cfg, n_envs=args.n_envs)  # {task_name: {0: vec_env}}
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)

    # CRITICAL: load the policy processors SAVED WITH THE CHECKPOINT (its training-time
    # normalizer stats), NOT a rebuild from a single dataset's meta.stats. The multi-task
    # model was trained with stats AGGREGATED across all tasks; rebuilding from one task's
    # stats gives wrong action-unnormalize -> garbage actions (wrong unnormalize stats give
    # garbage actions and SR=0). Mirrors lerobot-eval.
    from lerobot.policies import make_pre_post_processors
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=args.ckpt,
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )

    # Camera keys come straight from the env's raw pixel dict (verbatim RoboCasa
    # names robot0_agentview_left / robot0_eye_in_hand / robot0_agentview_right),
    # which match the demo schema. Discover them from one reset before create().
    first_task_map = envs[next(iter(envs))]
    probe_env = first_task_map[next(iter(first_task_map))]
    probe_obs, _ = probe_env.reset(seed=args.seed)
    camera_keys = list(probe_obs["pixels"].keys())

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        features=build_features(camera_keys),
        root=args.out,
        robot_type="PandaOmron",
        use_videos=True,
        video_backend="pyav",
    )

    all_succ = []
    ep_task_index = []
    for tindex, (task_name, task_map) in enumerate(envs.items()):
        env = task_map[next(iter(task_map))]
        su = collect_task(env, policy, env_pre, env_post, pre, post,
                          args.condition, args.n_episodes, args.seed, dataset, camera_keys)
        all_succ.extend(su)
        ep_task_index.extend([tindex] * len(su))
        sr = float(np.mean(su)) if len(su) else 0.0
        print(f"[collect] task {task_name}: episodes={len(su)} success_rate={sr:.2f}", flush=True)

    dataset.finalize()

    ep_success = np.asarray(all_succ, dtype=bool)
    episode_index = np.arange(len(ep_success), dtype=np.int64)
    np.savez(os.path.join(args.out, "episode_success.npz"),
             episode_index=episode_index, success=ep_success,
             task_index=np.asarray(ep_task_index, dtype=np.int64))
    overall = float(ep_success.mean()) if len(ep_success) else 0.0
    print(f"[collect] OVERALL success_rate={overall:.3f}  episodes={len(ep_success)} "
          f"n_fail={int((~ep_success).sum())} -> {args.out}", flush=True)
    json.dump({"task": args.task, "condition": args.condition, "repo_id": repo_id,
               "overall_success_rate": overall, "n_episodes": int(len(ep_success)),
               "n_fail": int((~ep_success).sum())},
              open(os.path.join(args.out, "summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
