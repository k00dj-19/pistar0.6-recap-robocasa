"""Record per-episode showcase videos with ground-truth success labels.

For the tutorial notebook: rolls out a policy on one RoboCasa task with DETERMINISTIC
per-episode seeds (seed+ep, identical scene k across different policies) and writes
each episode as its OWN mp4 named with its auto-label:

    <out>/ep00_success.mp4, <out>/ep01_fail.mp4, ...  +  summary.json

Unlike lerobot-eval (concatenated/limited videos, no per-episode labels) and the
LeRobotDataset collector (episodes packed into chunk files), this gives directly
embeddable, labeled clips. Reuses the validated rollout loop pieces from
collect_rollouts_lerobot_robocasa.py.

Usage:
  PYTHONPATH=recap python recap/scripts/record_showcase.py \
      --ckpt <policy_dir_or_hub_id> --task CloseFridge --condition positive \
      --n_episodes 10 --seed 5000 --camera robot0_agentview_left --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

def _local_version(repo_id, version=None, *a, **k):  # noqa: ANN001
    return version or "main"
import lerobot.datasets.utils as _du            # noqa: E402
import lerobot.datasets.dataset_metadata as _dm  # noqa: E402
import lerobot.datasets.lerobot_dataset as _dl   # noqa: E402
for _m in (_du, _dm, _dl):
    if hasattr(_m, "get_safe_version"):
        _m.get_safe_version = _local_version

import imageio.v3 as iio  # noqa: E402
import numpy as np  # noqa: E402
from lerobot.envs import make_env, make_env_pre_post_processors, preprocess_observation  # noqa: E402
from lerobot.envs.factory import make_env_config  # noqa: E402
from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # noqa: E402

ADV_POS = " Advantage: positive"
FPS = 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--task", default="CloseFridge")
    ap.add_argument("--condition", choices=["none", "positive"], default="none")
    ap.add_argument("--n_episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=5000)
    ap.add_argument("--n_action_steps", type=int, default=10)
    ap.add_argument("--camera", default="robot0_agentview_left")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    policy = PI05Policy.from_pretrained(args.ckpt)
    policy.config.n_action_steps = args.n_action_steps
    policy.config.device = "cuda"
    policy.to("cuda").eval()

    env_cfg = make_env_config("robocasa", task=args.task)
    envs = make_env(env_cfg, n_envs=1)
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)
    from lerobot.policies import make_pre_post_processors
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=args.ckpt,
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )

    task_map = envs[next(iter(envs))]
    env = task_map[next(iter(task_map))]
    results = []
    for ep in range(args.n_episodes):
        policy.reset()
        obs, info = env.reset(seed=args.seed + ep)
        done = False
        success = False
        frames = []
        max_steps = env.call("_max_episode_steps")[0]
        step = 0
        while not done and step < max_steps:
            o = preprocess_observation(obs)
            try:
                tasks = list(env.call("task_description"))
            except Exception:
                tasks = [""]
            o["task"] = [t + ADV_POS for t in tasks] if args.condition == "positive" else tasks
            frames.append(np.asarray(obs["pixels"][args.camera][0]))
            o2 = pre(env_pre(o))
            action = env_post({"action": post(policy.select_action(o2))})["action"]
            a_np = action.cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(a_np)
            # Gymnasium >= 1.0 vector env: final_info is a dict of arrays masked by _final_info
            if "final_info" in info and isinstance(info["final_info"], dict) \
                    and "is_success" in info["final_info"]:
                arr = np.asarray(info["final_info"]["is_success"]).reshape(-1)
                mask = np.asarray(info.get("_final_info", [True])).reshape(-1)
                if bool(arr[0]) and bool(mask[0]):
                    success = True
            done = bool(np.logical_or(terminated, truncated)[0])
            step += 1
        tag = "success" if success else "fail"
        path = os.path.join(args.out, f"ep{ep:02d}_{tag}.mp4")
        iio.imwrite(path, np.stack(frames), plugin="pyav", fps=FPS, codec="libx264")
        results.append({"episode": ep, "seed": args.seed + ep, "success": bool(success),
                        "steps": step, "video": os.path.basename(path)})
        print(f"[showcase] ep{ep:02d} seed={args.seed+ep} {tag} ({step} steps) -> {path}", flush=True)

    sr = float(np.mean([r["success"] for r in results]))
    json.dump({"ckpt": args.ckpt, "task": args.task, "condition": args.condition,
               "seed": args.seed, "n_episodes": args.n_episodes, "success_rate": sr,
               "episodes": results}, open(os.path.join(args.out, "summary.json"), "w"), indent=2)
    print(f"[showcase] success_rate={sr:.2f} -> {args.out}/summary.json", flush=True)


if __name__ == "__main__":
    main()
