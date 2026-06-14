"""Thin wrapper around lerobot-eval's eval_main (validated against a known-good reference checkpoint)
that optionally injects RECAP advantage conditioning by appending "Advantage: <cond>" to the
task before the pi05 prompt is built. Avoids my custom rollout (which had a SR=0 bug).

Usage (argv after the script are passed straight to lerobot-eval's draccus parser):
  ADV_COND=positive python recap/scripts/eval_cli_wrap.py --policy.path=outputs/pi05_recap \
      --env.type=robocasa --env.task=CloseFridge --eval.n_episodes=10 --output_dir=...
"""
import os
import sys

cond = os.environ.get("ADV_COND", "").strip()
if cond:
    from lerobot.types import TransitionKey
    import lerobot.policies.pi05.processor_pi05 as P

    _orig = P.Pi05PrepareStateTokenizerProcessorStep.__call__

    def _patched(self, transition):
        cd = transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}
        tasks = cd.get(self.task_key)
        if tasks is not None:
            cd[self.task_key] = [f"{t} Advantage: {cond}" for t in tasks]
        return _orig(self, transition)

    P.Pi05PrepareStateTokenizerProcessorStep.__call__ = _patched
    print(f"[wrap] RECAP advantage conditioning injected into task prompt: '{cond}'", flush=True)
else:
    print("[wrap] no advantage conditioning (plain eval)", flush=True)

from lerobot.scripts.lerobot_eval import eval_main  # noqa: E402

if __name__ == "__main__":
    eval_main()
