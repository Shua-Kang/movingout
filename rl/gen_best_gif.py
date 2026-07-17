"""Roll a checkpoint several times (different torch seeds), keep the BEST
episode (highest return; early termination = full success preferred), save GIF.

Usage: python gen_best_gif.py <ckpt.pt> <det|stoch> <n_tries> <out.gif>
"""
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchrl.envs.utils import ExplorationType, set_exploration_type

from benchmarl.hydra_config import reload_experiment_from_file


def unwrap_movingout(env):
    seen, stack = set(), [env]
    while stack:
        e = stack.pop()
        if id(e) in seen:
            continue
        seen.add(id(e))
        if hasattr(e, "global_score"):
            return e
        for a in ("base_env", "_env", "env"):
            if hasattr(e, a):
                stack.append(getattr(e, a))
    return None


def main():
    ckpt, mode, n_tries, out = (sys.argv[1], sys.argv[2], int(sys.argv[3]),
                                sys.argv[4])
    exp = reload_experiment_from_file(ckpt)
    env = exp.test_env
    group = list(exp.group_map.keys())[0]
    mo = unwrap_movingout(env)
    et = (ExplorationType.DETERMINISTIC if mode == "det"
          else ExplorationType.RANDOM)

    best = None  # (score, return, frames)
    for i in range(n_tries):
        torch.manual_seed(i * 7919 + 13)
        frames = []

        def cb(e, td):
            frames.append(np.asarray(
                exp.task.__class__.render_callback(exp, e, td)))

        with set_exploration_type(et), torch.no_grad():
            td = env.rollout(max_steps=exp.max_steps, policy=exp.policy,
                             callback=cb, break_when_any_done=True,
                             auto_cast_to_device=True)
        r = td.get(("next", group, "reward")).squeeze(-1)
        ret = float(r.mean(dim=1).sum())
        score = float(mo.global_score())
        print(f"try{i}: T={len(frames)} ret={ret:.1f} score={score:.2f}",
              flush=True)
        key = (score, ret)
        if best is None or key > (best[0], best[1]):
            best = (score, ret, frames)
        if score >= 0.99:
            break

    score, ret, frames = best
    step = max(1, len(frames) // 70)
    imgs = [Image.fromarray(f) for f in frames[::step]]
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=75,
                 loop=0)
    print(f"BEST score={score:.2f} ret={ret:.1f} frames={len(frames)} -> {out}",
          flush=True)


if __name__ == "__main__":
    main()
