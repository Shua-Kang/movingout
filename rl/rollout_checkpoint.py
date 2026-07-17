"""Reload a BenchMARL checkpoint, roll out the policy, and save a GIF with a
LIVE per-step running return overlaid. Lets us see where reward is earned.

Usage:
  python rollout_checkpoint.py <checkpoint.pt> <det|stoch> <out_dir>
"""
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchrl.envs.utils import ExplorationType, set_exploration_type

from benchmarl.hydra_config import reload_experiment_from_file

try:
    import matplotlib.font_manager as _fm
    FONT = ImageFont.truetype(
        _fm.findfont("DejaVu Sans", fallback_to_default=True), 24)
except Exception:
    FONT = ImageFont.load_default()


def main():
    ckpt, mode, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)
    det = mode != "stoch"
    exp = reload_experiment_from_file(ckpt)
    env = exp.test_env
    group = list(exp.group_map.keys())[0]

    frames = []

    def cb(e, td):
        frames.append(np.asarray(exp.task.__class__.render_callback(exp, e, td)))

    et = ExplorationType.DETERMINISTIC if det else ExplorationType.RANDOM
    with set_exploration_type(et), torch.no_grad():
        td = env.rollout(max_steps=exp.max_steps, policy=exp.policy, callback=cb,
                         break_when_any_done=True, auto_cast_to_device=True)

    # per-step reward, mean over agents (matches BenchMARL episode_reward metric)
    r = td.get(("next", group, "reward")).squeeze(-1)  # (T, n_agents)
    if r.dim() == 2:
        per_step = r.mean(dim=1)  # mean over agents
    else:
        per_step = r.reshape(r.shape[0], -1).mean(dim=1)
    per_step = per_step.cpu().numpy()
    running = np.cumsum(per_step)
    total = float(running[-1])

    n = min(len(frames), len(running))
    iter_tag = os.path.basename(ckpt).replace("checkpoint_", "").replace(".pt", "")
    out_imgs = []
    for i in range(n):
        im = Image.fromarray(frames[i]).convert("RGB")
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, im.width, 84], fill=(0, 0, 0))
        d.text((8, 4), f"ckpt {iter_tag}  [{'deterministic' if det else 'stochastic'}]",
               fill=(255, 255, 255), font=FONT)
        d.text((8, 32), f"step {i+1}/{n}", fill=(200, 200, 200), font=FONT)
        d.text((8, 56), f"running return: {running[i]:8.2f}", fill=(120, 255, 120), font=FONT)
        out_imgs.append(im)

    step = max(1, len(out_imgs) // 80)
    out_imgs = out_imgs[::step]
    gif = os.path.join(out_dir, f"rollout_{iter_tag}_{'det' if det else 'stoch'}.gif")
    out_imgs[0].save(gif, save_all=True, append_images=out_imgs[1:], duration=70, loop=0)
    print(f"RESULT mode={'det' if det else 'stoch'} ckpt={iter_tag} "
          f"steps={n} total_return(mean-over-agents)={total:.2f} -> {gif}", flush=True)


if __name__ == "__main__":
    main()
