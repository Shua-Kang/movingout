"""Reload a BenchMARL checkpoint, roll out the policy, and save an fps=1 video
with per-agent ACTION + REWARD + running return burned into a top banner, so it
can be stepped through frame by frame.

Usage:
  python rollout_detailed.py <checkpoint.pt> <det|stoch> <out_path_no_ext>
Action dims (movingout): [move_mag(-1..1), angle(-1..1 ->*pi), hold(>0 = grab toggle)]
"""
import os
import sys

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchrl.envs.utils import ExplorationType, set_exploration_type

from benchmarl.hydra_config import reload_experiment_from_file

BANNER = 132
try:
    import matplotlib.font_manager as _fm
    F = ImageFont.truetype(
        _fm.findfont("DejaVu Sans Mono", fallback_to_default=True), 17)
except Exception:
    F = ImageFont.load_default()


def hold(x):
    return "GRAB" if x > 0 else "rel "


def main():
    ckpt, mode, out = sys.argv[1], sys.argv[2], sys.argv[3]
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

    act = td.get((group, "action")).cpu().numpy()         # (T, n_agents, 3)
    rew = td.get(("next", group, "reward")).squeeze(-1).cpu().numpy()  # (T, n_agents)
    per_step = rew.mean(axis=1)                            # mean over agents
    running = np.cumsum(per_step)
    n = min(len(frames), len(act), len(rew))
    iter_tag = os.path.basename(ckpt).replace("checkpoint_", "").replace(".pt", "")

    out_frames = []
    for i in range(n):
        a0, a1 = act[i, 0], act[i, 1]
        r0, r1 = rew[i, 0], rew[i, 1]
        canvas = Image.new("RGB", (512, 512 + BANNER), (0, 0, 0))
        canvas.paste(Image.fromarray(frames[i]).convert("RGB").resize((512, 512)), (0, BANNER))
        d = ImageDraw.Draw(canvas)
        lines = [
            f"ckpt {iter_tag} [{'det' if det else 'stoch'}]   step {i+1:3d}/{n}",
            f"cumulative return = {running[i]:.2f}",
            f"agent0: move={a0[0]:+.2f} ang={a0[1]:+.2f} hold={hold(a0[2])}",
            f"agent1: move={a1[0]:+.2f} ang={a1[1]:+.2f} hold={hold(a1[2])}",
            f"reward: a0={r0:+.3f} a1={r1:+.3f} mean={per_step[i]:+.3f}",
        ]
        for j, t in enumerate(lines):
            d.text((6, 4 + j * 25), t, fill=(255, 255, 255), font=F)
        out_frames.append(np.asarray(canvas))

    mp4 = out + ".mp4"
    imageio.mimwrite(mp4, out_frames, fps=1, codec="libx264",
                     output_params=["-pix_fmt", "yuv420p"])
    print(f"WROTE {mp4}  ({n} frames @1fps, total_return={running[-1]:.2f})", flush=True)


if __name__ == "__main__":
    main()
