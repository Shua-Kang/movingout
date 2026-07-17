"""Reload a BenchMARL checkpoint and measure the TRUE task metric:
global_score (fraction of items fully inside target zones at episode end),
alongside episode return, for deterministic and stochastic rollouts.

Usage: python measure_score.py <checkpoint.pt> [n_episodes]
Prints one RESULT line per mode.
"""
import sys

import numpy as np
import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type

from benchmarl.hydra_config import reload_experiment_from_file


def unwrap_movingout(env):
    """Find the underlying MovingOutEnv inside torchrl wrappers."""
    seen = set()
    stack = [env]
    while stack:
        e = stack.pop()
        if id(e) in seen:
            continue
        seen.add(id(e))
        if hasattr(e, "global_score"):
            return e
        for attr in ("base_env", "_env", "env"):
            if hasattr(e, attr):
                stack.append(getattr(e, attr))
    return None


def main():
    ckpt = sys.argv[1]
    n_eps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    exp = reload_experiment_from_file(ckpt)
    env = exp.test_env
    group = list(exp.group_map.keys())[0]
    mo = unwrap_movingout(env)
    if hasattr(mo, 'cograb_curriculum'):
        mo.cograb_curriculum = 0.0  # eval always uses standard starts
    assert mo is not None, "could not unwrap MovingOutEnv"

    for mode, et in [("det", ExplorationType.DETERMINISTIC),
                     ("stoch", ExplorationType.RANDOM)]:
        rets, scores = [], []
        with set_exploration_type(et), torch.no_grad():
            for _ in range(n_eps):
                td = env.rollout(max_steps=exp.max_steps, policy=exp.policy,
                                 break_when_any_done=True, auto_cast_to_device=True)
                r = td.get(("next", group, "reward")).squeeze(-1)
                rets.append(float(r.mean(dim=1).sum()) if r.dim() == 2
                            else float(r.sum()))
                scores.append(float(mo.global_score()))
        print(f"RESULT mode={mode} ckpt={ckpt.split('/')[-1]} "
              f"return_mean={np.mean(rets):.2f} score_mean={np.mean(scores):.3f} "
              f"scores={['%.2f' % s for s in scores]}", flush=True)


if __name__ == "__main__":
    main()
