# Moving Out: Physically-Grounded Human-AI Collaboration

[Website](https://live-robotics-uva.github.io/moving_out_ai/index.html)

## Moving Out Environment

This repository provides baseline methods and tools. For the environment only, please refer to: [Env](https://github.com/live-robotics-uva/moving_out_env)

### Install the Environment

```bash
git submodule init
git submodule update
cd moving_out_env
pip install -e .
```

---

## Train Multi-Agent RL Methods

We use BenchMARL and integrate the environment into PettingZoo.

### Setup

```bash
git submodule init
git submodule update
cd rl/BenchMARL
pip install -e .
cd ../PettingZoo
pip install -e .
cd ../..
# pre-compute BFS distance caches for walled maps (once; strongly recommended)
python moving_out_env/scripts/gen_dist_cache.py
```

### Training

`scripts/train_benchmarl_eval.sh` wraps `benchmarl/run.py`; every option is
an environment variable:

```bash
MAP=Simple ALGO=ippo SHARE=True ITERS=3000 ENTROPY_COEF=0.01 \
SEED=0 RENDER=False SAVE=results/simple_ippo \
bash scripts/train_benchmarl_eval.sh
```

Key variables: `MAP` (any name from `AVAILABLE_MAPS`), `ALGO` (`ippo` |
`mappo`), `SHARE` (share policy params), `ITERS`, `ENTROPY_COEF`,
`SCALE` / `STEP_COST` (dense-reward knobs), `RESTORE` (checkpoint to
continue from), and `EXTRA_OVERRIDES` for any extra hydra override, e.g.
the optional co-op reward/curriculum features:

```bash
ML=task.dense_rewards_setting.middle_and_large
EXTRA_OVERRIDES="task.hold_mode=desired task.cograb_curriculum=0.8 \
 $ML.cohold_potential=10.0 $ML.canonical_unheld_shaping=True"
```

Evaluate a checkpoint (100 episodes, deterministic + stochastic score,
standard starts) and render rollout GIFs:

```bash
python rl/measure_score.py <checkpoint.pt> 100
python rl/gen_eval_gifs.py <checkpoint.pt> stoch 3 out/rollout
```

### Reference configs

* **Simple / most maps (IPPO, recommended):** `ALGO=ippo SHARE=True
  ENTROPY_COEF=0.01 ITERS=3000` (use `ENTROPY_COEF=0.03` on
  FourCorners / SingleRotation / CornerDecision / AdaptiveAssist).
* **Two-robot carry maps (e.g. SequentialRotations, SingleRotation with
  IPPO):** train in two stages with the co-op curriculum —
  stage 1: `EXTRA_OVERRIDES="... task.cograb_curriculum=0.8
  task.cograb_teleport_frac=0.0"` for 1500 iters; stage 2: `RESTORE=<stage-1
  ckpt>` with `task.cograb_teleport_frac=0.5` for another 1000+ iters.
* **Faster training with unchanged results:**
  `EXTRA_OVERRIDES="experiment.on_policy_n_minibatch_iters=15
  experiment.on_policy_minibatch_size=1500"` (~3.7x faster per iteration
  than the BenchMARL defaults).

---

## Train Behavior Cloning Models

Demonstrations are pulled from HuggingFace (`ShuaKang/movingout_task1` /
`..._task2`) and cached locally under `behavior_cloning/data_cache/`. Each
script trains one model per robot — robot 1's model only sees robot 1's data
and vice versa (`--robot_id both|1|2`); `--map_name` restricts training to
one map (recommended, e.g. `HandOff`). Checkpoints are written to
`behavior_cloning/weights/` with a `_robot1` / `_robot2` suffix; evaluation
outputs (scores, trajectories, videos) go to `behavior_cloning/exp_results/`.

### 1. Train MLP (deterministic)

```bash
cd behavior_cloning
python train_mlp.py --map_name HandOff --model_save_path weights/mlp_handoff.pt --epoch 300
```

### 2. Train GRU (stochastic — random-noise initial hidden state)

```bash
python train_gru.py --map_name HandOff --model_save_path weights/gru_handoff.pt --epoch 300
```

### 3. Train Diffusion Policy (stochastic)

```bash
python train_diffusion.py --map_name HandOff --model_save_path weights/dp_handoff.pt --epoch 300
```

---

## Evaluation

Any mix of architectures (`mlp` | `gru` | `dp`) can be paired:

```bash
cd behavior_cloning
python evaluate_models.py --map_name HandOff \
  --robot_1_model_path weights/mlp_handoff_robot1.pt --robot_1_arch mlp \
  --robot_2_model_path weights/mlp_handoff_robot2.pt --robot_2_arch mlp \
  --evaluation_times 3
```

Scores, trajectories, and rollout videos are saved under
`exp_results/<experiment_save_path>/<timestamp>/` (disable videos with
`--no_videos`).

---

## Other Useful Tools

### Cache Distance Computation

For maps without walls, we use Euclidean distance.
For maps with walls, we use BFS or A\*.
To speed up RL training (~20 -> ~1300 env steps/s on walled maps),
pre-compute and cache distances once:

```bash
python moving_out_env/scripts/gen_dist_cache.py            # all walled maps
python moving_out_env/scripts/gen_dist_cache.py HandOff    # a single map
```

Caches are written next to each map json (`*.distcache.npz`, gitignored).
Without a cache the environment still runs (live BFS) but much slower.

---

### Data Visualization

We host datasets on Hugging Face:

* Task 1: [https://huggingface.co/datasets/ShuaKang/movingout\_task1](https://huggingface.co/datasets/ShuaKang/movingout_task1)
* Task 2: [https://huggingface.co/datasets/ShuaKang/movingout\_task2](https://huggingface.co/datasets/ShuaKang/movingout_task2)


**Save trajectory as MP4:**

```bash
python dataset_to_video.py -f ShuaKang/movingout_task2 -m HandOff -t 4 -v video
```

**Show trajectory in popup window:**

```bash
python dataset_to_video.py -f ShuaKang/movingout_task2 -m HandOff -t 4 -v human
```
Use `-m` for map name and `-t` for trajectory ID.
