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

### 1. Train MLP

```bash
cd behavior_cloning
python train_mlp.py
```

### 2. Train Diffusion Policy (DP)

*Coming soon*

### 3. Train BASS Method

*Coming soon*

---

## Evaluation

```bash
cd behavior_cloning
python evaluate_models.py --robot_1_modal_path model2.pt --robot_2_modal_path model2.pt
```

Results are saved in the `outputs/` folder.

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
