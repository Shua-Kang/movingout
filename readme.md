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
cd ..
bash scripts/train_map_HandOff.sh
```

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
To speed up RL training, we pre-compute and cache distances:

```bash
bash scripts/cache_distance.bash
```

Modify the script to choose specific maps if needed.

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
