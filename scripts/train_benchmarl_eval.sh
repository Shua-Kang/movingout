#!/usr/bin/env bash
# BenchMARL training launcher for MovingOut (MAPPO/IPPO) with evaluation.
# Every option is an environment variable; see the block below. Assumes the
# current python environment has benchmarl + pettingzoo + moving_out
# installed (pip install -e rl/BenchMARL rl/PettingZoo moving_out_env).
set +e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO}/rl"
[ -f "${REPO}/.venv/bin/activate" ] && source "${REPO}/.venv/bin/activate"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
# sitecustomize patch: save eval videos as .pt (torchvision.io.write_video gone)
export PYTHONPATH="${REPO}/scripts/benchmarl_patch:${PYTHONPATH:-}"

MAP="${MAP:-DistancePriority}"
SAVE="${SAVE:-${REPO}/benchmarl_results/$(echo "${MAP}" | tr '[:upper:]' '[:lower:]')_eval}"
ITERS="${ITERS:-300}"
NENVS="${NENVS:-12}"
FPB="${FPB:-6000}"
MAXCYC="${MAXCYC:-500}"
EVAL_INT="${EVAL_INT:-60000}"     # every 10 iters (must be multiple of FPB)
EVAL_EPS="${EVAL_EPS:-2}"
ENTROPY_COEF="${ENTROPY_COEF:-0.0}"   # 0.0 = BenchMARL default; 0.01 avoids entropy collapse
CKPT_INT="${CKPT_INT:-0}"             # checkpoint interval in frames (0 = off)
SCALE="${SCALE:-20.0}"                # closeness-shaping scale (drip strength)
STEP_COST="${STEP_COST:--0.01}"       # per-step cost
SEED="${SEED:-0}"                     # experiment seed
RESTORE="${RESTORE:-}"                # checkpoint .pt to continue from (optional)
ALGO="${ALGO:-mappo}"                 # mappo (central critic) | ippo (independent)
SHARE="${SHARE:-False}"               # share_policy_params (True = one shared policy)
RENDER="${RENDER:-True}"              # False = skip eval videos (each .pt is ~350MB; quota!)
TRAIN_DEV="${TRAIN_DEV:-cpu}"         # cuda:N = PPO update on GPU
SAMPLE_DEV="${SAMPLE_DEV:-cpu}"       # policy device during collection
mkdir -p "${SAVE}"

EXTRA_ARGS=()
if [[ -n "${RESTORE}" ]]; then
  EXTRA_ARGS+=("experiment.restore_file=${RESTORE}")
fi
# EXTRA_OVERRIDES: space-separated extra hydra overrides, e.g.
#   "task.dense_rewards_setting.middle_and_large.move_items_to_target_areas=60"
if [[ -n "${EXTRA_OVERRIDES:-}" ]]; then
  read -r -a _EO <<< "${EXTRA_OVERRIDES}"
  EXTRA_ARGS+=("${_EO[@]}")
fi

exec python BenchMARL/benchmarl/run.py \
  algorithm="${ALGO}" \
  algorithm.entropy_coef="${ENTROPY_COEF}" \
  task=pettingzoo/movingout \
  experiment.train_device="${TRAIN_DEV}" \
  experiment.sampling_device="${SAMPLE_DEV}" \
  experiment.buffer_device="${TRAIN_DEV}" \
  experiment.on_policy_n_envs_per_worker="${NENVS}" \
  experiment.parallel_collection=True \
  experiment.on_policy_collected_frames_per_batch="${FPB}" \
  experiment.max_n_iters="${ITERS}" \
  experiment.evaluation=True \
  experiment.render="${RENDER}" \
  experiment.evaluation_interval="${EVAL_INT}" \
  experiment.evaluation_episodes="${EVAL_EPS}" \
  experiment.evaluation_deterministic_actions=True \
  experiment.checkpoint_interval="${CKPT_INT}" \
  experiment.checkpoint_at_end=True \
  experiment.keep_checkpoints_num=3 \
  experiment.loggers="[csv]" \
  experiment.share_policy_params="${SHARE}" \
  task.max_cycles="${MAXCYC}" \
  task.map_name="${MAP}" \
  task.reward_setting="dense" \
  task.dense_rewards_setting.step_cost="${STEP_COST}" \
  task.dense_rewards_setting.small_items.scale_for_agents_get_closer_to_cloest_small_items="${SCALE}" \
  seed="${SEED}" \
  experiment.save_folder="${SAVE}" \
  "${EXTRA_ARGS[@]}"
