#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/lzh/miniconda3/envs/swarm-rl/bin/python}"
MODEL_DIR="${MODEL_DIR:-${PROJECT_DIR}/quad-swarm-packaged-models}"
EXPERIMENT="${EXPERIMENT:-paper_single_quad_lidar_pid_best}"
CHECKPOINT_KIND="${CHECKPOINT_KIND:-latest}"
DEVICE="${DEVICE:-gpu}"
EPISODE_DURATION="${EPISODE_DURATION:-50.0}"
MAX_NUM_EPISODES="${MAX_NUM_EPISODES:-10}"
RENDER="${RENDER:-True}"
NO_RENDER="${NO_RENDER:-False}"

EXTRA_ARGS=()
if [[ "$RENDER" == "True" || "$RENDER" == "true" || "$RENDER" == "1" ]]; then
  EXTRA_ARGS+=(--quads_render=True)
fi
if [[ "$NO_RENDER" == "True" || "$NO_RENDER" == "true" || "$NO_RENDER" == "1" ]]; then
  EXTRA_ARGS+=(--no_render)
fi

exec "$PYTHON_BIN" -m swarm_rl.enjoy \
  --algo=APPO \
  --env=quadrotor_multi \
  --device="$DEVICE" \
  --train_dir="$MODEL_DIR" \
  --experiment="${EXPERIMENT}/" \
  --load_checkpoint_kind="$CHECKPOINT_KIND" \
  --max_num_episodes="$MAX_NUM_EPISODES" \
  --quads_episode_duration="$EPISODE_DURATION" \
  --quads_use_numba=False \
  --replay_buffer_sample_prob=0 \
  --quads_view_mode=chase \
  "${EXTRA_ARGS[@]}"
