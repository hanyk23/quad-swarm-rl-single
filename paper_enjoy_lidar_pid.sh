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
SPEED_MAX="${SPEED_MAX:-0.70}"
USE_NUMBA="${USE_NUMBA:-True}"
RENDER="${RENDER:-True}"
NO_RENDER="${NO_RENDER:-False}"
VIEW_MODE="${VIEW_MODE:-chase}"
PROJECTION_MAP="${PROJECTION_MAP:-True}"
OBSTACLE_POINT_CLOUD="${OBSTACLE_POINT_CLOUD:-False}"
RENDER_SPEED="${RENDER_SPEED:-1.20}"
RENDER_WIDTH="${RENDER_WIDTH:-520}"
RENDER_HEIGHT="${RENDER_HEIGHT:-416}"
CONTROL_TYPE="${CONTROL_TYPE:-velocity_yaw_avoid}"
AVOID_RADIUS="${AVOID_RADIUS:-1.35}"
AVOID_GAIN="${AVOID_GAIN:-1.60}"
AVOID_PID_KP="${AVOID_PID_KP:-2.40}"
AVOID_PID_KI="${AVOID_PID_KI:-0.02}"
AVOID_PID_KD="${AVOID_PID_KD:-0.55}"
AVOID_INTEGRAL_LIMIT="${AVOID_INTEGRAL_LIMIT:-1.20}"
AVOID_MAX_BIAS="${AVOID_MAX_BIAS:-2.30}"

read -r -a VIEW_MODE_ARGS <<< "$VIEW_MODE"

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
  --quads_use_numba="$USE_NUMBA" \
  --replay_buffer_sample_prob=0 \
  --quads_velocity_yaw_max_speed="$SPEED_MAX" \
  --quads_control_type="$CONTROL_TYPE" \
  --quads_velocity_yaw_avoid_radius="$AVOID_RADIUS" \
  --quads_velocity_yaw_avoid_gain="$AVOID_GAIN" \
  --quads_velocity_yaw_avoid_pid_kp="$AVOID_PID_KP" \
  --quads_velocity_yaw_avoid_pid_ki="$AVOID_PID_KI" \
  --quads_velocity_yaw_avoid_pid_kd="$AVOID_PID_KD" \
  --quads_velocity_yaw_avoid_integral_limit="$AVOID_INTEGRAL_LIMIT" \
  --quads_velocity_yaw_avoid_max_bias="$AVOID_MAX_BIAS" \
  --visualize_projection_map="$PROJECTION_MAP" \
  --visualize_obstacle_point_cloud="$OBSTACLE_POINT_CLOUD" \
  --quads_render_speed="$RENDER_SPEED" \
  --quads_render_width="$RENDER_WIDTH" \
  --quads_render_height="$RENDER_HEIGHT" \
  --quads_view_mode "${VIEW_MODE_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
