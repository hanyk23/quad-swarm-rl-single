#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/lzh/miniconda3/envs/swarm-rl/bin/python}"
MODEL_DIR="${MODEL_DIR:-${PROJECT_DIR}/quad-swarm-packaged-models}"
EXPERIMENT="${EXPERIMENT:-paper_single_quad_lidar_pid_best}"
CHECKPOINT_KIND="${CHECKPOINT_KIND:-best}"
DEVICE="${DEVICE:-gpu}"
EPISODE_DURATION="${EPISODE_DURATION:-50.0}"
MAX_NUM_EPISODES="${MAX_NUM_EPISODES:-10}"
RENDER="${RENDER:-True}"
NO_RENDER="${NO_RENDER:-False}"

EXTRA_ARGS=()
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
  --quads_control_mode=velocity_yaw_avoid \
  --quads_velocity_xy_max=1.0 \
  --quads_velocity_z_max=1.0 \
  --quads_velocity_yaw_rate_max=12.566370614 \
  --quads_controller_obstacle_avoidance=True \
  --quads_obstacle_avoidance_distance=1.35 \
  --quads_obstacle_avoidance_max_speed=0.95 \
  --quads_obstacle_avoidance_gain=1.1 \
  --quads_obstacle_avoidance_pid_kp=1.25 \
  --quads_obstacle_avoidance_pid_ki=0.0 \
  --quads_obstacle_avoidance_pid_kd=0.10 \
  --quads_obstacle_avoidance_pid_integral_limit=1.0 \
  --quads_goal_ball_capture_assist=False \
  --quads_episode_duration="$EPISODE_DURATION" \
  --quads_use_numba=False \
  --quads_render="$RENDER" \
  --replay_buffer_sample_prob=0 \
  --quads_view_mode=chase \
  "${EXTRA_ARGS[@]}"
