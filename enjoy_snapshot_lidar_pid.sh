#!/usr/bin/env bash
set -euo pipefail

SNAPSHOT_DIR="${SNAPSHOT_DIR:-/home/lzh/drone/quad-swarm-rl-single-snapshot-lidar-20260529}"
PYTHON_BIN="${PYTHON_BIN:-/home/lzh/miniconda3/envs/swarm-rl/bin/python}"
CHECKPOINT_KIND="${CHECKPOINT_KIND:-latest}"
DEVICE="${DEVICE:-gpu}"
EPISODE_DURATION="${EPISODE_DURATION:-50.0}"

RUN_NAME="single_quad_obstacles_lidar_wall"
EXP_GROUP="single_obstacles_lidar_wall_"
EXP_NAME="00_single_obstacles_lidar_wall_see_0"
EXP_REL="${RUN_NAME}/${EXP_GROUP}/${EXP_NAME}"
SRC_EXP_DIR="${SNAPSHOT_DIR}/train_dir/${EXP_REL}"
PID_TRAIN_DIR="${SNAPSHOT_DIR}/train_dir_pid"
PID_GROUP_DIR="${PID_TRAIN_DIR}/${RUN_NAME}/${EXP_GROUP}"
PID_EXP_DIR="${PID_GROUP_DIR}/${EXP_NAME}"

if [ ! -f "${PID_EXP_DIR}/config.json" ]; then
  if [ ! -f "${SRC_EXP_DIR}/config.json" ]; then
    echo "No snapshot checkpoint config found."
    exit 1
  fi
  mkdir -p "$PID_GROUP_DIR"
  cp -a "$SRC_EXP_DIR" "$PID_GROUP_DIR/"
fi

cd "$SNAPSHOT_DIR"

exec "$PYTHON_BIN" -m swarm_rl.enjoy \
  --algo=APPO \
  --env=quadrotor_multi \
  --device="$DEVICE" \
  --replay_buffer_sample_prob=0 \
  --quads_use_numba=False \
  --quads_render=True \
  --visualize_projection_map=True \
  --visualize_obstacle_point_cloud=True \
  --quads_vel_penalty_limit=1.2 \
  --quads_velocity_yaw_max_speed=1.0 \
  --quads_control_type=velocity_yaw_avoid \
  --quads_episode_duration="$EPISODE_DURATION" \
  --train_dir=./train_dir_pid \
  --experiment="${EXP_REL}/" \
  --quads_view_mode=chase \
  --load_checkpoint_kind="$CHECKPOINT_KIND"
