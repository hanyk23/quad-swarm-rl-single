#!/usr/bin/env bash
set -euo pipefail

SNAPSHOT_DIR="${SNAPSHOT_DIR:-/home/lzh/drone/quad-swarm-rl-single-snapshot-lidar-20260529}"
PYTHON_BIN="${PYTHON_BIN:-/home/lzh/miniconda3/envs/swarm-rl/bin/python}"
EXPERIMENT_PATH="${EXPERIMENT_PATH:-single_quad_obstacles_lidar_wall/single_obstacles_lidar_wall_/00_single_obstacles_lidar_wall_see_0}"
CHECKPOINT_KIND="${CHECKPOINT_KIND:-latest}"

if [ ! -d "$SNAPSHOT_DIR" ]; then
  echo "Snapshot repo not found: $SNAPSHOT_DIR"
  exit 1
fi

cd "$SNAPSHOT_DIR"

if [ ! -f "train_dir/${EXPERIMENT_PATH}/config.json" ]; then
  echo "No snapshot checkpoint config found:"
  echo "  $SNAPSHOT_DIR/train_dir/${EXPERIMENT_PATH}/config.json"
  echo "Train first with:"
  echo "  cd /home/lzh/drone/quad-swarm-rl-single && bash train_snapshot_lidar_obstacles.sh"
  echo "Or put the author's checkpoint directory under:"
  echo "  $SNAPSHOT_DIR/train_dir/${EXPERIMENT_PATH}/"
  exit 1
fi

exec "$PYTHON_BIN" -m swarm_rl.enjoy \
  --algo=APPO \
  --env=quadrotor_multi \
  --replay_buffer_sample_prob=0 \
  --quads_use_numba=False \
  --quads_render=True \
  --visualize_projection_map=True \
  --visualize_obstacle_point_cloud=True \
  --quads_vel_penalty_limit=2.0 \
  --quads_velocity_yaw_max_speed=2.0 \
  --train_dir=./train_dir \
  --experiment="${EXPERIMENT_PATH}/" \
  --quads_view_mode=chase \
  --load_checkpoint_kind="$CHECKPOINT_KIND"
