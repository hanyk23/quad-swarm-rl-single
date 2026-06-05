#!/bin/bash

# ==============================================================================
# 单无人机模拟激光雷达避障测试脚本
# ==============================================================================

set -e
cd "$(dirname "$0")"

echo "开始测试单无人机模拟激光雷达避障任务..."

EXPERIMENT_PATH="single_quad_obstacles_lidar_wall/single_obstacles_lidar_wall_/00_single_obstacles_lidar_wall_see_0/"

if [ ! -f "train_dir/${EXPERIMENT_PATH}/config.json" ]; then
  echo "找不到实验目录：train_dir/${EXPERIMENT_PATH}"
  echo "请先运行 bash train_single_quad_obstacles_lidar.sh 完成训练，或者检查 train_dir。"
  exit 1
fi

python -m swarm_rl.enjoy \
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
  --experiment="${EXPERIMENT_PATH}" \
  --quads_view_mode=chase \
  --load_checkpoint_kind=latest

echo "测试结束！"
