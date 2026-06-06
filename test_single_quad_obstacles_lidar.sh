#!/bin/bash

# ==============================================================================
# 单无人机模拟激光雷达避障测试脚本
# ==============================================================================

set -e
cd "$(dirname "$0")"

echo "开始测试单无人机模拟激光雷达避障任务..."

EXPERIMENT_PATH="single_quad_obstacles_lidar_body_v2/single_obstacles_lidar_body_v2_/00_single_obstacles_lidar_body_v2_see_0/"
CHECKPOINT_KIND="${1:-latest}"

case "${CHECKPOINT_KIND}" in
  latest|best)
    ;;
  -h|--help|help)
    echo "用法: bash test_single_quad_obstacles_lidar.sh [latest|best]"
    echo "  latest  默认，测试最新 checkpoint"
    echo "  best    测试 reward 最好的 checkpoint"
    exit 0
    ;;
  *)
    echo "未知 checkpoint 类型: ${CHECKPOINT_KIND}"
    echo "用法: bash test_single_quad_obstacles_lidar.sh [latest|best]"
    exit 2
    ;;
esac

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
  --quads_episode_duration=45.0 \
  --visualize_projection_map=True \
  --visualize_obstacle_point_cloud=True \
  --quads_vel_penalty_limit=1.6 \
  --quads_velocity_yaw_max_speed=1.2 \
  --train_dir=./train_dir \
  --experiment="${EXPERIMENT_PATH}" \
  --quads_view_mode=chase \
  --load_checkpoint_kind="${CHECKPOINT_KIND}"

echo "测试结束！"
