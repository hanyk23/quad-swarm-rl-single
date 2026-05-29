#!/bin/bash

# ==============================================================================
# 单无人机避障测试脚本
# ==============================================================================

# 确保脚本发生错误时退出
set -e

# 进入项目根目录
cd "$(dirname "$0")"

echo "开始测试单无人机避障任务..."

# 运行 enjoy.py 进行可视化测试
python -m swarm_rl.enjoy \
  --algo=APPO \
  --env=quadrotor_multi \
  --replay_buffer_sample_prob=0 \
  --quads_use_numba=False \
  --quads_render=True \
  --train_dir=./train_dir \
  --experiment=single_quad_obstacles_wall/single_obstacles_wall_/00_single_obstacles_wall_see_0/ \
  --quads_view_mode=chase \
  --load_checkpoint_kind=latest

echo "测试结束！"
