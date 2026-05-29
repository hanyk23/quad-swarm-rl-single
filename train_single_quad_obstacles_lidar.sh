#!/bin/bash

# ==============================================================================
# 单无人机模拟激光雷达避障训练脚本
# ==============================================================================

set -e
cd "$(dirname "$0")"

echo "开始训练单无人机模拟激光雷达避障任务..."

python -m sample_factory.launcher.run \
    --run=swarm_rl.runs.single_quad.single_quad_obstacles_lidar \
    --max_parallel=1 \
    --pause_between=1 \
    --experiments_per_gpu=1 \
    --num_gpus=1

echo "训练启动完成！"
