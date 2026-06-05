#!/bin/bash

# ==============================================================================
# 单无人机避障训练脚本
# ==============================================================================

# 确保脚本发生错误时退出
set -e

# 进入项目根目录
cd "$(dirname "$0")"

echo "开始训练单无人机避障任务..."

# 运行 sample_factory 启动器，加载 swarm_rl/runs/single_quad/single_quad_obstacles.py 中的配置
python -m sample_factory.launcher.run \
    --run=swarm_rl.runs.single_quad.single_quad_obstacles \
    --max_parallel=1 \
    --pause_between=1 \
    --experiments_per_gpu=1 \
    --num_gpus=1

echo "训练启动完成！"
