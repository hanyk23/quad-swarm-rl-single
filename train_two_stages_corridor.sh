#!/bin/bash

# ==============================================================================
# 两阶段避障训练脚本 - 长条形场地
# ==============================================================================

set -e
cd "$(dirname "$0")"

echo "开始第一阶段训练：长条形空旷场地 (v7, 无障碍物)..."
# 注意：可以在此修改第一阶段的训练时长（--train_for_env_steps）
python -m sample_factory.launcher.run \
    --run=swarm_rl.runs.single_quad.single_quad_corridor_stage1 \
    --max_parallel=1 --pause_between=1 \
    --experiments_per_gpu=1 --num_gpus=1

echo "第一阶段训练完成！"
echo "开始第二阶段训练：长条形复杂场地 (v7, 带障碍物)，将自动续接第一阶段最新 checkpoint..."

python -m sample_factory.launcher.run \
    --run=swarm_rl.runs.single_quad.single_quad_corridor_stage2 \
    --max_parallel=1 --pause_between=1 \
    --experiments_per_gpu=1 --num_gpus=1

echo "第二阶段训练已启动，使用的是第一阶段同一实验目录中的最新 checkpoint。"
