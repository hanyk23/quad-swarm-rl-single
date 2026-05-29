#!/bin/bash

set -e
cd "$(dirname "$0")"

echo "开始第一阶段训练：v7 长条形空旷场地..."
echo "阶段一已启用较小 reward_scale，以减少 critic loss 抖动。"
python -m sample_factory.launcher.run \
  --run=swarm_rl.runs.single_quad.single_quad_corridor_stage1 \
  --max_parallel=1 --pause_between=1 \
  --experiments_per_gpu=1 --num_gpus=1

echo "第一阶段完成，开始第二阶段训练：v7 长条形障碍场地..."
python -m sample_factory.launcher.run \
  --run=swarm_rl.runs.single_quad.single_quad_corridor_stage2 \
  --max_parallel=1 --pause_between=1 \
  --experiments_per_gpu=1 --num_gpus=1

echo "第二阶段已启动。"
