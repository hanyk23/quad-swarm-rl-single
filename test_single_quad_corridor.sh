#!/bin/bash

# ==============================================================================
# 两阶段长条形场地测试脚本
# ==============================================================================

set -e
cd "$(dirname "$0")"

echo "开始测试两阶段长条形场地模型 (v7)..."

EXPERIMENT_V7="single_quad_corridor_curriculum_v7/single_corridor_curriculum_v7_/00_single_corridor_curriculum_v7_see_0/"
EXPERIMENT_V6="single_quad_corridor_curriculum_v6/single_corridor_curriculum_v6_/00_single_corridor_curriculum_v6_see_0/"
EXPERIMENT_V5="single_quad_corridor_curriculum_v5/single_corridor_curriculum_v5_/00_single_corridor_curriculum_v5_see_0/"
EXPERIMENT_V4="single_quad_corridor_curriculum_v4/single_corridor_curriculum_v4_/00_single_corridor_curriculum_v4_see_0/"

if [ -f "train_dir/${EXPERIMENT_V7}/config.json" ]; then
  EXPERIMENT_PATH="${EXPERIMENT_V7}"
elif [ -f "train_dir/${EXPERIMENT_V6}/config.json" ]; then
  EXPERIMENT_PATH="${EXPERIMENT_V6}"
elif [ -f "train_dir/${EXPERIMENT_V5}/config.json" ]; then
  EXPERIMENT_PATH="${EXPERIMENT_V5}"
elif [ -f "train_dir/${EXPERIMENT_V4}/config.json" ]; then
  EXPERIMENT_PATH="${EXPERIMENT_V4}"
else
  echo "找不到可用的实验目录：v7、v6、v5 和 v4 都不存在。请先训练，或者检查 train_dir。"
  exit 1
fi

python -m swarm_rl.enjoy \
  --algo=APPO \
  --env=quadrotor_multi \
  --replay_buffer_sample_prob=0 \
  --quads_use_numba=False \
  --quads_render=True \
  --visualize_projection_map=True \
  --train_dir=./train_dir \
  --experiment="${EXPERIMENT_PATH}" \
  --quads_view_mode=chase \
  --load_checkpoint_kind=latest

echo "测试结束！"
