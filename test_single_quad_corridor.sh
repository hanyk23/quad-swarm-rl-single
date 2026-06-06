#!/bin/bash

# ==============================================================================
# 两阶段长条形场地测试脚本
# ==============================================================================

set -e
cd "$(dirname "$0")"

echo "开始测试两阶段长条形场地模型 (v7)..."

EXPERIMENT_PATH="single_quad_corridor_curriculum_v7/single_corridor_curriculum_v7_/00_single_corridor_curriculum_v7_see_0/"
CHECKPOINT_KIND="${1:-latest}"

case "${CHECKPOINT_KIND}" in
  latest|best)
    ;;
  -h|--help|help)
    echo "用法: bash test_single_quad_corridor.sh [latest|best]"
    exit 0
    ;;
  *)
    echo "未知 checkpoint 类型: ${CHECKPOINT_KIND}"
    echo "用法: bash test_single_quad_corridor.sh [latest|best]"
    exit 2
    ;;
esac

if [ ! -f "train_dir/${EXPERIMENT_PATH}/config.json" ]; then
  echo "找不到实验目录：train_dir/${EXPERIMENT_PATH}"
  echo "请先运行 bash train_two_stages_corridor.sh，或者检查 train_dir。"
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
  --load_checkpoint_kind="${CHECKPOINT_KIND}"

echo "测试结束！"
