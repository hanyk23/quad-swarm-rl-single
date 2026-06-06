#!/bin/bash

# ==============================================================================
# 单无人机模拟激光雷达避障训练脚本
# ==============================================================================

set -e
cd "$(dirname "$0")"

MODE="${1:-resume}"
case "${MODE}" in
  resume|--resume)
    STAGES=("stage2:resume")
    MODE_LABEL="续训阶段二"
    ;;
  retrain|--retrain|restart|--restart|overwrite|--overwrite)
    STAGES=("stage1:overwrite" "stage2:resume")
    MODE_LABEL="两阶段重训"
    ;;
  stage1|--stage1)
    STAGES=("stage1:resume")
    MODE_LABEL="只运行阶段一"
    ;;
  stage1-retrain|--stage1-retrain)
    STAGES=("stage1:overwrite")
    MODE_LABEL="重训阶段一"
    ;;
  stage2|--stage2)
    STAGES=("stage2:resume")
    MODE_LABEL="只运行阶段二"
    ;;
  two-stage|--two-stage)
    STAGES=("stage1:resume" "stage2:resume")
    MODE_LABEL="两阶段续训"
    ;;
  -h|--help|help)
    echo "用法: bash train_single_quad_obstacles_lidar.sh [resume|retrain|stage1|stage1-retrain|stage2|two-stage]"
    echo "  resume          默认，继续阶段二 checkpoint"
    echo "  retrain         阶段一覆盖旧目录重训，然后阶段二从阶段一 checkpoint 续训"
    echo "  stage1          只续训阶段一"
    echo "  stage1-retrain  只覆盖重训阶段一"
    echo "  stage2          只续训阶段二"
    echo "  two-stage       阶段一续训完成后，阶段二续训"
    exit 0
    ;;
  *)
    echo "未知模式: ${MODE}"
    echo "用法: bash train_single_quad_obstacles_lidar.sh [resume|retrain|stage1|stage1-retrain|stage2|two-stage]"
    exit 2
    ;;
esac

echo "开始训练单无人机模拟激光雷达避障任务 (${MODE_LABEL})..."

EXPERIMENT_CONFIG="train_dir/single_quad_obstacles_lidar_body_v2/single_obstacles_lidar_body_v2_/00_single_obstacles_lidar_body_v2_see_0/config.json"

for STAGE_SPEC in "${STAGES[@]}"; do
    export QUAD_LIDAR_STAGE="${STAGE_SPEC%%:*}"
    export QUAD_LIDAR_RESTART_BEHAVIOR="${STAGE_SPEC##*:}"

    if [ "${QUAD_LIDAR_RESTART_BEHAVIOR}" = "resume" ] && [ ! -f "${EXPERIMENT_CONFIG}" ]; then
        echo "找不到 body_v2 checkpoint，不能直接续训 ${QUAD_LIDAR_STAGE}。"
        echo "观测坐标系已经改变，请先运行：bash train_single_quad_obstacles_lidar.sh retrain"
        exit 1
    fi

    echo "启动 ${QUAD_LIDAR_STAGE} (restart_behavior=${QUAD_LIDAR_RESTART_BEHAVIOR})..."
    python -m sample_factory.launcher.run \
        --run=swarm_rl.runs.single_quad.single_quad_obstacles_lidar \
        --max_parallel=1 \
        --pause_between=1 \
        --experiments_per_gpu=1 \
        --num_gpus=1
done

echo "训练启动完成！"
