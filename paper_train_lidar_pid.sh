#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/lzh/miniconda3/envs/swarm-rl/bin/python}"
MODEL_SOURCE="${MODEL_SOURCE:-${PROJECT_DIR}/quad-swarm-packaged-models/paper_single_quad_lidar_pid_best}"
TRAIN_DIR="${TRAIN_DIR:-${PROJECT_DIR}/train_dir_paper_lidar_pid}"
EXPERIMENT="${EXPERIMENT:-paper_single_quad_lidar_pid_finetune}"
TRAIN_FOR_ENV_STEPS="${TRAIN_FOR_ENV_STEPS:-12000000}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NUM_ENVS_PER_WORKER="${NUM_ENVS_PER_WORKER:-4}"
DEVICE="${DEVICE:-gpu}"
SPEED_MAX="${SPEED_MAX:-2.10}"
CONTROL_TYPE="${CONTROL_TYPE:-velocity_yaw_avoid}"
AVOID_RADIUS="${AVOID_RADIUS:-1.35}"
AVOID_GAIN="${AVOID_GAIN:-1.60}"
AVOID_PID_KP="${AVOID_PID_KP:-2.40}"
AVOID_PID_KI="${AVOID_PID_KI:-0.02}"
AVOID_PID_KD="${AVOID_PID_KD:-0.55}"
AVOID_INTEGRAL_LIMIT="${AVOID_INTEGRAL_LIMIT:-1.20}"
AVOID_MAX_BIAS="${AVOID_MAX_BIAS:-2.30}"

if [ ! -f "${MODEL_SOURCE}/config.json" ]; then
  echo "Model source not found: ${MODEL_SOURCE}/config.json"
  exit 1
fi

if [ ! -f "${TRAIN_DIR}/${EXPERIMENT}/config.json" ]; then
  mkdir -p "$TRAIN_DIR"
  cp -a "$MODEL_SOURCE" "${TRAIN_DIR}/${EXPERIMENT}"
fi

exec "$PYTHON_BIN" -m swarm_rl.train \
  --algo=APPO \
  --env=quadrotor_multi \
  --device="$DEVICE" \
  --train_dir="$TRAIN_DIR" \
  --experiment="$EXPERIMENT" \
  --restart_behavior=resume \
  --load_checkpoint_kind=latest \
  --train_for_env_steps="$TRAIN_FOR_ENV_STEPS" \
  --num_workers="$NUM_WORKERS" \
  --num_envs_per_worker="$NUM_ENVS_PER_WORKER" \
  --quads_use_numba=True \
  --quads_velocity_yaw_max_speed="$SPEED_MAX" \
  --quads_control_type="$CONTROL_TYPE" \
  --quads_velocity_yaw_avoid_radius="$AVOID_RADIUS" \
  --quads_velocity_yaw_avoid_gain="$AVOID_GAIN" \
  --quads_velocity_yaw_avoid_pid_kp="$AVOID_PID_KP" \
  --quads_velocity_yaw_avoid_pid_ki="$AVOID_PID_KI" \
  --quads_velocity_yaw_avoid_pid_kd="$AVOID_PID_KD" \
  --quads_velocity_yaw_avoid_integral_limit="$AVOID_INTEGRAL_LIMIT" \
  --quads_velocity_yaw_avoid_max_bias="$AVOID_MAX_BIAS"
