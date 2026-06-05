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
  --learning_rate=0.00005 \
  --rollout=128 \
  --batch_size=1024 \
  --quads_control_mode=velocity_yaw_avoid \
  --quads_velocity_xy_max=1.0 \
  --quads_velocity_z_max=1.0 \
  --quads_velocity_yaw_rate_max=12.566370614 \
  --quads_controller_obstacle_avoidance=True \
  --quads_obstacle_avoidance_distance=1.35 \
  --quads_obstacle_avoidance_max_speed=0.95 \
  --quads_obstacle_avoidance_gain=1.1 \
  --quads_obstacle_avoidance_pid_kp=1.25 \
  --quads_obstacle_avoidance_pid_ki=0.0 \
  --quads_obstacle_avoidance_pid_kd=0.10 \
  --quads_obstacle_avoidance_pid_integral_limit=1.0 \
  --quads_episode_duration=15.0 \
  --quads_use_numba=True \
  --quads_render=False
