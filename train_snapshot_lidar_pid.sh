#!/usr/bin/env bash
set -euo pipefail

SNAPSHOT_DIR="${SNAPSHOT_DIR:-/home/lzh/drone/quad-swarm-rl-single-snapshot-lidar-20260529}"
PYTHON_BIN="${PYTHON_BIN:-/home/lzh/miniconda3/envs/swarm-rl/bin/python}"
TRAIN_FOR_ENV_STEPS="${TRAIN_FOR_ENV_STEPS:-12000000}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NUM_ENVS_PER_WORKER="${NUM_ENVS_PER_WORKER:-4}"
DEVICE="${DEVICE:-gpu}"

RUN_NAME="single_quad_obstacles_lidar_wall"
EXP_GROUP="single_obstacles_lidar_wall_"
EXP_NAME="00_single_obstacles_lidar_wall_see_0"
EXP_REL="${RUN_NAME}/${EXP_GROUP}/${EXP_NAME}"
SRC_EXP_DIR="${SNAPSHOT_DIR}/train_dir/${EXP_REL}"
PID_TRAIN_DIR="${SNAPSHOT_DIR}/train_dir_pid"
PID_GROUP_DIR="${PID_TRAIN_DIR}/${RUN_NAME}/${EXP_GROUP}"
PID_EXP_DIR="${PID_GROUP_DIR}/${EXP_NAME}"

if [ ! -f "${SRC_EXP_DIR}/config.json" ]; then
  echo "Original snapshot model not found: ${SRC_EXP_DIR}/config.json"
  exit 1
fi

if [ ! -f "${PID_EXP_DIR}/config.json" ]; then
  mkdir -p "$PID_GROUP_DIR"
  cp -a "$SRC_EXP_DIR" "$PID_GROUP_DIR/"
fi

cd "$SNAPSHOT_DIR"

exec "$PYTHON_BIN" -m swarm_rl.train \
  --env=quadrotor_multi \
  --device="$DEVICE" \
  --train_for_env_steps="$TRAIN_FOR_ENV_STEPS" \
  --algo=APPO \
  --use_rnn=False \
  --num_workers="$NUM_WORKERS" \
  --num_envs_per_worker="$NUM_ENVS_PER_WORKER" \
  --learning_rate=0.00005 \
  --ppo_clip_value=5.0 \
  --recurrence=1 \
  --nonlinearity=tanh \
  --actor_critic_share_weights=False \
  --policy_initialization=xavier_uniform \
  --adaptive_stddev=False \
  --with_vtrace=False \
  --max_policy_lag=100000000 \
  --rnn_size=256 \
  --with_pbt=False \
  --gae_lambda=1.00 \
  --max_grad_norm=5.0 \
  --exploration_loss_coeff=0.0 \
  --rollout=128 \
  --batch_size=1024 \
  --quads_use_numba=True \
  --quads_num_agents=1 \
  --quads_mode=o_random \
  --quads_episode_duration=15.0 \
  --quads_neighbor_encoder_type=no_encoder \
  --quads_neighbor_hidden_size=0 \
  --quads_neighbor_obs_type=none \
  --quads_neighbor_visible_num=0 \
  --replay_buffer_sample_prob=0.75 \
  --anneal_collision_steps=300000000 \
  --normalize_input=False \
  --normalize_returns=False \
  --reward_clip=10.0 \
  --save_milestones_sec=3600 \
  --quads_collision_reward=8.0 \
  --quads_collision_smooth_max_penalty=6.0 \
  --quads_obst_collision_reward=5.0 \
  --quads_floor_stall_reward=10.0 \
  --quads_room_floor_reward=10.0 \
  --quads_room_wall_reward=10.0 \
  --quads_room_ceiling_reward=10.0 \
  --quads_orient_reward=2.0 \
  --quads_spin_reward=0.5 \
  --quads_z_reward=0.5 \
  --quads_stable_z_reward=0.5 \
  --quads_stable_spin_reward=0.5 \
  --quads_use_obstacles=True \
  --quads_room_dims 12 12 10 \
  --quads_obst_spawn_area 10 10 \
  --quads_obst_density=0.18 \
  --quads_obst_size=0.5 \
  --quads_obstacle_obs_type=lidar \
  --quads_obstacle_scan_resolution=0.15 \
  --quads_use_downwash=False \
  --quads_vel_penalty_limit=1.2 \
  --quads_velocity_yaw_max_speed=1.0 \
  --quads_progress_reward=0.8 \
  --quads_success_reward=1.0 \
  --quads_first_success_reward=10.0 \
  --quads_vel_reward=0.4 \
  --quads_obs_repr=xyz_vxyz_R_omega_wall \
  --quads_control_type=velocity_yaw_avoid \
  --train_dir=./train_dir_pid/single_quad_obstacles_lidar_wall/single_obstacles_lidar_wall_ \
  --experiment=00_single_obstacles_lidar_wall_see_0
