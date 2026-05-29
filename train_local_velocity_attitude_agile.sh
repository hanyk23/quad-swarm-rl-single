#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/lzh/miniconda3/envs/swarm-rl/bin/python}"

MODE="${1:-train}"
DEVICE="${DEVICE:-gpu}"
TRAIN_DIR="${TRAIN_DIR:-$ROOT_DIR/train_dir_velocity_nav}"
SOURCE_EXPERIMENT="${SOURCE_EXPERIMENT:-single_quad_velocity_nav_lidar_v17_flexible_dense}"
SOURCE_CHECKPOINT_KIND="${SOURCE_CHECKPOINT_KIND:-best}"
EXPERIMENT="${EXPERIMENT:-single_quad_velocity_nav_lidar_v17_agile_attitude}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NUM_ENVS_PER_WORKER="${NUM_ENVS_PER_WORKER:-4}"
TOTAL_STEPS="${TOTAL_STEPS:-30000000}"
LEARNING_RATE="${LEARNING_RATE:-0.000004}"
EPISODE_DURATION="${EPISODE_DURATION:-30.0}"
SIM_STEPS="${SIM_STEPS:-1}"
LIDAR_NUM_RAYS="${LIDAR_NUM_RAYS:-17}"

VELOCITY_XY_MAX="${VELOCITY_XY_MAX:-2.15}"
VELOCITY_Z_MAX="${VELOCITY_Z_MAX:-0.50}"
VELOCITY_MAX_TILT_DEG="${VELOCITY_MAX_TILT_DEG:-38.0}"
VELOCITY_ATTITUDE_MAX_ANGLE_DEG="${VELOCITY_ATTITUDE_MAX_ANGLE_DEG:-28.0}"
VELOCITY_ATTITUDE_BLEND="${VELOCITY_ATTITUDE_BLEND:-0.20}"
VELOCITY_MAX_ACC_XY="${VELOCITY_MAX_ACC_XY:-7.0}"
VELOCITY_MAX_ACC_Z="${VELOCITY_MAX_ACC_Z:-2.2}"
VELOCITY_COMMAND_SMOOTHING_TAU="${VELOCITY_COMMAND_SMOOTHING_TAU:-0.0}"

OBST_DENSITY="${OBST_DENSITY:-0.24}"
OBST_DENSITY_MIN="${OBST_DENSITY_MIN:-0.18}"
OBST_DENSITY_MAX="${OBST_DENSITY_MAX:-0.34}"
OBST_SIZE="${OBST_SIZE:-0.62}"
OBST_SIZE_MIN="${OBST_SIZE_MIN:-0.50}"
OBST_SIZE_MAX="${OBST_SIZE_MAX:-0.76}"
GOAL_BALL_COUNT="${GOAL_BALL_COUNT:-14}"

OBST_COLLISION_REWARD="${OBST_COLLISION_REWARD:-8.0}"
OBST_PROXIMITY_REWARD="${OBST_PROXIMITY_REWARD:-0.55}"
OBST_SAFE_DISTANCE="${OBST_SAFE_DISTANCE:-1.25}"
OBST_GUARD_DISTANCE="${OBST_GUARD_DISTANCE:-0.34}"
WALL_COLLISION_REWARD="${WALL_COLLISION_REWARD:-8.0}"
WALL_PROXIMITY_REWARD="${WALL_PROXIMITY_REWARD:-0.55}"
WALL_SAFE_DISTANCE="${WALL_SAFE_DISTANCE:-1.20}"
WALL_GUARD_DISTANCE="${WALL_GUARD_DISTANCE:-0.34}"
SAFE_FLIGHT_REWARD="${SAFE_FLIGHT_REWARD:-0.18}"
PATH_ALIGNMENT_REWARD="${PATH_ALIGNMENT_REWARD:-0.35}"
REWARD_SCALE="${REWARD_SCALE:-0.2}"
GOAL_BALL_VELOCITY_RESET="${GOAL_BALL_VELOCITY_RESET:-True}"
GOAL_BALL_VELOCITY_RESET_RATIO="${GOAL_BALL_VELOCITY_RESET_RATIO:-0.0}"

if [[ "$MODE" == "train" ]]; then
  echo "== Preparing agile attitude warm start from $SOURCE_EXPERIMENT ($SOURCE_CHECKPOINT_KIND) =="
  "$PYTHON_BIN" -m swarm_rl.vision.prepare_depth_warmstart \
    --source_experiment="$SOURCE_EXPERIMENT" \
    --target_experiment="$EXPERIMENT" \
    --source_train_dir="$TRAIN_DIR" \
    --train_dir="$TRAIN_DIR" \
    --checkpoint_kind="$SOURCE_CHECKPOINT_KIND" \
    --target_action_dim=6 \
    --quads_control_mode=velocity_attitude \
    --quads_obstacle_obs_type=lidar \
    --quads_lidar_num_rays="$LIDAR_NUM_RAYS" \
    --reset_optimizer=True \
    --reset_training_progress=True \
    --checkpoint_lr="$LEARNING_RATE" \
    --reward_scale="$REWARD_SCALE" \
    --quads_velocity_xy_max="$VELOCITY_XY_MAX" \
    --quads_velocity_z_max="$VELOCITY_Z_MAX" \
    --quads_velocity_max_tilt_deg="$VELOCITY_MAX_TILT_DEG" \
    --quads_velocity_attitude_max_angle_deg="$VELOCITY_ATTITUDE_MAX_ANGLE_DEG" \
    --quads_velocity_attitude_blend="$VELOCITY_ATTITUDE_BLEND" \
    --quads_velocity_max_acc_xy="$VELOCITY_MAX_ACC_XY" \
    --quads_velocity_max_acc_z_up="$VELOCITY_MAX_ACC_Z" \
    --quads_velocity_max_acc_z_down="$VELOCITY_MAX_ACC_Z" \
    --quads_velocity_command_smoothing_tau="$VELOCITY_COMMAND_SMOOTHING_TAU" \
    --quads_obst_density="$OBST_DENSITY" \
    --quads_obst_density_min="$OBST_DENSITY_MIN" \
    --quads_obst_density_max="$OBST_DENSITY_MAX" \
    --quads_obst_size="$OBST_SIZE" \
    --quads_obst_size_min="$OBST_SIZE_MIN" \
    --quads_obst_size_max="$OBST_SIZE_MAX" \
    --quads_episode_duration="$EPISODE_DURATION" \
    --quads_sim_steps="$SIM_STEPS" \
    --quads_obst_collision_reward="$OBST_COLLISION_REWARD" \
    --quads_reward_obstacle_proximity="$OBST_PROXIMITY_REWARD" \
    --quads_obstacle_safe_distance="$OBST_SAFE_DISTANCE" \
    --quads_obstacle_guard_distance="$OBST_GUARD_DISTANCE" \
    --quads_obst_collision_terminate=True \
    --quads_obstacle_guard_terminate=False \
    --quads_wall_collision_reward="$WALL_COLLISION_REWARD" \
    --quads_reward_wall_proximity="$WALL_PROXIMITY_REWARD" \
    --quads_wall_safe_distance="$WALL_SAFE_DISTANCE" \
    --quads_wall_guard_distance="$WALL_GUARD_DISTANCE" \
    --quads_wall_collision_terminate=True \
    --quads_wall_guard_terminate=False \
    --quads_reward_progress=3.2 \
    --quads_reward_action_change=0.025 \
    --quads_reward_vertical_velocity=0.15 \
    --quads_reward_height_error=0.15 \
    --quads_reward_thrust=0.03 \
    --quads_reward_stagnation=0.10 \
    --quads_reward_safe_flight="$SAFE_FLIGHT_REWARD" \
    --quads_reward_path_alignment="$PATH_ALIGNMENT_REWARD" \
    --quads_goal_ball_reward=1.4 \
    --quads_goal_ball_radius=0.45 \
    --quads_goal_ball_count="$GOAL_BALL_COUNT" \
    --quads_goal_ball_velocity_reset="$GOAL_BALL_VELOCITY_RESET" \
    --quads_goal_ball_velocity_reset_ratio="$GOAL_BALL_VELOCITY_RESET_RATIO" \
    --force=True
  RESTART_BEHAVIOR="resume"
elif [[ "$MODE" == "resume" ]]; then
  RESTART_BEHAVIOR="resume"
elif [[ "$MODE" == "scratch" ]]; then
  RESTART_BEHAVIOR="overwrite"
else
  echo "Usage: $0 [train|resume|scratch]" >&2
  exit 1
fi

echo "== 360-degree lidar agile attitude obstacle-avoidance training =="

"$PYTHON_BIN" -m swarm_rl.train \
  --env=quadrotor_multi \
  --algo=APPO \
  --device="$DEVICE" \
  --experiment="$EXPERIMENT" \
  --train_dir="$TRAIN_DIR" \
  --restart_behavior="$RESTART_BEHAVIOR" \
  --load_checkpoint_kind=latest \
  --train_for_env_steps="$TOTAL_STEPS" \
  --use_rnn=False \
  --recurrence=1 \
  --num_workers="$NUM_WORKERS" \
  --num_envs_per_worker="$NUM_ENVS_PER_WORKER" \
  --learning_rate="$LEARNING_RATE" \
  --ppo_clip_value=5.0 \
  --nonlinearity=tanh \
  --actor_critic_share_weights=False \
  --policy_initialization=xavier_uniform \
  --adaptive_stddev=False \
  --with_vtrace=False \
  --max_policy_lag=100000000 \
  --gae_lambda=1.0 \
  --max_grad_norm=5.0 \
  --exploration_loss_coeff=0.0 \
  --rollout=128 \
  --batch_size=1024 \
  --with_pbt=False \
  --normalize_input=False \
  --normalize_returns=False \
  --reward_scale="$REWARD_SCALE" \
  --reward_clip=30 \
  --save_every_sec=240 \
  --keep_checkpoints=2 \
  --save_milestones_sec=100000000 \
  --replay_buffer_sample_prob=0.0 \
  --anneal_collision_steps=0 \
  --quads_use_numba=True \
  --quads_render=False \
  --quads_num_agents=1 \
  --quads_sim_steps="$SIM_STEPS" \
  --quads_control_mode=velocity_attitude \
  --quads_velocity_xy_max="$VELOCITY_XY_MAX" \
  --quads_velocity_z_max="$VELOCITY_Z_MAX" \
  --quads_velocity_max_tilt_deg="$VELOCITY_MAX_TILT_DEG" \
  --quads_velocity_attitude_max_angle_deg="$VELOCITY_ATTITUDE_MAX_ANGLE_DEG" \
  --quads_velocity_attitude_blend="$VELOCITY_ATTITUDE_BLEND" \
  --quads_velocity_max_acc_xy="$VELOCITY_MAX_ACC_XY" \
  --quads_velocity_max_acc_z_up="$VELOCITY_MAX_ACC_Z" \
  --quads_velocity_max_acc_z_down="$VELOCITY_MAX_ACC_Z" \
  --quads_velocity_command_smoothing_tau="$VELOCITY_COMMAND_SMOOTHING_TAU" \
  --quads_episode_duration="$EPISODE_DURATION" \
  --quads_obs_repr=xyz_vxyz_R_omega_wall \
  --quads_neighbor_visible_num=0 \
  --quads_neighbor_obs_type=none \
  --quads_neighbor_encoder_type=no_encoder \
  --quads_neighbor_hidden_size=64 \
  --quads_obst_hidden_size=128 \
  --quads_collision_reward=0.0 \
  --quads_collision_hitbox_radius=2.0 \
  --quads_collision_falloff_radius=-1.0 \
  --quads_collision_smooth_max_penalty=0.0 \
  --quads_use_obstacles=True \
  --quads_obstacle_obs_type=lidar \
  --quads_lidar_num_rays="$LIDAR_NUM_RAYS" \
  --quads_depth_min_distance=0.05 \
  --quads_depth_max_distance=10.0 \
  --quads_depth_noise_std=0.0 \
  --quads_depth_dropout_prob=0.0 \
  --quads_depth_normalize=False \
  --quads_obst_spawn_area 9 9 \
  --quads_room_dims 10 10 4 \
  --quads_goal_z_min=1.4 \
  --quads_goal_z_max=2.2 \
  --quads_camera_width=320 \
  --quads_camera_height=240 \
  --quads_camera_fov=145 \
  --quads_camera_pitch_deg=15 \
  --quads_use_downwash=False \
  --quads_mode=o_random \
  --quads_obst_density="$OBST_DENSITY" \
  --quads_obst_size="$OBST_SIZE" \
  --quads_obst_collision_reward="$OBST_COLLISION_REWARD" \
  --quads_reward_obstacle_proximity="$OBST_PROXIMITY_REWARD" \
  --quads_obstacle_safe_distance="$OBST_SAFE_DISTANCE" \
  --quads_obstacle_guard_distance="$OBST_GUARD_DISTANCE" \
  --quads_obst_collision_terminate=True \
  --quads_obstacle_guard_terminate=False \
  --quads_wall_collision_reward="$WALL_COLLISION_REWARD" \
  --quads_reward_wall_proximity="$WALL_PROXIMITY_REWARD" \
  --quads_wall_safe_distance="$WALL_SAFE_DISTANCE" \
  --quads_wall_guard_distance="$WALL_GUARD_DISTANCE" \
  --quads_wall_collision_terminate=True \
  --quads_wall_guard_terminate=False \
  --quads_domain_random=True \
  --quads_obst_density_random=True \
  --quads_obst_density_min="$OBST_DENSITY_MIN" \
  --quads_obst_density_max="$OBST_DENSITY_MAX" \
  --quads_obst_size_random=True \
  --quads_obst_size_min="$OBST_SIZE_MIN" \
  --quads_obst_size_max="$OBST_SIZE_MAX" \
  --quads_reward_progress=3.2 \
  --quads_reward_action_change=0.025 \
  --quads_reward_vertical_velocity=0.15 \
  --quads_reward_height_error=0.15 \
  --quads_reward_thrust=0.03 \
  --quads_reward_stagnation=0.10 \
  --quads_reward_safe_flight="$SAFE_FLIGHT_REWARD" \
  --quads_reward_path_alignment="$PATH_ALIGNMENT_REWARD" \
  --quads_use_goal_ball=True \
  --quads_goal_ball_reward=1.4 \
  --quads_goal_ball_radius=0.45 \
  --quads_goal_ball_count="$GOAL_BALL_COUNT" \
  --quads_goal_ball_velocity_reset="$GOAL_BALL_VELOCITY_RESET" \
  --quads_goal_ball_velocity_reset_ratio="$GOAL_BALL_VELOCITY_RESET_RATIO"
