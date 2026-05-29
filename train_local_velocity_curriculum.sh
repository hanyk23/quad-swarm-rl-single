#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/lzh/miniconda3/envs/swarm-rl/bin/python}"

MODE="${1:-train}"
DEVICE="${DEVICE:-gpu}"
TRAIN_DIR="${TRAIN_DIR:-$ROOT_DIR/train_dir_velocity_nav}"
EXPERIMENT="${EXPERIMENT:-single_quad_velocity_nav_yolo_v4_15box_stable}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NUM_ENVS_PER_WORKER="${NUM_ENVS_PER_WORKER:-4}"
TOTAL_STEPS="${TOTAL_STEPS:-20000000}"
RESTART_BEHAVIOR="${RESTART_BEHAVIOR:-overwrite}"

if [[ "$MODE" == "resume" ]]; then
  RESTART_BEHAVIOR="resume"
elif [[ "$MODE" != "train" ]]; then
  echo "Usage: $0 [train|resume]" >&2
  exit 1
fi

echo "== Stable 15-box YOLO-FPV obstacle-avoidance training =="

"$PYTHON_BIN" -m swarm_rl.train \
  --env=quadrotor_multi \
  --algo=APPO \
  --device="$DEVICE" \
  --experiment="$EXPERIMENT" \
  --train_dir="$TRAIN_DIR" \
  --restart_behavior="$RESTART_BEHAVIOR" \
  --train_for_env_steps="$TOTAL_STEPS" \
  --use_rnn=False \
  --recurrence=1 \
  --num_workers="$NUM_WORKERS" \
  --num_envs_per_worker="$NUM_ENVS_PER_WORKER" \
  --learning_rate=0.00015 \
  --ppo_clip_value=5.0 \
  --nonlinearity=tanh \
  --actor_critic_share_weights=False \
  --policy_initialization=xavier_uniform \
  --adaptive_stddev=False \
  --with_vtrace=False \
  --max_policy_lag=100000000 \
  --rnn_size=128 \
  --gae_lambda=1.0 \
  --max_grad_norm=5.0 \
  --exploration_loss_coeff=0.0 \
  --rollout=128 \
  --batch_size=1024 \
  --with_pbt=False \
  --normalize_input=False \
  --normalize_returns=False \
  --reward_clip=10 \
  --save_every_sec=240 \
  --keep_checkpoints=1 \
  --save_milestones_sec=100000000 \
  --replay_buffer_sample_prob=0.0 \
  --anneal_collision_steps=800000 \
  --quads_use_numba=True \
  --quads_render=False \
  --quads_num_agents=1 \
  --quads_control_mode=velocity \
  --quads_velocity_xy_max=10.0 \
  --quads_velocity_z_max=3.0 \
  --quads_episode_duration=8.0 \
  --quads_obs_repr=xyz_vxyz_R_omega_wall \
  --quads_neighbor_visible_num=0 \
  --quads_neighbor_obs_type=none \
  --quads_neighbor_encoder_type=no_encoder \
  --quads_neighbor_hidden_size=64 \
  --quads_obst_hidden_size=256 \
  --quads_collision_reward=0.0 \
  --quads_collision_hitbox_radius=2.0 \
  --quads_collision_falloff_radius=-1.0 \
  --quads_collision_smooth_max_penalty=0.0 \
  --quads_use_obstacles=True \
  --quads_obstacle_obs_type=yolo \
  --quads_obst_spawn_area 9 9 \
  --quads_room_dims 10 10 4 \
  --quads_goal_z_min=1.4 \
  --quads_goal_z_max=2.2 \
  --quads_camera_width=320 \
  --quads_camera_height=240 \
  --quads_camera_fov=145 \
  --quads_camera_pitch_deg=15 \
  --quads_yolo_source=oracle \
  --quads_use_downwash=False \
  --quads_mode=o_random \
  --quads_obst_density=0.26 \
  --quads_obst_size=0.72 \
  --quads_obst_collision_reward=6.0 \
  --quads_domain_random=True \
  --quads_obst_density_random=True \
  --quads_obst_density_min=0.22 \
  --quads_obst_density_max=0.32 \
  --quads_obst_size_random=True \
  --quads_obst_size_min=0.62 \
  --quads_obst_size_max=0.82 \
  --quads_reward_progress=2.8 \
  --quads_reward_action_change=0.015 \
  --quads_reward_vertical_velocity=0.10 \
  --quads_reward_thrust=0.05 \
  --quads_reward_stagnation=0.22 \
  --quads_use_goal_ball=True \
  --quads_goal_ball_reward=0.8 \
  --quads_goal_ball_radius=0.35 \
  --quads_goal_ball_count=10