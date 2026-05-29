from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from sample_factory.utils.utils import str2bool


REPO_ROOT = Path(__file__).resolve().parents[2]


def latest_checkpoint(checkpoint_dir: Path, kind: str) -> Path:
    prefix = {"latest": "checkpoint", "best": "best"}[kind]
    matches = sorted(checkpoint_dir.glob(f"{prefix}_*.pth"))
    if not matches:
        raise FileNotFoundError(f"No {kind} checkpoint found in {checkpoint_dir}")
    return matches[-1]


def update_config(config: dict, args: argparse.Namespace) -> dict:
    config = dict(config)
    updates = {
        "experiment": args.target_experiment,
        "train_dir": str(args.train_dir),
        "restart_behavior": "resume",
        "load_checkpoint_kind": "latest",
        "learning_rate": args.checkpoint_lr,
        "reward_scale": args.reward_scale,
        "quads_control_mode": args.quads_control_mode,
        "quads_obstacle_obs_type": args.quads_obstacle_obs_type,
        "quads_depth_grid_width": args.quads_depth_grid_width,
        "quads_depth_grid_height": args.quads_depth_grid_height,
        "quads_depth_min_distance": args.quads_depth_min_distance,
        "quads_depth_max_distance": args.quads_depth_max_distance,
        "quads_lidar_num_rays": args.quads_lidar_num_rays,
        "quads_depth_noise_std": args.quads_depth_noise_std,
        "quads_depth_dropout_prob": args.quads_depth_dropout_prob,
        "quads_depth_normalize": args.quads_depth_normalize,
        "quads_velocity_xy_max": args.quads_velocity_xy_max,
        "quads_velocity_z_max": args.quads_velocity_z_max,
        "quads_velocity_max_tilt_deg": args.quads_velocity_max_tilt_deg,
        "quads_velocity_max_acc_xy": args.quads_velocity_max_acc_xy,
        "quads_velocity_max_acc_z_up": args.quads_velocity_max_acc_z_up,
        "quads_velocity_max_acc_z_down": args.quads_velocity_max_acc_z_down,
        "quads_velocity_yaw_mode": args.quads_velocity_yaw_mode,
        "quads_velocity_yaw_min_speed": args.quads_velocity_yaw_min_speed,
        "quads_velocity_yaw_rate_max": args.quads_velocity_yaw_rate_max,
        "quads_velocity_yaw_control_scale": args.quads_velocity_yaw_control_scale,
        "quads_velocity_command_smoothing_tau": args.quads_velocity_command_smoothing_tau,
        "quads_controller_obstacle_avoidance": args.quads_controller_obstacle_avoidance,
        "quads_obstacle_avoidance_distance": args.quads_obstacle_avoidance_distance,
        "quads_obstacle_avoidance_max_speed": args.quads_obstacle_avoidance_max_speed,
        "quads_obstacle_avoidance_gain": args.quads_obstacle_avoidance_gain,
        "quads_obstacle_avoidance_pid_kp": args.quads_obstacle_avoidance_pid_kp,
        "quads_obstacle_avoidance_pid_ki": args.quads_obstacle_avoidance_pid_ki,
        "quads_obstacle_avoidance_pid_kd": args.quads_obstacle_avoidance_pid_kd,
        "quads_obstacle_avoidance_pid_integral_limit": args.quads_obstacle_avoidance_pid_integral_limit,
        "quads_goal_ball_capture_assist": args.quads_goal_ball_capture_assist,
        "quads_goal_ball_capture_assist_distance": args.quads_goal_ball_capture_assist_distance,
        "quads_goal_ball_capture_assist_speed": args.quads_goal_ball_capture_assist_speed,
        "quads_goal_ball_tangent_damping": args.quads_goal_ball_tangent_damping,
        "quads_velocity_attitude_max_angle_deg": args.quads_velocity_attitude_max_angle_deg,
        "quads_velocity_attitude_blend": args.quads_velocity_attitude_blend,
        "quads_episode_duration": args.quads_episode_duration,
        "quads_sim_freq": args.quads_sim_freq,
        "quads_sim_steps": args.quads_sim_steps,
        "quads_obst_density": args.quads_obst_density,
        "quads_obst_density_random": args.quads_obst_density_random,
        "quads_obst_density_min": args.quads_obst_density_min,
        "quads_obst_density_max": args.quads_obst_density_max,
        "quads_obst_size": args.quads_obst_size,
        "quads_obst_size_random": args.quads_obst_size_random,
        "quads_obst_size_min": args.quads_obst_size_min,
        "quads_obst_size_max": args.quads_obst_size_max,
        "quads_obst_collision_reward": args.quads_obst_collision_reward,
        "quads_reward_obstacle_proximity": args.quads_reward_obstacle_proximity,
        "quads_reward_obstacle_clearance_delta": args.quads_reward_obstacle_clearance_delta,
        "quads_obstacle_safe_distance": args.quads_obstacle_safe_distance,
        "quads_obstacle_guard_distance": args.quads_obstacle_guard_distance,
        "quads_obstacle_guard_terminate": args.quads_obstacle_guard_terminate,
        "quads_wall_collision_reward": args.quads_wall_collision_reward,
        "quads_reward_wall_proximity": args.quads_reward_wall_proximity,
        "quads_reward_wall_clearance_delta": args.quads_reward_wall_clearance_delta,
        "quads_wall_safe_distance": args.quads_wall_safe_distance,
        "quads_wall_guard_distance": args.quads_wall_guard_distance,
        "quads_wall_guard_terminate": args.quads_wall_guard_terminate,
        "quads_reward_progress": args.quads_reward_progress,
        "quads_reward_action_change": args.quads_reward_action_change,
        "quads_reward_vertical_velocity": args.quads_reward_vertical_velocity,
        "quads_reward_height_error": args.quads_reward_height_error,
        "quads_reward_thrust": args.quads_reward_thrust,
        "quads_reward_stagnation": args.quads_reward_stagnation,
        "quads_reward_overspeed": args.quads_reward_overspeed,
        "quads_reward_safe_flight": args.quads_reward_safe_flight,
        "quads_reward_path_alignment": args.quads_reward_path_alignment,
        "quads_obst_collision_terminate": args.quads_obst_collision_terminate,
        "quads_wall_collision_terminate": args.quads_wall_collision_terminate,
        "quads_goal_ball_reward": args.quads_goal_ball_reward,
        "quads_goal_ball_radius": args.quads_goal_ball_radius,
        "quads_goal_ball_count": args.quads_goal_ball_count,
        "quads_goal_ball_velocity_reset": args.quads_goal_ball_velocity_reset,
        "quads_goal_ball_velocity_reset_ratio": args.quads_goal_ball_velocity_reset_ratio,
    }
    config.update(updates)

    cli_args = dict(config.get("cli_args", {}))
    cli_args.update(updates)
    config["cli_args"] = cli_args
    config["command_line"] = (
        f"warm-started from {args.source_train_dir / args.source_experiment} "
        f"with quads_obstacle_obs_type={args.quads_obstacle_obs_type}"
    )
    return config


def resize_continuous_action_head(checkpoint: dict, action_dim: int) -> None:
    model = checkpoint.get("model", {})
    weight_key = "action_parameterization.distribution_linear.weight"
    bias_key = "action_parameterization.distribution_linear.bias"
    std_key = "action_parameterization.learned_stddev"

    if weight_key not in model or bias_key not in model or std_key not in model:
        return

    old_weight = model[weight_key]
    old_bias = model[bias_key]
    old_std = model[std_key]
    old_dim = int(old_weight.shape[0])
    action_dim = int(action_dim)
    if old_dim == action_dim:
        return

    new_weight = torch.zeros((action_dim, old_weight.shape[1]), dtype=old_weight.dtype)
    new_bias = torch.zeros((action_dim,), dtype=old_bias.dtype)
    new_std = torch.empty((action_dim,), dtype=old_std.dtype)
    new_std.fill_(min(float(old_std.mean()), 0.15))

    copy_dim = min(old_dim, action_dim)
    new_weight[:copy_dim] = old_weight[:copy_dim]
    new_bias[:copy_dim] = old_bias[:copy_dim]
    new_std[:copy_dim] = old_std[:copy_dim]

    model[weight_key] = new_weight
    model[bias_key] = new_bias
    model[std_key] = new_std


def target_obstacle_obs_dim(args: argparse.Namespace) -> int:
    if args.quads_obstacle_obs_type == "lidar":
        return int(args.quads_lidar_num_rays)
    if args.quads_obstacle_obs_type == "depth":
        return int(args.quads_depth_grid_width) * int(args.quads_depth_grid_height)
    return 9


def resize_obstacle_encoder_input(checkpoint: dict, obstacle_obs_dim: int) -> None:
    model = checkpoint.get("model", {})
    obstacle_obs_dim = int(obstacle_obs_dim)
    for key, old_weight in list(model.items()):
        if not key.endswith("obstacle_encoder.0.weight"):
            continue
        if int(old_weight.shape[1]) == obstacle_obs_dim:
            continue

        new_weight = torch.zeros((old_weight.shape[0], obstacle_obs_dim), dtype=old_weight.dtype)
        copy_dim = min(int(old_weight.shape[1]), obstacle_obs_dim)
        new_weight[:, :copy_dim] = old_weight[:, :copy_dim]
        if obstacle_obs_dim > copy_dim:
            new_weight[:, copy_dim:] = old_weight.mean(dim=1, keepdim=True)
        model[key] = new_weight


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Create an obstacle-avoidance RL experiment initialized from an existing 9D obstacle policy."
    )
    parser.add_argument("--source_experiment", default="single_quad_velocity_nav_randomball_v3", type=str)
    parser.add_argument("--target_experiment", default="single_quad_velocity_nav_lidar_attitude_v9", type=str)
    parser.add_argument("--source_train_dir", default=str(REPO_ROOT / "train_dir_velocity_nav"), type=Path)
    parser.add_argument("--train_dir", default=str(REPO_ROOT / "train_dir_velocity_nav"), type=Path)
    parser.add_argument("--checkpoint_kind", default="latest", choices=["best", "latest"], type=str)
    parser.add_argument("--target_action_dim", default=6, type=int)
    parser.add_argument("--force", default=False, type=str2bool)
    parser.add_argument("--reset_optimizer", default=True, type=str2bool)
    parser.add_argument(
        "--reset_training_progress",
        default=True,
        type=str2bool,
        help="Start target train_step/env_steps/best metric from zero while keeping warm-started model weights.",
    )
    parser.add_argument("--checkpoint_lr", default=5e-6, type=float)
    parser.add_argument("--reward_scale", default=0.2, type=float)
    parser.add_argument(
        "--quads_control_mode",
        default="velocity_attitude",
        choices=["velocity", "velocity_yaw", "velocity_attitude"],
        type=str,
    )
    parser.add_argument("--quads_obstacle_obs_type", default="lidar", choices=["octomap", "depth", "lidar"], type=str)
    parser.add_argument("--quads_depth_grid_width", default=3, type=int)
    parser.add_argument("--quads_depth_grid_height", default=3, type=int)
    parser.add_argument("--quads_depth_min_distance", default=0.05, type=float)
    parser.add_argument("--quads_depth_max_distance", default=10.0, type=float)
    parser.add_argument("--quads_lidar_num_rays", default=9, type=int)
    parser.add_argument("--quads_depth_noise_std", default=0.0, type=float)
    parser.add_argument("--quads_depth_dropout_prob", default=0.0, type=float)
    parser.add_argument("--quads_depth_normalize", default=False, type=str2bool)
    parser.add_argument("--quads_velocity_xy_max", default=2.4, type=float)
    parser.add_argument("--quads_velocity_z_max", default=0.45, type=float)
    parser.add_argument("--quads_velocity_max_tilt_deg", default=35.0, type=float)
    parser.add_argument("--quads_velocity_max_acc_xy", default=5.5, type=float)
    parser.add_argument("--quads_velocity_max_acc_z_up", default=2.0, type=float)
    parser.add_argument("--quads_velocity_max_acc_z_down", default=2.0, type=float)
    parser.add_argument("--quads_velocity_yaw_mode", default="velocity_or_goal",
                        choices=["keep", "velocity", "goal", "velocity_or_goal"], type=str)
    parser.add_argument("--quads_velocity_yaw_min_speed", default=0.05, type=float)
    parser.add_argument("--quads_velocity_yaw_rate_max", default=0.0, type=float)
    parser.add_argument("--quads_velocity_yaw_control_scale", default=1.5, type=float)
    parser.add_argument("--quads_velocity_command_smoothing_tau", default=0.08, type=float)
    parser.add_argument("--quads_controller_obstacle_avoidance", default=False, type=str2bool)
    parser.add_argument("--quads_obstacle_avoidance_distance", default=1.2, type=float)
    parser.add_argument("--quads_obstacle_avoidance_max_speed", default=0.8, type=float)
    parser.add_argument("--quads_obstacle_avoidance_gain", default=1.2, type=float)
    parser.add_argument("--quads_obstacle_avoidance_pid_kp", default=1.0, type=float)
    parser.add_argument("--quads_obstacle_avoidance_pid_ki", default=0.0, type=float)
    parser.add_argument("--quads_obstacle_avoidance_pid_kd", default=0.0, type=float)
    parser.add_argument("--quads_obstacle_avoidance_pid_integral_limit", default=1.0, type=float)
    parser.add_argument("--quads_goal_ball_capture_assist", default=False, type=str2bool)
    parser.add_argument("--quads_goal_ball_capture_assist_distance", default=1.2, type=float)
    parser.add_argument("--quads_goal_ball_capture_assist_speed", default=0.8, type=float)
    parser.add_argument("--quads_goal_ball_tangent_damping", default=0.25, type=float)
    parser.add_argument("--quads_velocity_attitude_max_angle_deg", default=45.0, type=float)
    parser.add_argument("--quads_velocity_attitude_blend", default=0.75, type=float)
    parser.add_argument("--quads_episode_duration", default=24.0, type=float)
    parser.add_argument("--quads_sim_freq", default=200.0, type=float)
    parser.add_argument("--quads_sim_steps", default=2, type=int)
    parser.add_argument("--quads_obst_density", default=0.18, type=float)
    parser.add_argument("--quads_obst_density_random", default=True, type=str2bool)
    parser.add_argument("--quads_obst_density_min", default=0.12, type=float)
    parser.add_argument("--quads_obst_density_max", default=0.24, type=float)
    parser.add_argument("--quads_obst_size", default=0.66, type=float)
    parser.add_argument("--quads_obst_size_random", default=True, type=str2bool)
    parser.add_argument("--quads_obst_size_min", default=0.54, type=float)
    parser.add_argument("--quads_obst_size_max", default=0.78, type=float)
    parser.add_argument("--quads_obst_collision_reward", default=5.0, type=float)
    parser.add_argument("--quads_reward_obstacle_proximity", default=0.45, type=float)
    parser.add_argument("--quads_reward_obstacle_clearance_delta", default=0.0, type=float)
    parser.add_argument("--quads_obstacle_safe_distance", default=0.80, type=float)
    parser.add_argument("--quads_obstacle_guard_distance", default=0.0, type=float)
    parser.add_argument("--quads_obstacle_guard_terminate", default=True, type=str2bool)
    parser.add_argument("--quads_wall_collision_reward", default=0.0, type=float)
    parser.add_argument("--quads_reward_wall_proximity", default=0.0, type=float)
    parser.add_argument("--quads_reward_wall_clearance_delta", default=0.0, type=float)
    parser.add_argument("--quads_wall_safe_distance", default=0.80, type=float)
    parser.add_argument("--quads_wall_guard_distance", default=0.0, type=float)
    parser.add_argument("--quads_wall_guard_terminate", default=True, type=str2bool)
    parser.add_argument("--quads_reward_progress", default=3.2, type=float)
    parser.add_argument("--quads_reward_action_change", default=0.035, type=float)
    parser.add_argument("--quads_reward_vertical_velocity", default=0.25, type=float)
    parser.add_argument("--quads_reward_height_error", default=0.25, type=float)
    parser.add_argument("--quads_reward_thrust", default=0.05, type=float)
    parser.add_argument("--quads_reward_stagnation", default=0.10, type=float)
    parser.add_argument("--quads_reward_overspeed", default=0.0, type=float)
    parser.add_argument("--quads_reward_safe_flight", default=0.0, type=float)
    parser.add_argument("--quads_reward_path_alignment", default=0.0, type=float)
    parser.add_argument("--quads_obst_collision_terminate", default=False, type=str2bool)
    parser.add_argument("--quads_wall_collision_terminate", default=False, type=str2bool)
    parser.add_argument("--quads_goal_ball_reward", default=1.4, type=float)
    parser.add_argument("--quads_goal_ball_radius", default=0.45, type=float)
    parser.add_argument("--quads_goal_ball_count", default=10, type=int)
    parser.add_argument("--quads_goal_ball_velocity_reset", default=False, type=str2bool)
    parser.add_argument("--quads_goal_ball_velocity_reset_ratio", default=0.0, type=float)
    args = parser.parse_args(argv)

    if args.quads_obstacle_obs_type == "depth" and args.quads_depth_grid_width * args.quads_depth_grid_height != 9:
        raise ValueError("Warm-start from the current model requires 9 obstacle features.")
    if args.quads_lidar_num_rays <= 0:
        raise ValueError("quads_lidar_num_rays must be positive.")

    source_exp_dir = args.source_train_dir / args.source_experiment
    target_exp_dir = args.train_dir / args.target_experiment
    source_cfg_path = source_exp_dir / "config.json"
    source_ckpt_dir = source_exp_dir / "checkpoint_p0"

    if not source_cfg_path.is_file():
        raise FileNotFoundError(f"Source config not found: {source_cfg_path}")

    if target_exp_dir.exists():
        if not args.force:
            raise FileExistsError(
                f"Target experiment already exists: {target_exp_dir}. "
                "Pass --force=True to reinitialize it from the source checkpoint."
            )
        shutil.rmtree(target_exp_dir)

    target_ckpt_dir = target_exp_dir / "checkpoint_p0"
    target_ckpt_dir.mkdir(parents=True, exist_ok=True)

    with source_cfg_path.open("r", encoding="utf-8") as cfg_file:
        config = json.load(cfg_file)
    config = update_config(config, args)

    target_cfg_path = target_exp_dir / "config.json"
    with target_cfg_path.open("w", encoding="utf-8") as cfg_file:
        json.dump(config, cfg_file, indent=2)

    source_ckpt = latest_checkpoint(source_ckpt_dir, args.checkpoint_kind)
    checkpoint = torch.load(source_ckpt, map_location="cpu", weights_only=False)
    resize_continuous_action_head(checkpoint, action_dim=args.target_action_dim)
    resize_obstacle_encoder_input(checkpoint, obstacle_obs_dim=target_obstacle_obs_dim(args))
    if args.reset_optimizer and "optimizer" in checkpoint:
        optimizer = checkpoint["optimizer"]
        optimizer["state"] = {}
        for group in optimizer.get("param_groups", []):
            group["lr"] = args.checkpoint_lr
        checkpoint["optimizer"] = optimizer
    checkpoint["curr_lr"] = args.checkpoint_lr
    if args.reset_training_progress:
        checkpoint["train_step"] = 0
        checkpoint["env_steps"] = 0
        checkpoint["best_performance"] = -1e9
    train_step = int(checkpoint.get("train_step", 0))
    env_steps = int(checkpoint.get("env_steps", 0))

    target_latest = target_ckpt_dir / f"checkpoint_{train_step:09d}_{env_steps}.pth"
    target_best = target_ckpt_dir / f"best_{train_step:09d}_{env_steps}_reward_warmstart.pth"
    torch.save(checkpoint, target_latest)
    torch.save(checkpoint, target_best)

    print(f"Prepared obstacle warm-start experiment: {target_exp_dir}")
    print(f"Source checkpoint: {source_ckpt}")
    print(f"Target checkpoint: {target_latest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
