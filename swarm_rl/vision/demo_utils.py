from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sample_factory.utils.utils import str2bool

from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env_multi
from swarm_rl.env_wrappers.quadrotor_params import add_quadrotors_env_args


REPO_ROOT = Path(__file__).resolve().parents[2]


def build_vision_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    add_quadrotors_env_args(None, parser)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--with_pbt", default=False, type=str2bool)
    parser.add_argument("--seed", default=0, type=int)

    parser.set_defaults(
        quads_num_agents=1,
        quads_episode_duration=16.0,
        quads_obs_repr="xyz_vxyz_R_omega_wall",
        quads_control_mode="velocity",
        quads_velocity_xy_max=2.8,
        quads_velocity_z_max=0.9,
        quads_neighbor_visible_num=0,
        quads_neighbor_obs_type="none",
        quads_neighbor_hidden_size=64,
        quads_neighbor_encoder_type="no_encoder",
        quads_obst_hidden_size=128,
        quads_use_obstacles=True,
        quads_obstacle_obs_type="yolo",
        quads_obst_spawn_area=[9.0, 9.0],
        quads_room_dims=[10.0, 10.0, 4.0],
        quads_goal_z_min=1.4,
        quads_goal_z_max=2.2,
        quads_use_downwash=False,
        quads_mode="o_random",
        quads_obst_density=0.26,
        quads_obst_size=0.72,
        quads_use_goal_ball=True,
        quads_goal_ball_reward=0.8,
        quads_goal_ball_radius=0.35,
        quads_goal_ball_count=10,
        quads_render=False,
        quads_use_numba=False,
        quads_view_mode=["fpv"],
        quads_camera_width=640,
        quads_camera_height=480,
        quads_camera_fov=100.0,
        quads_camera_pitch_deg=25.0,
        quads_yolo_source="oracle_mask",
        quads_yolo_model_path="",
        quads_yolo_conf_threshold=0.25,
        replay_buffer_sample_prob=0.0,
        visualize_v_value=False,
    )
    return parser


def finalize_env_cfg(args: argparse.Namespace) -> SimpleNamespace:
    cfg = SimpleNamespace(**vars(args))
    cfg.policy_index = 0
    cfg.load_checkpoint_kind = "best"
    return cfg


def make_base_env(args: argparse.Namespace, render_mode: str = "rgb_array"):
    cfg = finalize_env_cfg(args)
    np.random.seed(cfg.seed)
    wrapped_env = make_quadrotor_env_multi(cfg, render_mode=render_mode)
    return wrapped_env.unwrapped


def heuristic_goal_velocity(env, drone_index: int = 0, speed_xy: float = 2.2, speed_z: float = 0.6) -> np.ndarray:
    goal = np.asarray(env.envs[drone_index].goal, dtype=np.float32)
    pos = np.asarray(env.envs[drone_index].dynamics.pos, dtype=np.float32)
    delta = goal - pos

    action = np.zeros(3, dtype=np.float32)
    delta_xy = delta[:2]
    delta_xy_norm = float(np.linalg.norm(delta_xy))
    if delta_xy_norm > 1e-6:
        action[:2] = delta_xy / delta_xy_norm * min(speed_xy, delta_xy_norm)
    action[2] = np.clip(delta[2], -speed_z, speed_z)

    action = np.clip(action, env.action_space.low.astype(np.float32), env.action_space.high.astype(np.float32))
    return action
