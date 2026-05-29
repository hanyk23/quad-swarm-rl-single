from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.sampling.batched_sampling import preprocess_actions
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.env_info import extract_env_info
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.algo.utils.rl_utils import make_dones, prepare_and_normalize_obs
from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor
from sample_factory.cfg.arguments import load_from_checkpoint, parse_full_cfg, parse_sf_args
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.utils import log, str2bool

from swarm_rl.env_wrappers.quadrotor_params import add_quadrotors_env_args, quadrotors_override_defaults
from swarm_rl.train import register_swarm_components


def parse_viewer_cfg(argv=None):
    parser, partial_cfg = parse_sf_args(argv=argv, evaluation=True)
    for action in parser._actions:
        if action.dest == "env":
            action.required = False
    add_quadrotors_env_args(partial_cfg.env, parser)
    quadrotors_override_defaults(partial_cfg.env, parser)

    parser.add_argument("--display", default=True, type=str2bool, help="Show an OpenCV window")
    parser.add_argument("--display_scale", default=1.0, type=float, help="Resize the viewer window by this factor")
    parser.add_argument("--output_video", default="", type=str, help="Optional MP4 path for saving the chase + trajectory view")
    parser.add_argument("--viewer_fps", default=30, type=int, help="Target playback FPS for the viewer")
    parser.add_argument("--hud", default=True, type=str2bool, help="Show text overlays")
    parser.add_argument("--trajectory_history", default=2000, type=int, help="Number of recent positions to draw")

    parser.set_defaults(
        env="quadrotor_multi",
        algo="APPO",
        quads_view_mode=["chase"],
        quads_render=False,
        quads_use_numba=False,
        quads_obstacle_obs_type="depth",
        quads_depth_grid_width=3,
        quads_depth_grid_height=3,
        quads_depth_min_distance=0.05,
        quads_depth_max_distance=10.0,
        quads_lidar_num_rays=9,
        quads_depth_noise_std=0.0,
        quads_depth_dropout_prob=0.0,
        quads_depth_normalize=False,
        quads_camera_width=640,
        quads_camera_height=480,
        quads_camera_fov=145.0,
        quads_camera_pitch_deg=15.0,
        eval_deterministic=True,
        fps=30,
        max_num_episodes=10,
        display_scale=1.0,
    )
    return parse_full_cfg(parser, argv)


def open_video_writer(path: str, width: int, height: int, fps: int):
    if not path:
        return None, None

    video_path = Path(path).resolve()
    video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        video_path.as_posix(),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(max(fps, 1)),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {video_path}")
    return writer, video_path


def draw_hud(frame_bgr: np.ndarray, lines: list[str]):
    y = 22
    for line in lines:
        cv2.putText(frame_bgr, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame_bgr, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22


def world_to_panel(point_xy, room_dims, panel_w: int, panel_h: int, margin: int = 28):
    room_l = float(room_dims[0])
    room_w = float(room_dims[1])
    usable_w = max(panel_w - 2 * margin, 1)
    usable_h = max(panel_h - 2 * margin, 1)
    scale = min(usable_w / max(room_l, 1e-6), usable_h / max(room_w, 1e-6))
    map_w = room_l * scale
    map_h = room_w * scale
    origin_x = 0.5 * (panel_w - map_w)
    origin_y = 0.5 * (panel_h - map_h)

    x_norm = (float(point_xy[0]) + 0.5 * room_l) / max(room_l, 1e-6)
    y_norm = (float(point_xy[1]) + 0.5 * room_w) / max(room_w, 1e-6)
    px = int(round(origin_x + x_norm * map_w))
    py = int(round(origin_y + (1.0 - y_norm) * map_h))
    return px, py, scale, (int(round(origin_x)), int(round(origin_y)), int(round(map_w)), int(round(map_h)))


def draw_trajectory_panel(env, drone_index: int, trajectory: deque, panel_w: int, panel_h: int):
    panel = np.full((panel_h, panel_w, 3), (246, 248, 250), dtype=np.uint8)
    _, _, scale, map_rect = world_to_panel((0.0, 0.0), env.room_dims, panel_w, panel_h)
    map_x, map_y, map_w, map_h = map_rect

    cv2.rectangle(panel, (map_x, map_y), (map_x + map_w, map_y + map_h), (214, 220, 226), -1)
    cv2.rectangle(panel, (map_x, map_y), (map_x + map_w, map_y + map_h), (82, 93, 106), 2)

    grid_step = 1.0
    room_l = float(env.room_dims[0])
    room_w = float(env.room_dims[1])
    for gx in np.arange(-room_l / 2.0, room_l / 2.0 + 1e-6, grid_step):
        p0 = world_to_panel((gx, -room_w / 2.0), env.room_dims, panel_w, panel_h)[0:2]
        p1 = world_to_panel((gx, room_w / 2.0), env.room_dims, panel_w, panel_h)[0:2]
        cv2.line(panel, p0, p1, (232, 236, 240), 1, cv2.LINE_AA)
    for gy in np.arange(-room_w / 2.0, room_w / 2.0 + 1e-6, grid_step):
        p0 = world_to_panel((-room_l / 2.0, gy), env.room_dims, panel_w, panel_h)[0:2]
        p1 = world_to_panel((room_l / 2.0, gy), env.room_dims, panel_w, panel_h)[0:2]
        cv2.line(panel, p0, p1, (232, 236, 240), 1, cv2.LINE_AA)

    if getattr(env, "use_obstacles", False) and env.obstacles is not None:
        obstacle_radius_px = max(2, int(round(0.5 * float(env.obst_size) * scale)))
        for obstacle in env.obstacles.pos_arr:
            center = world_to_panel(obstacle[:2], env.room_dims, panel_w, panel_h)[0:2]
            cv2.circle(panel, center, obstacle_radius_px, (71, 119, 91), -1, cv2.LINE_AA)
            cv2.circle(panel, center, obstacle_radius_px, (33, 78, 53), 1, cv2.LINE_AA)

    current_goal = np.asarray(env.envs[drone_index].goal, dtype=np.float32)
    goal_px = world_to_panel(current_goal[:2], env.room_dims, panel_w, panel_h)[0:2]
    cv2.circle(panel, goal_px, 7, (55, 169, 88), -1, cv2.LINE_AA)
    cv2.circle(panel, goal_px, 11, (55, 169, 88), 2, cv2.LINE_AA)

    final_goals = getattr(env, "final_goals", None)
    if final_goals is not None and final_goals[drone_index] is not None:
        final_goal = np.asarray(final_goals[drone_index], dtype=np.float32)
        final_px = world_to_panel(final_goal[:2], env.room_dims, panel_w, panel_h)[0:2]
        marker_type = getattr(cv2, "MARKER_DIAMOND", cv2.MARKER_CROSS)
        cv2.drawMarker(panel, final_px, (39, 111, 191), markerType=marker_type, markerSize=16, thickness=2)

    if len(trajectory) >= 2:
        pts = np.array(
            [world_to_panel(point[:2], env.room_dims, panel_w, panel_h)[0:2] for point in trajectory],
            dtype=np.int32,
        )
        cv2.polylines(panel, [pts], False, (227, 132, 40), 3, cv2.LINE_AA)
        cv2.polylines(panel, [pts], False, (255, 207, 112), 1, cv2.LINE_AA)

    dyn = env.envs[drone_index].dynamics
    pos = np.asarray(dyn.pos, dtype=np.float32)
    drone_px = world_to_panel(pos[:2], env.room_dims, panel_w, panel_h)[0:2]
    cv2.circle(panel, drone_px, 8, (40, 57, 69), -1, cv2.LINE_AA)
    cv2.circle(panel, drone_px, 11, (255, 255, 255), 2, cv2.LINE_AA)

    heading = np.asarray(dyn.rot[:, 0], dtype=np.float32)
    heading_xy = heading[:2]
    heading_norm = float(np.linalg.norm(heading_xy))
    if heading_norm > 1e-6:
        heading_xy = heading_xy / heading_norm
        arrow_tip = (
            int(round(drone_px[0] + heading_xy[0] * 22.0)),
            int(round(drone_px[1] - heading_xy[1] * 22.0)),
        )
        cv2.arrowedLine(panel, drone_px, arrow_tip, (40, 57, 69), 2, cv2.LINE_AA, tipLength=0.35)

    cv2.putText(panel, "trajectory", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(panel, "trajectory", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(
        panel,
        f"z={pos[2]:.2f}m trail={len(trajectory)}",
        (12, panel_h - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (40, 57, 69),
        1,
        cv2.LINE_AA,
    )
    return panel


def compose_trajectory_view(frame_rgb: np.ndarray, trajectory_panel: np.ndarray):
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    if trajectory_panel.shape[:2] != frame_bgr.shape[:2]:
        trajectory_panel = cv2.resize(
            trajectory_panel,
            (frame_bgr.shape[1], frame_bgr.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    return np.concatenate((frame_bgr, trajectory_panel), axis=1)


def main(argv=None):
    register_swarm_components()
    cfg = parse_viewer_cfg(argv=argv)

    viewer_overrides = dict(
        device=cfg.device,
        load_checkpoint_kind=cfg.load_checkpoint_kind,
        quads_obstacle_obs_type=cfg.quads_obstacle_obs_type,
        quads_depth_grid_width=cfg.quads_depth_grid_width,
        quads_depth_grid_height=cfg.quads_depth_grid_height,
        quads_depth_min_distance=cfg.quads_depth_min_distance,
        quads_depth_max_distance=cfg.quads_depth_max_distance,
        quads_lidar_num_rays=cfg.quads_lidar_num_rays,
        quads_depth_noise_std=cfg.quads_depth_noise_std,
        quads_depth_dropout_prob=cfg.quads_depth_dropout_prob,
        quads_depth_normalize=cfg.quads_depth_normalize,
        quads_camera_width=cfg.quads_camera_width,
        quads_camera_height=cfg.quads_camera_height,
        quads_camera_fov=cfg.quads_camera_fov,
        quads_camera_pitch_deg=cfg.quads_camera_pitch_deg,
        quads_camera_drone_index=cfg.quads_camera_drone_index,
        quads_view_mode=["chase"],
        quads_render=False,
        quads_use_numba=False,
        fps=cfg.fps,
        max_num_episodes=cfg.max_num_episodes,
        eval_deterministic=cfg.eval_deterministic,
        viewer_fps=cfg.viewer_fps,
        display=cfg.display,
        display_scale=cfg.display_scale,
        output_video=cfg.output_video,
        hud=cfg.hud,
        trajectory_history=cfg.trajectory_history,
    )

    cfg = load_from_checkpoint(cfg)
    for key, value in viewer_overrides.items():
        setattr(cfg, key, value)

    eval_env_frameskip = cfg.env_frameskip if cfg.eval_env_frameskip is None else cfg.eval_env_frameskip
    assert cfg.env_frameskip % eval_env_frameskip == 0
    render_action_repeat = cfg.env_frameskip // eval_env_frameskip
    cfg.env_frameskip = cfg.eval_env_frameskip = eval_env_frameskip
    cfg.num_envs = 1

    env = make_env_func_batched(
        cfg, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0), render_mode="rgb_array"
    )
    env_info = extract_env_info(env, cfg)

    actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
    actor_critic.eval()

    device = torch.device("cpu" if cfg.device == "cpu" else "cuda")
    actor_critic.model_to_device(device)

    policy_id = cfg.policy_index
    name_prefix = dict(latest="checkpoint", best="best")[cfg.load_checkpoint_kind]
    checkpoints = Learner.get_checkpoints(Learner.checkpoint_dir(cfg, policy_id), f"{name_prefix}_*")
    checkpoint_dict = Learner.load_checkpoint(checkpoints, device)
    actor_critic.load_state_dict(checkpoint_dict["model"])

    obs, infos = env.reset()
    rnn_states = torch.zeros([env.num_agents, get_rnn_size(cfg)], dtype=torch.float32, device=device)
    episode_reward = None
    episode_rewards = [deque([], maxlen=20) for _ in range(env.num_agents)]
    num_episodes = 0
    num_frames = 0

    frame_w = int(cfg.quads_camera_width)
    frame_h = int(cfg.quads_camera_height)
    writer, video_path = open_video_writer(cfg.output_video, frame_w * 2, frame_h, cfg.viewer_fps)

    last_frame_time = time.time()
    fps_smooth = None
    trajectory = deque([], maxlen=max(int(cfg.trajectory_history), 2))
    last_chase_frame = None

    with torch.no_grad():
        while num_episodes < cfg.max_num_episodes:
            normalized_obs = prepare_and_normalize_obs(actor_critic, obs)
            policy_outputs = actor_critic(normalized_obs, rnn_states)

            actions = policy_outputs["actions"]
            if cfg.eval_deterministic:
                action_distribution = actor_critic.action_distribution()
                actions = argmax_actions(action_distribution)

            if actions.ndim == 1:
                actions = unsqueeze_tensor(actions, dim=-1)
            actions = preprocess_actions(env_info, actions)
            rnn_states = policy_outputs["new_rnn_states"]

            for _ in range(render_action_repeat):
                obs, rew, terminated, truncated, infos = env.step(actions)
                dones = make_dones(terminated, truncated).cpu().numpy()
                rew = rew.float()
                if episode_reward is None:
                    episode_reward = rew.clone()
                else:
                    episode_reward += rew

                env_unwrapped = env.unwrapped
                if np.any(dones):
                    trajectory.clear()
                trajectory.append(
                    np.asarray(
                        env_unwrapped.envs[cfg.quads_camera_drone_index].dynamics.pos,
                        dtype=np.float32,
                    ).copy()
                )

                frame_rgb = env_unwrapped.render()
                if frame_rgb is not None:
                    last_chase_frame = frame_rgb
                elif last_chase_frame is not None:
                    frame_rgb = last_chase_frame
                else:
                    frame_rgb = env_unwrapped.get_drone_fpv_image(drone_index=cfg.quads_camera_drone_index)
                if frame_rgb.shape[:2] != (frame_h, frame_w):
                    frame_rgb = cv2.resize(frame_rgb, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)
                trajectory_panel = draw_trajectory_panel(
                    env_unwrapped,
                    drone_index=cfg.quads_camera_drone_index,
                    trajectory=trajectory,
                    panel_w=frame_w,
                    panel_h=frame_h,
                )
                frame_bgr = compose_trajectory_view(
                    frame_rgb,
                    trajectory_panel,
                )

                now = time.time()
                frame_dt = max(now - last_frame_time, 1e-6)
                current_fps = 1.0 / frame_dt
                fps_smooth = current_fps if fps_smooth is None else 0.9 * fps_smooth + 0.1 * current_fps
                last_frame_time = now

                if cfg.hud:
                    reward_value = float(episode_reward[0].item()) if episode_reward is not None and env.num_agents > 0 else 0.0
                    hud_lines = [
                        f"policy={cfg.experiment} checkpoint={cfg.load_checkpoint_kind}",
                        f"obs={cfg.quads_obstacle_obs_type} left=chase right=trajectory trail={len(trajectory)} fps={fps_smooth:.1f}",
                        f"episode={num_episodes + 1}/{cfg.max_num_episodes} env_frames={num_frames} reward={reward_value:.2f}",
                    ]
                    draw_hud(frame_bgr, hud_lines)

                if writer is not None:
                    writer.write(frame_bgr)

                if cfg.display:
                    if cfg.display_scale != 1.0:
                        disp = cv2.resize(
                            frame_bgr,
                            (int(frame_bgr.shape[1] * cfg.display_scale), int(frame_bgr.shape[0] * cfg.display_scale)),
                            interpolation=cv2.INTER_LINEAR,
                        )
                    else:
                        disp = frame_bgr
                    cv2.imshow("Chase + Trajectory Policy Viewer", disp)
                    if cv2.waitKey(1) & 0xFF == 27:
                        num_episodes = cfg.max_num_episodes
                        break

                num_frames += 1

                for agent_i, done_flag in enumerate(dones):
                    if done_flag:
                        episode_rewards[agent_i].append(float(episode_reward[agent_i].item()))
                        episode_reward[agent_i] = 0.0
                        rnn_states[agent_i] = torch.zeros([get_rnn_size(cfg)], dtype=torch.float32, device=device)
                        num_episodes += 1
                        if len(episode_rewards[agent_i]) > 0:
                            log.info(
                                "Episode %d finished, agent %d reward %.3f, recent mean %.3f",
                                num_episodes,
                                agent_i,
                                episode_rewards[agent_i][-1],
                                float(np.mean(episode_rewards[agent_i])),
                            )

                target_delay = 1.0 / max(cfg.viewer_fps, 1)
                if frame_dt < target_delay:
                    time.sleep(target_delay - frame_dt)

                if num_episodes >= cfg.max_num_episodes:
                    break

    env.close()
    if writer is not None:
        writer.release()
    if cfg.display:
        cv2.destroyAllWindows()

    if video_path is not None:
        print(f"Saved viewer video to: {video_path}")


if __name__ == "__main__":
    main()
