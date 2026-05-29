from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import Tensor

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
from swarm_rl.vision.yolo_obstacles import annotate_detections


def parse_viewer_cfg(argv=None):
    parser, partial_cfg = parse_sf_args(argv=argv, evaluation=True)
    for action in parser._actions:
        if action.dest == "env":
            action.required = False
    add_quadrotors_env_args(partial_cfg.env, parser)
    quadrotors_override_defaults(partial_cfg.env, parser)

    parser.add_argument("--display", default=True, type=str2bool, help="Show an OpenCV window")
    parser.add_argument("--display_scale", default=1.0, type=float, help="Resize the viewer window by this factor")
    parser.add_argument("--output_video", default="", type=str, help="Optional MP4 path for saving the annotated FPV")
    parser.add_argument("--viewer_fps", default=30, type=int, help="Target playback FPS for the viewer")
    parser.add_argument("--hud", default=True, type=str2bool, help="Show text overlays such as fps and detections")

    parser.set_defaults(
        env="quadrotor_multi",
        algo="APPO",
        quads_view_mode=["fpv"],
        quads_render=False,
        quads_use_numba=False,
        quads_camera_width=640,
        quads_camera_height=480,
        quads_camera_fov=100.0,
        quads_camera_pitch_deg=25.0,
        quads_yolo_source="oracle_mask",
        eval_deterministic=True,
        fps=30,
        max_num_episodes=10,
        display_scale=1.5,
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


def main(argv=None):
    register_swarm_components()
    cfg = parse_viewer_cfg(argv=argv)

    viewer_overrides = dict(
        device=cfg.device,
        load_checkpoint_kind=cfg.load_checkpoint_kind,
        quads_yolo_source=cfg.quads_yolo_source,
        quads_yolo_model_path=cfg.quads_yolo_model_path,
        quads_yolo_conf_threshold=cfg.quads_yolo_conf_threshold,
        quads_camera_width=cfg.quads_camera_width,
        quads_camera_height=cfg.quads_camera_height,
        quads_camera_fov=cfg.quads_camera_fov,
        quads_camera_pitch_deg=cfg.quads_camera_pitch_deg,
        quads_camera_drone_index=cfg.quads_camera_drone_index,
        quads_view_mode=["fpv"],
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
    writer, video_path = open_video_writer(cfg.output_video, frame_w, frame_h, cfg.viewer_fps)

    last_frame_time = time.time()
    fps_smooth = None

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
                frame_rgb = env_unwrapped.get_drone_fpv_image(drone_index=cfg.quads_camera_drone_index)
                detections = env_unwrapped.get_current_obstacle_detections(
                    drone_index=cfg.quads_camera_drone_index,
                    source=cfg.quads_yolo_source,
                    image_rgb=frame_rgb,
                )
                annotated_rgb = annotate_detections(frame_rgb, detections)
                frame_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

                now = time.time()
                frame_dt = max(now - last_frame_time, 1e-6)
                current_fps = 1.0 / frame_dt
                fps_smooth = current_fps if fps_smooth is None else 0.9 * fps_smooth + 0.1 * current_fps
                last_frame_time = now

                if cfg.hud:
                    reward_value = float(episode_reward[0].item()) if episode_reward is not None and env.num_agents > 0 else 0.0
                    hud_lines = [
                        f"policy={cfg.experiment} checkpoint={cfg.load_checkpoint_kind}",
                        f"source={cfg.quads_yolo_source} detections={len(detections)} fps={fps_smooth:.1f}",
                        f"episode={num_episodes + 1}/{cfg.max_num_episodes} env_frames={num_frames} reward={reward_value:.2f}",
                    ]
                    draw_hud(frame_bgr, hud_lines)

                if writer is not None:
                    writer.write(frame_bgr)

                if cfg.display:
                    if cfg.display_scale != 1.0:
                        disp = cv2.resize(
                            frame_bgr,
                            (int(frame_w * cfg.display_scale), int(frame_h * cfg.display_scale)),
                            interpolation=cv2.INTER_LINEAR,
                        )
                    else:
                        disp = frame_bgr
                    cv2.imshow("YOLO FPV Policy Viewer", disp)
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
