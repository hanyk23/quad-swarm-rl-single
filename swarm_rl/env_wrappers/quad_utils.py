import copy

import torch
from sample_factory.algo.learning.learner import Learner
from sample_factory.model.actor_critic import create_actor_critic

from gym_art.quadrotor_multi.quad_experience_replay import ExperienceReplayWrapper
from swarm_rl.env_wrappers.compatibility import QuadEnvCompatibility
from swarm_rl.env_wrappers.reward_shaping import DEFAULT_QUAD_REWARD_SHAPING, QuadsRewardShapingWrapper
from swarm_rl.env_wrappers.v_value_map import V_ValueMapWrapper


class AnnealSchedule:
    def __init__(self, coeff_name, final_value, anneal_env_steps):
        self.coeff_name = coeff_name
        self.final_value = final_value
        self.anneal_env_steps = anneal_env_steps


def make_quadrotor_env_multi(cfg, render_mode=None, **kwargs):
    from gym_art.quadrotor_multi.quadrotor_multi import QuadrotorEnvMulti
    quad = 'Crazyflie'
    dyn_randomize_every = dyn_randomization_ratio = None
    control_mode = cfg.quads_control_mode
    legacy_control_mode = getattr(cfg, "quads_control_type", None)
    if control_mode == 'raw' and legacy_control_mode:
        control_mode = "legacy_velocity_yaw" if legacy_control_mode == "velocity_yaw" else legacy_control_mode

    velocity_max_xy = cfg.quads_velocity_xy_max
    velocity_max_z = cfg.quads_velocity_z_max
    legacy_velocity_yaw_max_speed = getattr(cfg, "quads_velocity_yaw_max_speed", None)
    if control_mode in ("legacy_velocity_yaw", "velocity_yaw_avoid") and legacy_velocity_yaw_max_speed is not None:
        velocity_max_xy = float(legacy_velocity_yaw_max_speed)
        velocity_max_z = float(legacy_velocity_yaw_max_speed)

    controller_obstacle_avoidance = cfg.quads_controller_obstacle_avoidance or control_mode == "velocity_yaw_avoid"
    raw_control = control_mode == 'raw'
    raw_control_zero_middle = True

    sampler_1 = None
    if dyn_randomization_ratio is not None:
        sampler_1 = dict(type='RelativeSampler', noise_ratio=dyn_randomization_ratio, sampler='normal')

    sense_noise = 'default'
    dynamics_change = dict(noise=dict(thrust_noise_ratio=0.05), damp=dict(vel=0, omega_quadratic=0))

    rew_coeff = DEFAULT_QUAD_REWARD_SHAPING['quad_rewards']
    use_replay_buffer = cfg.replay_buffer_sample_prob > 0.0
    use_domain_random_wrapper = (
        cfg.quads_domain_random
        and (cfg.quads_obst_density_random or cfg.quads_obst_size_random)
    )

    env = QuadrotorEnvMulti(
        num_agents=cfg.quads_num_agents, ep_time=cfg.quads_episode_duration, rew_coeff=rew_coeff,
        obs_repr=cfg.quads_obs_repr, sim_freq=cfg.quads_sim_freq, sim_steps=cfg.quads_sim_steps,
        # Neighbor
        neighbor_visible_num=cfg.quads_neighbor_visible_num, neighbor_obs_type=cfg.quads_neighbor_obs_type,
        collision_hitbox_radius=cfg.quads_collision_hitbox_radius,
        collision_falloff_radius=cfg.quads_collision_falloff_radius,
        # Obstacle
        use_obstacles=cfg.quads_use_obstacles, obst_density=cfg.quads_obst_density, obst_size=cfg.quads_obst_size,
        obst_spawn_area=cfg.quads_obst_spawn_area, obstacle_obs_type=cfg.quads_obstacle_obs_type,
        obstacle_safe_distance=cfg.quads_obstacle_safe_distance,
        obstacle_guard_distance=cfg.quads_obstacle_guard_distance,
        terminate_on_obstacle_collision=cfg.quads_obst_collision_terminate,
        wall_safe_distance=cfg.quads_wall_safe_distance,
        wall_guard_distance=cfg.quads_wall_guard_distance,
        terminate_on_wall_collision=cfg.quads_wall_collision_terminate,

        # Aerodynamics
        use_downwash=cfg.quads_use_downwash,
        # Numba Speed Up
        use_numba=cfg.quads_use_numba,
        # Scenarios
        quads_mode=cfg.quads_mode,
        # Room
        room_dims=cfg.quads_room_dims,
        # Replay Buffer
        use_replay_buffer=use_replay_buffer,
        # Rendering
        quads_view_mode=cfg.quads_view_mode, quads_render=cfg.quads_render,
        camera_hw=(cfg.quads_camera_width, cfg.quads_camera_height), camera_fov=cfg.quads_camera_fov,
        camera_pitch_deg=cfg.quads_camera_pitch_deg,
        camera_drone_index=cfg.quads_camera_drone_index,
        depth_grid_hw=(cfg.quads_depth_grid_width, cfg.quads_depth_grid_height),
        depth_min_distance=cfg.quads_depth_min_distance, depth_max_distance=cfg.quads_depth_max_distance,
        lidar_num_rays=cfg.quads_lidar_num_rays,
        depth_noise_std=cfg.quads_depth_noise_std, depth_dropout_prob=cfg.quads_depth_dropout_prob,
        depth_normalize=cfg.quads_depth_normalize,
        # Quadrotor Specific (Do Not Change)
        dynamics_params=quad, raw_control=raw_control, raw_control_zero_middle=raw_control_zero_middle,
        dynamics_randomize_every=dyn_randomize_every, dynamics_change=dynamics_change, dyn_sampler_1=sampler_1,
        sense_noise=sense_noise, init_random_state=False, control_mode=control_mode,
        velocity_max_xy=velocity_max_xy, velocity_max_z=velocity_max_z,
        velocity_max_tilt_deg=cfg.quads_velocity_max_tilt_deg,
        velocity_max_acc_xy=cfg.quads_velocity_max_acc_xy,
        velocity_max_acc_z_up=cfg.quads_velocity_max_acc_z_up,
        velocity_max_acc_z_down=cfg.quads_velocity_max_acc_z_down,
        velocity_yaw_mode=cfg.quads_velocity_yaw_mode,
        velocity_yaw_min_speed=cfg.quads_velocity_yaw_min_speed,
        velocity_yaw_rate_max=cfg.quads_velocity_yaw_rate_max,
        velocity_yaw_control_scale=cfg.quads_velocity_yaw_control_scale,
        velocity_command_smoothing_tau=cfg.quads_velocity_command_smoothing_tau,
        controller_obstacle_avoidance=controller_obstacle_avoidance,
        obstacle_avoidance_distance=cfg.quads_obstacle_avoidance_distance,
        obstacle_avoidance_max_speed=cfg.quads_obstacle_avoidance_max_speed,
        obstacle_avoidance_gain=cfg.quads_obstacle_avoidance_gain,
        obstacle_avoidance_pid_kp=cfg.quads_obstacle_avoidance_pid_kp,
        obstacle_avoidance_pid_ki=cfg.quads_obstacle_avoidance_pid_ki,
        obstacle_avoidance_pid_kd=cfg.quads_obstacle_avoidance_pid_kd,
        obstacle_avoidance_pid_integral_limit=cfg.quads_obstacle_avoidance_pid_integral_limit,
        goal_ball_capture_assist=cfg.quads_goal_ball_capture_assist,
        goal_ball_capture_assist_distance=cfg.quads_goal_ball_capture_assist_distance,
        goal_ball_capture_assist_speed=cfg.quads_goal_ball_capture_assist_speed,
        goal_ball_tangent_damping=cfg.quads_goal_ball_tangent_damping,
        velocity_attitude_max_angle_deg=cfg.quads_velocity_attitude_max_angle_deg,
        velocity_attitude_blend=cfg.quads_velocity_attitude_blend,
        obstacle_guard_terminate=cfg.quads_obstacle_guard_terminate,
        wall_guard_terminate=cfg.quads_wall_guard_terminate,
        goal_z_range=(cfg.quads_goal_z_min, cfg.quads_goal_z_max), use_goal_ball=cfg.quads_use_goal_ball,
        goal_ball_reward=cfg.quads_goal_ball_reward, goal_ball_radius=cfg.quads_goal_ball_radius,
        goal_ball_count=cfg.quads_goal_ball_count,
        goal_ball_velocity_reset=cfg.quads_goal_ball_velocity_reset,
        goal_ball_velocity_reset_ratio=cfg.quads_goal_ball_velocity_reset_ratio,
        yolo_source=cfg.quads_yolo_source,
        yolo_model_path=cfg.quads_yolo_model_path, yolo_conf_threshold=cfg.quads_yolo_conf_threshold,
        # Rendering
        render_mode=render_mode,
    )

    if use_replay_buffer or use_domain_random_wrapper:
        env = ExperienceReplayWrapper(env, cfg.replay_buffer_sample_prob, cfg.quads_obst_density, cfg.quads_obst_size,
                                      cfg.quads_domain_random, cfg.quads_obst_density_random, cfg.quads_obst_size_random,
                                      cfg.quads_obst_density_min, cfg.quads_obst_density_max, cfg.quads_obst_size_min, cfg.quads_obst_size_max)

    reward_shaping = copy.deepcopy(DEFAULT_QUAD_REWARD_SHAPING)

    reward_shaping['quad_rewards']['quadcol_bin'] = cfg.quads_collision_reward
    reward_shaping['quad_rewards']['quadcol_bin_smooth_max'] = cfg.quads_collision_smooth_max_penalty
    reward_shaping['quad_rewards']['quadcol_bin_obst'] = cfg.quads_obst_collision_reward
    reward_shaping['quad_rewards']['progress'] = cfg.quads_reward_progress
    reward_shaping['quad_rewards']['action_change'] = cfg.quads_reward_action_change
    reward_shaping['quad_rewards']['vz'] = cfg.quads_reward_vertical_velocity
    reward_shaping['quad_rewards']['height_error'] = cfg.quads_reward_height_error
    reward_shaping['quad_rewards']['thrust'] = cfg.quads_reward_thrust
    reward_shaping['quad_rewards']['stagnation'] = cfg.quads_reward_stagnation
    reward_shaping['quad_rewards']['overspeed'] = cfg.quads_reward_overspeed
    reward_shaping['quad_rewards']['safe_flight'] = cfg.quads_reward_safe_flight
    reward_shaping['quad_rewards']['path_alignment'] = cfg.quads_reward_path_alignment
    reward_shaping['quad_rewards']['obst_proximity'] = cfg.quads_reward_obstacle_proximity
    reward_shaping['quad_rewards']['obstacle_clearance_delta'] = cfg.quads_reward_obstacle_clearance_delta
    reward_shaping['quad_rewards']['wallcol_bin'] = cfg.quads_wall_collision_reward
    reward_shaping['quad_rewards']['wall_proximity'] = cfg.quads_reward_wall_proximity
    reward_shaping['quad_rewards']['wall_clearance_delta'] = cfg.quads_reward_wall_clearance_delta

    # this is annealed by the reward shaping wrapper
    if cfg.anneal_collision_steps > 0:
        reward_shaping['quad_rewards']['quadcol_bin'] = 0.0
        reward_shaping['quad_rewards']['quadcol_bin_smooth_max'] = 0.0
        reward_shaping['quad_rewards']['quadcol_bin_obst'] = 0.0
        reward_shaping['quad_rewards']['obst_proximity'] = 0.0
        reward_shaping['quad_rewards']['wallcol_bin'] = 0.0
        reward_shaping['quad_rewards']['wall_proximity'] = 0.0
        annealing = [
            AnnealSchedule('quadcol_bin', cfg.quads_collision_reward, cfg.anneal_collision_steps),
            AnnealSchedule('quadcol_bin_smooth_max', cfg.quads_collision_smooth_max_penalty,
                           cfg.anneal_collision_steps),
            AnnealSchedule('quadcol_bin_obst', cfg.quads_obst_collision_reward, cfg.anneal_collision_steps),
            AnnealSchedule('obst_proximity', cfg.quads_reward_obstacle_proximity, cfg.anneal_collision_steps),
            AnnealSchedule('wallcol_bin', cfg.quads_wall_collision_reward, cfg.anneal_collision_steps),
            AnnealSchedule('wall_proximity', cfg.quads_reward_wall_proximity, cfg.anneal_collision_steps),
        ]
    else:
        annealing = None

    env = QuadsRewardShapingWrapper(env, reward_shaping_scheme=reward_shaping, annealing=annealing,
                                    with_pbt=cfg.with_pbt)
    env = QuadEnvCompatibility(env)

    if cfg.visualize_v_value:
        actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
        actor_critic.eval()

        device = torch.device("cpu" if cfg.device == "cpu" else "cuda")
        actor_critic.model_to_device(device)

        policy_id = cfg.policy_index
        name_prefix = dict(latest="checkpoint", best="best")[cfg.load_checkpoint_kind]
        checkpoints = Learner.get_checkpoints(Learner.checkpoint_dir(cfg, policy_id), f"{name_prefix}_*")
        checkpoint_dict = Learner.load_checkpoint(checkpoints, device)
        actor_critic.load_state_dict(checkpoint_dict["model"])
        env = V_ValueMapWrapper(env, actor_critic)

    return env


def make_quadrotor_env(env_name, cfg=None, _env_config=None, render_mode=None, **kwargs):
    if env_name == 'quadrotor_multi':
        return make_quadrotor_env_multi(cfg, render_mode, **kwargs)
    else:
        raise NotImplementedError
