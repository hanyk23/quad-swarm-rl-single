import copy

import torch
from sample_factory.algo.learning.learner import Learner
from sample_factory.model.actor_critic import create_actor_critic

from gym_art.quadrotor_multi.quad_experience_replay import ExperienceReplayWrapper
from swarm_rl.env_wrappers.compatibility import QuadEnvCompatibility
from swarm_rl.env_wrappers.projection_map import ProjectionMapWrapper
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
    raw_control = raw_control_zero_middle = True

    sampler_1 = None
    if dyn_randomization_ratio is not None:
        sampler_1 = dict(type='RelativeSampler', noise_ratio=dyn_randomization_ratio, sampler='normal')

    sense_noise = 'default'
    dynamics_change = dict(noise=dict(thrust_noise_ratio=0.05), damp=dict(vel=0, omega_quadratic=0))

    rew_coeff = DEFAULT_QUAD_REWARD_SHAPING['quad_rewards']
    use_replay_buffer = cfg.replay_buffer_sample_prob > 0.0

    env = QuadrotorEnvMulti(
        num_agents=cfg.quads_num_agents, ep_time=cfg.quads_episode_duration, rew_coeff=rew_coeff,
        obs_repr=cfg.quads_obs_repr,
        # Neighbor
        neighbor_visible_num=cfg.quads_neighbor_visible_num, neighbor_obs_type=cfg.quads_neighbor_obs_type,
        collision_hitbox_radius=cfg.quads_collision_hitbox_radius,
        collision_falloff_radius=cfg.quads_collision_falloff_radius,
        # Obstacle
        use_obstacles=cfg.quads_use_obstacles, obst_density=cfg.quads_obst_density, obst_size=cfg.quads_obst_size,
        obst_spawn_area=cfg.quads_obst_spawn_area,
        obstacle_scan_resolution=cfg.quads_obstacle_scan_resolution,
        obstacle_obs_type=cfg.quads_obstacle_obs_type,

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
        # Quadrotor Specific (Do Not Change)
        dynamics_params=quad, raw_control=raw_control, raw_control_zero_middle=raw_control_zero_middle,
        dynamics_randomize_every=dyn_randomize_every, dynamics_change=dynamics_change, dyn_sampler_1=sampler_1,
        sense_noise=sense_noise, init_random_state=False, control_type=cfg.quads_control_type,
        velocity_yaw_max_speed=cfg.quads_velocity_yaw_max_speed,
        velocity_yaw_avoid_radius=cfg.quads_velocity_yaw_avoid_radius,
        velocity_yaw_avoid_gain=cfg.quads_velocity_yaw_avoid_gain,
        velocity_yaw_avoid_pid_kp=cfg.quads_velocity_yaw_avoid_pid_kp,
        velocity_yaw_avoid_pid_ki=cfg.quads_velocity_yaw_avoid_pid_ki,
        velocity_yaw_avoid_pid_kd=cfg.quads_velocity_yaw_avoid_pid_kd,
        velocity_yaw_avoid_integral_limit=cfg.quads_velocity_yaw_avoid_integral_limit,
        velocity_yaw_avoid_max_bias=cfg.quads_velocity_yaw_avoid_max_bias,
        quads_render_speed=cfg.quads_render_speed,
        quads_render_width=cfg.quads_render_width,
        quads_render_height=cfg.quads_render_height,
        # Rendering
        render_mode=render_mode,
    )

    if use_replay_buffer:
        env = ExperienceReplayWrapper(env, cfg.replay_buffer_sample_prob, cfg.quads_obst_density, cfg.quads_obst_size,
                                      cfg.quads_domain_random, cfg.quads_obst_density_random, cfg.quads_obst_size_random,
                                      cfg.quads_obst_density_min, cfg.quads_obst_density_max, cfg.quads_obst_size_min, cfg.quads_obst_size_max)

    reward_shaping = copy.deepcopy(DEFAULT_QUAD_REWARD_SHAPING)

    reward_shaping['quad_rewards']['quadcol_bin'] = cfg.quads_collision_reward
    reward_shaping['quad_rewards']['quadcol_bin_smooth_max'] = cfg.quads_collision_smooth_max_penalty
    reward_shaping['quad_rewards']['quadcol_bin_obst'] = cfg.quads_obst_collision_reward
    reward_shaping['quad_rewards']['orient'] = cfg.quads_orient_reward
    reward_shaping['quad_rewards']['spin'] = cfg.quads_spin_reward
    reward_shaping['quad_rewards']['vel'] = cfg.quads_vel_reward
    reward_shaping['quad_rewards']['vel_limit'] = cfg.quads_vel_penalty_limit
    reward_shaping['quad_rewards']['progress'] = cfg.quads_progress_reward
    reward_shaping['quad_rewards']['success'] = cfg.quads_success_reward
    reward_shaping['quad_rewards']['first_success'] = cfg.quads_first_success_reward
    reward_shaping['quad_rewards']['z'] = cfg.quads_z_reward
    reward_shaping['quad_rewards']['stable_z'] = cfg.quads_stable_z_reward
    reward_shaping['quad_rewards']['stable_spin'] = cfg.quads_stable_spin_reward
    reward_shaping['quad_rewards']['floor_stall'] = cfg.quads_floor_stall_reward
    reward_shaping['quad_rewards']['room_floor'] = cfg.quads_room_floor_reward
    reward_shaping['quad_rewards']['room_wall'] = cfg.quads_room_wall_reward
    reward_shaping['quad_rewards']['room_ceiling'] = cfg.quads_room_ceiling_reward

    # this is annealed by the reward shaping wrapper
    if cfg.anneal_collision_steps > 0:
        reward_shaping['quad_rewards']['quadcol_bin'] = 0.0
        reward_shaping['quad_rewards']['quadcol_bin_smooth_max'] = 0.0
        reward_shaping['quad_rewards']['quadcol_bin_obst'] = 0.0
        reward_shaping['quad_rewards']['orient'] = cfg.quads_orient_reward
        reward_shaping['quad_rewards']['spin'] = cfg.quads_spin_reward
        reward_shaping['quad_rewards']['vel'] = 0.0
        reward_shaping['quad_rewards']['vel_limit'] = cfg.quads_vel_penalty_limit
        reward_shaping['quad_rewards']['progress'] = cfg.quads_progress_reward
        reward_shaping['quad_rewards']['success'] = cfg.quads_success_reward
        reward_shaping['quad_rewards']['first_success'] = cfg.quads_first_success_reward
        reward_shaping['quad_rewards']['z'] = cfg.quads_z_reward
        reward_shaping['quad_rewards']['stable_z'] = cfg.quads_stable_z_reward
        reward_shaping['quad_rewards']['stable_spin'] = cfg.quads_stable_spin_reward
        reward_shaping['quad_rewards']['floor_stall'] = cfg.quads_floor_stall_reward
        reward_shaping['quad_rewards']['room_floor'] = cfg.quads_room_floor_reward
        reward_shaping['quad_rewards']['room_wall'] = cfg.quads_room_wall_reward
        reward_shaping['quad_rewards']['room_ceiling'] = cfg.quads_room_ceiling_reward
        annealing = [
            AnnealSchedule('quadcol_bin', cfg.quads_collision_reward, cfg.anneal_collision_steps),
            AnnealSchedule('quadcol_bin_smooth_max', cfg.quads_collision_smooth_max_penalty,
                           cfg.anneal_collision_steps),
            AnnealSchedule('quadcol_bin_obst', cfg.quads_obst_collision_reward, cfg.anneal_collision_steps),
            AnnealSchedule('vel', cfg.quads_vel_reward, cfg.anneal_collision_steps),
        ]
    else:
        annealing = None

    env = QuadsRewardShapingWrapper(env, reward_shaping_scheme=reward_shaping, annealing=annealing,
                                    with_pbt=cfg.with_pbt)
    env = QuadEnvCompatibility(env)
    env = ProjectionMapWrapper(
        env,
        enabled=cfg.visualize_projection_map or cfg.visualize_obstacle_point_cloud,
        show_obstacle_point_cloud=cfg.visualize_obstacle_point_cloud,
    )

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
