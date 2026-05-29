from sample_factory.utils.utils import str2bool


def quadrotors_override_defaults(env, parser):
    parser.set_defaults(
        encoder_type='mlp',
        encoder_subtype='mlp_quads',
        rnn_size=256,
        encoder_extra_fc_layers=0,
        env_frameskip=1,
    )


# noinspection PyUnusedLocal
def add_quadrotors_env_args(env, parser):
    p = parser

    # Quadrotor features
    p.add_argument('--quads_num_agents', default=8, type=int, help='Override default value for the number of quadrotors')
    p.add_argument('--quads_obs_repr', default='xyz_vxyz_R_omega', type=str,
                   choices=['xyz_vxyz_R_omega', 'xyz_vxyz_R_omega_floor', 'xyz_vxyz_R_omega_wall'],
                   help='obs space for quadrotor self')
    p.add_argument('--quads_episode_duration', default=15.0, type=float,
                   help='Override default value for episode duration')
    p.add_argument('--quads_sim_freq', default=200.0, type=float,
                   help='Low-level physics simulation frequency in Hz')
    p.add_argument('--quads_sim_steps', default=2, type=int,
                   help='Number of low-level physics steps per RL/control step')
    p.add_argument('--quads_encoder_type', default="corl", type=str, help='The type of the neighborhood encoder')
    p.add_argument('--quads_control_mode', default='raw', type=str,
                   choices=['raw', 'velocity', 'velocity_yaw', 'velocity_attitude'],
                   help='Action interface: raw motor thrust, velocity, body velocity plus yaw rate, or velocity plus attitude')
    p.add_argument('--quads_velocity_xy_max', default=2.0, type=float,
                   help='Maximum commanded speed in x/y when using velocity control')
    p.add_argument('--quads_velocity_z_max', default=1.0, type=float,
                   help='Maximum commanded speed in z when using velocity control')
    p.add_argument('--quads_velocity_max_tilt_deg', default=35.0, type=float,
                   help='Maximum tilt angle for the internal velocity controller')
    p.add_argument('--quads_velocity_max_acc_xy', default=6.0, type=float,
                   help='Maximum horizontal acceleration for the internal velocity controller')
    p.add_argument('--quads_velocity_max_acc_z_up', default=4.0, type=float,
                   help='Maximum upward acceleration above hover for the internal velocity controller')
    p.add_argument('--quads_velocity_max_acc_z_down', default=4.0, type=float,
                   help='Maximum downward acceleration below hover for the internal velocity controller')
    p.add_argument('--quads_velocity_yaw_mode', default='keep', type=str,
                   choices=['keep', 'velocity', 'goal', 'velocity_or_goal'],
                   help='How the internal velocity controller chooses camera/body yaw')
    p.add_argument('--quads_velocity_yaw_min_speed', default=0.15, type=float,
                   help='Minimum horizontal speed/command used for velocity-based yaw alignment')
    p.add_argument('--quads_velocity_yaw_rate_max', default=0.0, type=float,
                   help='Yaw rate limit in rad/s for automatic alignment or policy yaw control')
    p.add_argument('--quads_velocity_yaw_control_scale', default=1.0, type=float,
                   help='Multiplier for yaw attitude correction in the internal velocity controller')
    p.add_argument('--quads_velocity_command_smoothing_tau', default=0.0, type=float,
                   help='First-order smoothing time constant for xyz velocity commands. 0 disables smoothing')
    p.add_argument('--quads_controller_obstacle_avoidance', default=False, type=str2bool,
                   help='Inject a bounded repulsive velocity before the velocity controller near obstacles')
    p.add_argument('--quads_obstacle_avoidance_distance', default=1.2, type=float,
                   help='Obstacle clearance where controller-level repulsive velocity starts')
    p.add_argument('--quads_obstacle_avoidance_max_speed', default=0.8, type=float,
                   help='Maximum controller-level repulsive velocity in m/s')
    p.add_argument('--quads_obstacle_avoidance_gain', default=1.2, type=float,
                   help='Exponent for how quickly repulsive velocity grows as clearance shrinks')
    p.add_argument('--quads_obstacle_avoidance_pid_kp', default=1.0, type=float,
                   help='PID proportional gain for controller-level obstacle avoidance')
    p.add_argument('--quads_obstacle_avoidance_pid_ki', default=0.0, type=float,
                   help='PID integral gain for controller-level obstacle avoidance')
    p.add_argument('--quads_obstacle_avoidance_pid_kd', default=0.0, type=float,
                   help='PID derivative gain for controller-level obstacle avoidance')
    p.add_argument('--quads_obstacle_avoidance_pid_integral_limit', default=1.0, type=float,
                   help='Integral windup limit for controller-level obstacle avoidance')
    p.add_argument('--quads_goal_ball_capture_assist', default=False, type=str2bool,
                   help='Damp tangential velocity and add mild target-directed velocity near reward balls')
    p.add_argument('--quads_goal_ball_capture_assist_distance', default=1.2, type=float,
                   help='Distance from reward ball where capture assist starts')
    p.add_argument('--quads_goal_ball_capture_assist_speed', default=0.8, type=float,
                   help='Minimum target-directed speed near reward balls')
    p.add_argument('--quads_goal_ball_tangent_damping', default=0.25, type=float,
                   help='Tangential velocity damping strength near reward balls')
    p.add_argument('--quads_velocity_attitude_max_angle_deg', default=45.0, type=float,
                   help='Maximum absolute roll/pitch command for velocity_attitude control')
    p.add_argument('--quads_velocity_attitude_blend', default=1.0, type=float,
                   help='Blend from velocity-derived attitude to policy attitude for velocity_attitude control')

    # Neighbor
    # Neighbor Features
    p.add_argument('--quads_neighbor_visible_num', default=-1, type=int, help='Number of neighbors to consider. -1=all '
                                                                          '0=blind agents, '
                                                                          '0<n<num_agents-1 = nonzero number of agents')
    p.add_argument('--quads_neighbor_obs_type', default='none', type=str,
                   choices=['none', 'pos_vel'], help='Choose what kind of obs to send to encoder.')

    # # Neighbor Encoder
    p.add_argument('--quads_neighbor_hidden_size', default=256, type=int,
                   help='The hidden size for the neighbor encoder')
    p.add_argument('--quads_neighbor_encoder_type', default='attention', type=str,
                   choices=['attention', 'mean_embed', 'mlp', 'no_encoder'],
                   help='The type of the neighborhood encoder')

    # # Neighbor Collision Reward
    p.add_argument('--quads_collision_reward', default=0.0, type=float,
                   help='Override default value for quadcol_bin reward, which means collisions between quadrotors')
    p.add_argument('--quads_collision_hitbox_radius', default=2.0, type=float,
                   help='When the distance between two drones are less than N arm_length, we would view them as '
                        'collide.')
    p.add_argument('--quads_collision_falloff_radius', default=-1.0, type=float,
                   help='The falloff radius for the smooth penalty. -1.0: no smooth penalty')
    p.add_argument('--quads_collision_smooth_max_penalty', default=10.0, type=float,
                   help='The upper bound of the collision function given distance among drones')

    # Obstacle
    # # Obstacle Features
    p.add_argument('--quads_use_obstacles', default=False, type=str2bool, help='Use obstacles or not')
    p.add_argument('--quads_obstacle_obs_type', default='none', type=str,
                   choices=['none', 'octomap', 'depth', 'lidar', 'yolo'], help='Choose what kind of obs to send to encoder.')
    p.add_argument('--quads_obst_density', default=0.2, type=float, help='Obstacle density in the map')
    p.add_argument('--quads_obst_size', default=1.0, type=float, help='The radius of obstacles')
    p.add_argument('--quads_obst_spawn_area', nargs='+', default=[6.0, 6.0], type=float,
                   help='The spawning area of obstacles')
    p.add_argument('--quads_goal_z_min', default=1.0, type=float,
                   help='Lower bound for sampled obstacle-navigation cruise altitude')
    p.add_argument('--quads_goal_z_max', default=3.0, type=float,
                   help='Upper bound for sampled obstacle-navigation cruise altitude')
    p.add_argument('--quads_domain_random', default=False, type=str2bool, help='Use domain randomization or not')
    p.add_argument('--quads_obst_density_random', default=False, type=str2bool, help='Enable obstacle density randomization or not')
    p.add_argument('--quads_obst_density_min', default=0.05, type=float,
                   help='The minimum of obstacle density when enabling domain randomization')
    p.add_argument('--quads_obst_density_max', default=0.2, type=float,
                   help='The maximum of obstacle density when enabling domain randomization')
    p.add_argument('--quads_obst_size_random', default=False, type=str2bool, help='Enable obstacle size randomization or not')
    p.add_argument('--quads_obst_size_min', default=0.3, type=float,
                   help='The minimum obstacle size when enabling domain randomization')
    p.add_argument('--quads_obst_size_max', default=0.6, type=float,
                   help='The maximum obstacle size when enabling domain randomization')

    # # Obstacle Encoder
    p.add_argument('--quads_obst_hidden_size', default=256, type=int, help='The hidden size for the obstacle encoder')
    p.add_argument('--quads_obst_encoder_type', default='mlp', type=str, help='The type of the obstacle encoder')

    # # Obstacle Collision Reward
    p.add_argument('--quads_obst_collision_reward', default=0.0, type=float,
                   help='Override default value for quadcol_bin_obst reward, which means collisions between quadrotor '
                        'and obstacles')
    p.add_argument('--quads_reward_progress', default=0.0, type=float,
                   help='Dense reward for reducing distance to the current target')
    p.add_argument('--quads_reward_action_change', default=0.0, type=float,
                   help='Penalty for abrupt command changes')
    p.add_argument('--quads_reward_vertical_velocity', default=0.0, type=float,
                   help='Penalty for large vertical speed')
    p.add_argument('--quads_reward_height_error', default=0.0, type=float,
                   help='Penalty for altitude error relative to the active target')
    p.add_argument('--quads_reward_thrust', default=0.0, type=float,
                   help='Penalty for sustained high thrust usage')
    p.add_argument('--quads_reward_stagnation', default=0.0, type=float,
                   help='Penalty for staying almost still while far from the target')
    p.add_argument('--quads_reward_overspeed', default=0.0, type=float,
                   help='Penalty for horizontal speed beyond the configured velocity command limit')
    p.add_argument('--quads_reward_safe_flight', default=0.0, type=float,
                   help='Small positive reward for maintaining a collision-free safe-clearance streak')
    p.add_argument('--quads_reward_path_alignment', default=0.0, type=float,
                   help='Penalty for orbiting sideways or moving away from the active target in open space')
    p.add_argument('--quads_reward_obstacle_proximity', default=0.0, type=float,
                   help='Penalty for flying inside the obstacle safety margin')
    p.add_argument('--quads_reward_obstacle_clearance_delta', default=0.0, type=float,
                   help='Reward for increasing clearance from obstacles and penalty for decreasing it')
    p.add_argument('--quads_obstacle_safe_distance', default=1.0, type=float,
                   help='Clearance in meters where the obstacle proximity penalty starts')
    p.add_argument('--quads_obstacle_guard_distance', default=0.0, type=float,
                   help='Terminate before pillar contact once obstacle clearance falls below this margin')
    p.add_argument('--quads_obstacle_guard_terminate', default=True, type=str2bool,
                   help='End the episode immediately when obstacle guard distance is violated')
    p.add_argument('--quads_obst_collision_terminate', default=False, type=str2bool,
                   help='End the episode immediately when a quadrotor hits a pillar')
    p.add_argument('--quads_wall_collision_reward', default=0.0, type=float,
                   help='Penalty for hitting a horizontal room wall')
    p.add_argument('--quads_reward_wall_proximity', default=0.0, type=float,
                   help='Penalty for flying inside the configured room-wall safety margin')
    p.add_argument('--quads_reward_wall_clearance_delta', default=0.0, type=float,
                   help='Reward for increasing clearance from room walls and penalty for decreasing it')
    p.add_argument('--quads_wall_safe_distance', default=0.8, type=float,
                   help='Distance from a horizontal room wall where the wall proximity penalty starts')
    p.add_argument('--quads_wall_guard_distance', default=0.0, type=float,
                   help='Terminate before wall contact once wall clearance falls below this margin')
    p.add_argument('--quads_wall_guard_terminate', default=True, type=str2bool,
                   help='End the episode immediately when wall guard distance is violated')
    p.add_argument('--quads_wall_collision_terminate', default=False, type=str2bool,
                   help='End the episode immediately when a quadrotor hits a horizontal room wall')
    p.add_argument('--quads_use_goal_ball', default=False, type=str2bool,
                   help='Use a temporary reward ball as an intermediate visible target')
    p.add_argument('--quads_goal_ball_reward', default=0.0, type=float,
                   help='Bonus reward for collecting the intermediate reward ball')
    p.add_argument('--quads_goal_ball_radius', default=0.4, type=float,
                   help='Collection radius of the intermediate reward ball')
    p.add_argument('--quads_goal_ball_count', default=1, type=int,
                   help='Number of sequential intermediate reward balls to spawn during stage-1 curriculum')
    p.add_argument('--quads_goal_ball_velocity_reset', default=False, type=str2bool,
                   help='Damp horizontal velocity immediately after collecting a goal ball')
    p.add_argument('--quads_goal_ball_velocity_reset_ratio', default=0.0, type=float,
                   help='Horizontal velocity multiplier after collecting a goal ball')

    # Aerodynamics
    # # Downwash
    p.add_argument('--quads_use_downwash', default=False, type=str2bool, help='Apply downwash or not')

    # Numba Speed Up
    p.add_argument('--quads_use_numba', default=False, type=str2bool, help='Whether to use numba for jit or not')

    # Scenarios
    p.add_argument('--quads_mode', default='static_same_goal', type=str,
                   choices=['static_same_goal', 'static_diff_goal', 'dynamic_same_goal', 'dynamic_diff_goal',
                            'ep_lissajous3D', 'ep_rand_bezier', 'swarm_vs_swarm', 'swap_goals', 'dynamic_formations',
                            'mix', 'o_uniform_same_goal_spawn', 'o_random',
                            'o_dynamic_diff_goal', 'o_dynamic_same_goal', 'o_diagonal', 'o_static_same_goal',
                            'o_static_diff_goal', 'o_swap_goals', 'o_ep_rand_bezier'],
                   help='Choose which scenario to run. ep = evader pursuit')

    # Room
    p.add_argument('--quads_room_dims', nargs='+', default=[10., 10., 10.], type=float,
                   help='Length, width, and height dimensions respectively of the quadrotor env')

    # Replay Buffer
    p.add_argument('--replay_buffer_sample_prob', default=0.0, type=float,
                   help='Probability at which we sample from it rather than resetting the env. Set to 0.0 (default) '
                        'to disable the replay. Set to value in (0.0, 1.0] to use replay buffer')

    # Annealing
    p.add_argument('--anneal_collision_steps', default=0.0, type=float, help='Anneal collision penalties over this '
                                                                             'many steps. Default (0.0) is no '
                                                                             'annealing')

    # Rendering
    p.add_argument('--quads_view_mode', nargs='+', default=['topdown', 'chase', 'global'],
                   type=str, choices=['topdown', 'chase', 'side', 'global', 'corner0', 'corner1', 'corner2', 'corner3', 'topdownfollow', 'fpv'],
                   help='Choose which kind of view/camera to use')
    p.add_argument('--quads_render', default=False, type=str2bool, help='Use render or not')
    p.add_argument('--quads_camera_width', default=320, type=int,
                   help='Width of the drone first-person RGB camera image')
    p.add_argument('--quads_camera_height', default=240, type=int,
                   help='Height of the drone first-person RGB camera image')
    p.add_argument('--quads_camera_fov', default=90.0, type=float,
                   help='Field of view in degrees for the drone first-person camera')
    p.add_argument('--quads_camera_pitch_deg', default=25.0, type=float,
                   help='How many degrees the drone first-person camera points downward from the forward axis')
    p.add_argument('--quads_camera_drone_index', default=0, type=int,
                   help='Which drone index to use when rendering the first-person camera')
    p.add_argument('--quads_depth_grid_width', default=3, type=int,
                   help='Depth observation grid width. Keep 3 for warm-start compatibility with the 9D octomap model.')
    p.add_argument('--quads_depth_grid_height', default=3, type=int,
                   help='Depth observation grid height. Keep 3 for warm-start compatibility with the 9D octomap model.')
    p.add_argument('--quads_depth_min_distance', default=0.05, type=float,
                   help='Minimum valid depth value in meters')
    p.add_argument('--quads_depth_max_distance', default=10.0, type=float,
                   help='Maximum depth value in meters used for no-return rays')
    p.add_argument('--quads_lidar_num_rays', default=9, type=int,
                   help='Number of horizontal 360-degree lidar rays when quads_obstacle_obs_type=lidar')
    p.add_argument('--quads_depth_noise_std', default=0.03, type=float,
                   help='Gaussian depth noise standard deviation in meters for sim-to-real robustness')
    p.add_argument('--quads_depth_dropout_prob', default=0.02, type=float,
                   help='Probability of replacing a depth ray with max distance')
    p.add_argument('--quads_depth_normalize', default=False, type=str2bool,
                   help='Normalize depth rays to [0, 1]. Leave False when warm-starting from the current octomap model.')
    p.add_argument('--quads_yolo_source', default='oracle', type=str, choices=['oracle', 'oracle_mask', 'detector'],
                   help='Use projected oracle boxes, pixel-mask oracle boxes, or a real YOLO detector on the FPV image')
    p.add_argument('--quads_yolo_model_path', default='', type=str,
                   help='Path to an ultralytics YOLO model for obstacle detection when quads_yolo_source=detector')
    p.add_argument('--quads_yolo_conf_threshold', default=0.25, type=float,
                   help='Confidence threshold for YOLO obstacle detections')
    p.add_argument('--visualize_v_value', action='store_true', help="Visualize v value map")

    # Sim2Real
    p.add_argument('--quads_sim2real', default=False, type=str2bool, help='Whether to use sim2real or not')
