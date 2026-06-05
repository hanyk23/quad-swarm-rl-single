from sample_factory.launcher.run_description import RunDescription, Experiment, ParamGrid
from swarm_rl.runs.single_quad.baseline import QUAD_BASELINE_CLI

_params = ParamGrid([
    ('seed', [0000, 1111, 2222, 3333]),
])

QUAD_OBSTACLE_LIDAR_CLI = QUAD_BASELINE_CLI + (
    ' --quads_use_obstacles=True --quads_room_dims 12 12 10 --quads_obst_spawn_area 10 10 --quads_obst_density=0.12 '
    '--quads_obst_size=0.5 --quads_obst_collision_reward=8.0 '
    '--quads_obstacle_obs_type=lidar --quads_obstacle_scan_resolution=0.15 '
    '--quads_use_downwash=False --quads_vel_penalty_limit=1.6 --quads_velocity_yaw_max_speed=1.2 '
    '--quads_collision_reward=8.0 --quads_collision_smooth_max_penalty=6.0 '
    '--quads_progress_reward=0.8 --quads_success_reward=1.0 --quads_first_success_reward=10.0 '
    '--quads_vel_reward=0.4 --quads_orient_reward=2.0 --quads_spin_reward=0.5 '
    '--quads_mode=o_random --quads_obs_repr=xyz_vxyz_R_omega_wall '
    '--quads_control_type=velocity_yaw_avoid'
)

_experiment = Experiment(
    'single_obstacles_lidar_wall',
    QUAD_OBSTACLE_LIDAR_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('single_quad_obstacles_lidar_wall', experiments=[_experiment])
