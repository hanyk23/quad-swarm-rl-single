from sample_factory.launcher.run_description import RunDescription, Experiment, ParamGrid
from swarm_rl.runs.single_quad.baseline import QUAD_BASELINE_CLI

_params = ParamGrid([
    ('seed', [0000, 1111, 2222, 3333]),
])

QUAD_OBSTACLE_CLI = QUAD_BASELINE_CLI + (
    ' --quads_use_obstacles=True --quads_room_dims 12 12 10 --quads_obst_spawn_area 10 10 --quads_obst_density=0.2 '
    '--quads_obst_size=0.5 --quads_obst_collision_reward=1.0 '
    '--quads_obstacle_obs_type=octomap --quads_use_downwash=False '
    '--quads_mode=o_random --quads_obs_repr=xyz_vxyz_R_omega_wall'
)

_experiment = Experiment(
    'single_obstacles_wall',
    QUAD_OBSTACLE_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('single_quad_obstacles_wall', experiments=[_experiment])