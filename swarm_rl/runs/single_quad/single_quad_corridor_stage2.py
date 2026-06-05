from sample_factory.launcher.run_description import RunDescription, Experiment, ParamGrid
from swarm_rl.runs.single_quad.baseline import QUAD_BASELINE_CLI

_params = ParamGrid([
    ('seed', [0000]),
])

# Stage 2: Dense obstacles, long strip.
# Reuse the exact same experiment directory/name as stage 1 so Sample Factory resumes
# from the latest checkpoint and only overrides the CLI arguments below.
STAGE2_CLI = QUAD_BASELINE_CLI + (
    ' --quads_use_obstacles=True --quads_room_dims 40 10 10 --quads_obst_spawn_area 30 6 --quads_obst_density=0.06 '
    '--quads_obst_size=0.6 --quads_obst_collision_reward=12.0 '
    '--quads_obstacle_obs_type=octomap --quads_use_downwash=False '
    '--quads_obstacle_scan_resolution=0.25 '
    '--quads_mode=o_random --quads_obs_repr=xyz_vxyz_R_omega_wall '
    '--quads_collision_reward=12.0 --quads_collision_smooth_max_penalty=10.0 '
    '--quads_vel_reward=0.6 --quads_vel_penalty_limit=2.3 '
    '--quads_progress_reward=0.8 --quads_success_reward=1.0 '
    '--quads_first_success_reward=10.0 '
    '--exploration_loss_coeff=0.001 '
    '--quads_domain_random=True --quads_obst_density_random=True --quads_obst_density_min=0.03 '
    '--quads_obst_density_max=0.06 --quads_obst_size_random=True --quads_obst_size_min=0.50 '
    '--quads_obst_size_max=0.50 '
    '--save_best_metric=reward '
    '--train_for_env_steps=20000000 --restart_behavior=resume '
    '--save_every_sec=120'
)

_experiment = Experiment(
    'single_corridor_curriculum_v7',
    STAGE2_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('single_quad_corridor_curriculum_v7', experiments=[_experiment])
