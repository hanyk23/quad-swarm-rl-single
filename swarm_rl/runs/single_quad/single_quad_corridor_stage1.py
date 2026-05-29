from sample_factory.launcher.run_description import RunDescription, Experiment, ParamGrid
from swarm_rl.runs.single_quad.baseline import QUAD_BASELINE_CLI

_params = ParamGrid([
    ('seed', [0000]),
])

# Stage 1: Obstacle-free warmup in the same observation/model space as stage 2.
# Keep obstacle observations enabled, but set density to zero so the checkpoint
# is structurally compatible with stage 2.
STAGE1_CLI = QUAD_BASELINE_CLI + (
    ' --quads_use_obstacles=True --quads_room_dims 40 10 10 --quads_obst_spawn_area 30 6 --quads_obst_density=0.0 '
    '--quads_obst_size=0.6 --quads_obst_collision_reward=1.0 '
    '--quads_obstacle_obs_type=octomap --quads_use_downwash=False '
    '--quads_mode=o_random --quads_obs_repr=xyz_vxyz_R_omega_wall '
    '--restart_behavior=overwrite '
    '--reward_scale=0.2 '
    '--quads_collision_reward=2.0 --quads_collision_smooth_max_penalty=2.0 '
    '--quads_first_success_reward=10.0 --quads_success_reward=1.0 '
    '--quads_floor_stall_reward=1.0 --quads_room_floor_reward=1.0 --quads_room_wall_reward=1.0 '
    '--quads_room_ceiling_reward=1.0 --quads_orient_reward=3.0 --quads_spin_reward=1.0 '
    '--quads_z_reward=0.2 --quads_stable_z_reward=0.2 --quads_stable_spin_reward=0.2 '
    '--train_for_env_steps=5000000 ' # Train for a limited number of steps
    '--save_every_sec=120'
)

_experiment = Experiment(
    'single_corridor_curriculum_v7',
    STAGE1_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('single_quad_corridor_curriculum_v7', experiments=[_experiment])
