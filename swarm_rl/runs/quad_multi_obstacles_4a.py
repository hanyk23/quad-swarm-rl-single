from sample_factory.launcher.run_description import RunDescription, Experiment, ParamGrid

# 使用仓库自带的单机基础配置，然后手动叠加 multi + obstacles 参数（最可靠的方式）
from swarm_rl.runs.single_quad.baseline import QUAD_BASELINE_CLI

_params = ParamGrid([
    ('seed', [1000, 2000, 3000, 4000]),
])

# 4无人机 + 障碍物核心配置
QUAD_MULTI_OBSTACLES_4A_CLI = QUAD_BASELINE_CLI + (
    ' --quads_num_agents=4'
    ' --quads_use_obstacles=True'
    ' --quads_room_dims 12 12 10'
    ' --quads_obst_spawn_area 10 10'
    ' --quads_obst_density=0.28'
    ' --quads_obst_size=0.6'
    ' --quads_obst_collision_reward=1.5'
    ' --quads_obstacle_obs_type=octomap'
    ' --quads_use_downwash=True'
    ' --quads_mode=o_random'
    ' --quads_neighbor_visible_num=3'
    ' --quads_neighbor_obs_type=pos_vel'      # 可选，pos_vel 通常需要这个
)

_experiment = Experiment(
    'multi_obstacles_4a',
    QUAD_MULTI_OBSTACLES_4A_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription(
    'quad_multi_obstacles_4a',
    experiments=[_experiment]
)