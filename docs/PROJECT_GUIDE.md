# 项目说明

## 项目目标

训练单无人机长条形场景的两阶段课程学习模型。

## 当前版本

- 阶段一：`swarm_rl.runs.single_quad.single_quad_corridor_stage1`
- 阶段二：`swarm_rl.runs.single_quad.single_quad_corridor_stage2`
- 新一轮实验名：`single_quad_corridor_curriculum_v7`

## 训练脚本

- `bash train_two_stages_corridor.sh`

## 测试脚本

- `bash test_single_quad_corridor.sh`

## 关键机制

- 阶段一先用 `quads_obst_density=0.0` 做无障碍预热，但保留同样的观测结构，保证阶段二能直接续训。
- 阶段一用 `restart_behavior=overwrite`，阶段二用 `restart_behavior=resume`。
- 阶段二会复用阶段一相同的实验目录，自动加载最新 checkpoint。
- 阶段二现在回到按 `reward` 选 best checkpoint。
- `o_random` 会在无人机进入目标阈值后自动采样新目标，不会在同一个目标点上重复刷 `success`。
- `o_random` 的首个/后续目标奖励由 `QuadrotorEnvMulti.reset()` 在场景重置前注入，避免读取到空的 `rew_coeff`。
- 阶段二把障碍物密度从 `0.08` 调低到 `0.06`，减少硬碰撞概率。
- 阶段二把障碍物 SDF 采样尺度从 `0.2` 放大到 `0.25`，让无人机更早看到障碍物。
- v6 的探索和随机化偏激进，可能导致 reward 越训越低；v7 已回到固定 stddev，并把探索系数降到 `0.001`。
- 阶段二保留障碍物密度/尺寸随机化，但范围更窄；障碍物尺寸上限已降到 `0.50`，用来减少局部最优，同时避免训练目标突然变太难。
- 测试时开启 `visualize_projection_map=True` 可以看到二维投影地图，方便观察路径选择。
- 这轮建议直接从 `v7` 新目录重新开始训练。

## 奖励重点

- 总奖励先由环境原始项算出，再交给 `QuadsRewardShapingWrapper` 记录、统计和 anneal。
- `pos` 是到目标的距离惩罚，越远越亏。
- `effort` 是控制输出惩罚，防止一直猛打电机。
- `orient` 是姿态惩罚，压住机身不要歪得太厉害。
- `spin` 是角速度惩罚，压住持续乱转。
- `vel` 是超速惩罚，只在速度超过 `vel_limit` 后明显生效。
- `z` 是低空惩罚，离地太低会扣分。
- `stable_z` 和 `stable_spin` 是稳定性奖励，鼓励高度和角速度都稳住。
- `progress` 是朝目标方向的速度奖励，飞对方向才有正向反馈。
- `success` 是到达当前目标的一次性奖励，`first_success` 是本回合第一个目标的额外奖励。
- `floor_stall` 是长时间趴地的惩罚。
- `room_floor / room_wall / room_ceiling` 分别是撞地、撞墙、撞顶的惩罚。
- `quadcol_bin` 是机体之间的硬碰撞惩罚。
- `quadcol_bin_smooth_max` 是机体靠太近时的平滑碰撞惩罚上限。
- `quadcol_bin_obst` 是机体撞障碍物的惩罚。
- `o_random` 的目标 z 范围更高，目标点不会老贴着低空刷。
- stage2 的障碍物感知范围更大了，能更早看到障碍物。
- 新增的 `lidar` 障碍物观测会同时感知柱子和场地边界，输出仍是 9 维局部距离值。
- velocity_yaw 控制器现在把速度指令硬限制在 3 m/s。
- 训练侧保留轻量探索：小的 `exploration_loss_coeff` 和较窄随机化会把策略从固定套路里往外推一点。

## 二阶段回报不涨时的处理

- 先看日志里的 `rewraw_success`、`rewraw_progress`、`rewraw_quadcol_obstacle`，确认是不是碰撞项把正向回报盖住了。
- 如果经常撞障碍物，优先继续提高 `quads_obstacle_scan_resolution`，或者再把 `quads_obst_density` 往下调一点。
- 如果成功信号太弱，可以适当提高 `quads_first_success_reward`，或者小幅抬高 `quads_progress_reward`。
- 如果策略总是冲太猛，可以保持 `quads_vel_penalty_limit` 不变，继续让超速惩罚起作用。
- 如果还是容易卡住同一条路径，优先保留随机化和探索项，再考虑小幅提高目标奖励。
- 如果只看 best checkpoint，会忽略训练过程里 reward 的真实变化，还是要盯 episode reward 曲线。
- 如果 reward 越训越低，优先怀疑探索过强、随机化太难、碰撞/速度惩罚逐步 anneal 上来，而不是马上认为策略完全退化。

## 目录约定

- 训练输出在 `train_dir/`
- 测试时 `--experiment` 需要指向具体实验目录
