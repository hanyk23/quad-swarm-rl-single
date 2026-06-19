# Quad Swarm RL Single

本仓库是一个面向单无人机障碍物导航的强化学习实验工程。代码基于 `gym_art` 四旋翼动力学和
Sample Factory/APPO 训练框架，当前主要维护两条实验线：

- `corridor`：长走廊课程学习，使用局部 SDF/octomap 距离采样。
- `lidar body CBF`：9 个机体系方向扇区雷达、机体系速度控制和 CBF-QP 安全过滤。

仓库中的训练脚本、评估脚本和测试都围绕这两条链路组织。`train_dir/` 是本地训练输出目录，
默认不提交；当前也提供了 `train_dir.zip` 用于保存已有训练结果快照。

## 环境安装

推荐使用 Python 3.11：

```bash
conda create -n swarm-rl python=3.11
conda activate swarm-rl
pip install -e .
```

可视化和部分测试依赖 OpenGL。没有桌面的服务器上可以通过 `xvfb-run` 运行评估或测试命令。

## 快速使用

运行全部测试：

```bash
bash run_tests.sh
```

训练长走廊课程学习任务：

```bash
bash train_two_stages_corridor.sh
```

从 v5 best checkpoint 创建独立 v6 雷达避障微调实验，并开始训练：

```bash
bash train_single_quad_obstacles_lidar.sh
```

继续已有 v6 雷达避障实验：

```bash
bash train_single_quad_obstacles_lidar.sh resume-v6
```

评估长走廊模型：

```bash
bash test_single_quad_corridor.sh latest
bash test_single_quad_corridor.sh best
```

评估雷达避障模型：

```bash
bash test_single_quad_obstacles_lidar.sh latest v6
bash test_single_quad_obstacles_lidar.sh best v6
```

## 训练入口

### Corridor

```bash
bash train_two_stages_corridor.sh
```

该脚本执行长走廊两阶段课程学习：

- 阶段一在低难度环境中预热策略。
- 阶段二切换到带柱状障碍物的长走廊环境。
- 障碍物观测类型为 `octomap`，实际是无人机附近 3 x 3 局部 SDF 距离采样。
- 控制类型使用普通 `velocity_yaw`。

实验目录为 `train_dir/single_quad_corridor_curriculum_v7/...`。

### Lidar Body CBF

主入口是：

```bash
bash train_single_quad_obstacles_lidar.sh [mode]
```

常用模式：

| 模式 | 作用 |
| --- | --- |
| `finetune-v6` | 默认模式，从 v5 best 创建 v6 并微调 |
| `resume-v6` | 继续最新 v6 checkpoint |
| `retrain` | 覆盖并完整重训 v5 阶段一和阶段二 |
| `resume` | 继续旧 v5 阶段二 |
| `stage1` | 只续训 v5 阶段一 |
| `stage1-retrain` | 覆盖并重训 v5 阶段一 |
| `stage2` | 只续训 v5 阶段二 |
| `stage2-retrain` | 不加载旧权重，直接从零训练 v5 阶段二 |
| `stage2-from-v4` | 复制 v4 checkpoint，并用 v5 参数直接训练阶段二 |
| `two-stage` | 依次续训 v5 阶段一和阶段二 |

v6 也可以直接通过独立脚本启动：

```bash
bash train_single_quad_obstacles_lidar_v6.sh bootstrap
bash train_single_quad_obstacles_lidar_v6.sh resume
```

`bootstrap` 会调用 `swarm_rl/bootstrap_lidar_v6.py`，从 v5 best checkpoint 创建新的 v6
实验目录；如果 v6 目录已经存在，脚本会改为从最新 checkpoint 继续。

## 项目结构

```text
gym_art/quadrotor_multi/              四旋翼动力学、碰撞、障碍物、雷达和渲染
gym_art/quadrotor_multi/obstacles/    柱状障碍物生成、扇区雷达子射线检测和单元测试
swarm_rl/env_wrappers/                Sample Factory 环境封装、参数和 reward shaping
swarm_rl/models/                      策略网络和注意力层
swarm_rl/runs/single_quad/            单无人机实验配置
swarm_rl/bootstrap_lidar_v6.py        从 v5 best 创建 v6 微调 checkpoint
train_two_stages_corridor.sh          长走廊两阶段训练入口
train_single_quad_obstacles_lidar.sh  雷达避障主训练入口
train_single_quad_obstacles_lidar_v6.sh  v6 微调训练入口
test_single_quad_corridor.sh          长走廊评估入口
test_single_quad_obstacles_lidar.sh   雷达避障评估入口
TRAINING_IMPROVEMENTS.md              本轮训练和控制改进记录
```

## 实现思路

### 1. 两类障碍物观测分离

`corridor` 和 `lidar` 使用不同的观测语义，因此 checkpoint 不应混用：

- `corridor` 使用 `quads_obstacle_obs_type=octomap`，给策略提供局部 SDF 距离采样。
- `lidar` 使用 `quads_obstacle_obs_type=lidar`，给策略提供机体系下的方向化雷达距离。

这种分离可以避免“同样维度、不同含义”的观测进入同一个策略，降低训练和迁移时的隐性错误。

### 2. 九方向扇区雷达

当前雷达观测是 9 维，也就是 9 个机身 yaw 坐标系方向扇区。9 个方向每隔 40 度布置，
每一维都表示对应扇区内最近的障碍物或墙面距离；它不是旧版“8 条中心射线 + 1 个全局
clearance”的结构。每个方向也不是单条中心射线，而是一个小扇区：

- 扇区角度默认 `30.0` 度。
- 每个扇区采样 `5` 条子射线。
- 返回障碍物或墙面的最近交点距离。

这样第 9 维也具有明确方向语义，不再是旧版无方向的全局 clearance。同时，多子射线扇区能降低
细柱体刚好落在两条中心射线之间时的漏检概率。

### 3. 机体系导航

雷达任务使用 `velocity_yaw_body_avoid` 控制类型。目标方向、水平速度和雷达距离都被统一到
无人机 yaw 机体系下，策略输出的水平速度再转换回世界坐标执行。

这样做的核心目的是减少策略需要学习的坐标变换：策略看到的“前、后、左、右”始终跟机身方向一致，
不会因为机头朝向改变而让同一个动作含义发生旋转。

### 4. CBF-QP 安全过滤

旧避障逻辑类似 PID 偏置，会直接向策略动作叠加修正量，容易出现策略和避障器互相拉扯。
当前实现改为二维 CBF-QP，把策略给出的水平速度投影到安全速度集合：

```text
minimize  ||v_safe - v_policy||^2
subject to n_i * v_safe <= alpha * (distance_i - safe_distance)
```

其中 `n_i` 是雷达方向对应的障碍物法向，`distance_i` 是雷达测距。该设计有三个好处：

- 没有危险时尽量保持 RL 原始动作。
- 接近障碍物时只移除朝向障碍物的不安全速度分量。
- 横向绕行动作只要安全就会被保留。

v5/v6 当前使用：

```text
quads_avoid_radius=1.0
quads_cbf_safe_distance=0.3
quads_cbf_alpha=2.0
```

相比 v4，这组参数降低了窄通道内过早限速和 QP 无解后悬停的概率。

### 5. 雷达平滑和 CBF 滞回

v6 加入了两项稳定性处理：

```text
quads_avoid_lidar_filter_alpha=0.5
quads_avoid_activation_hysteresis=0.04
```

雷达 EMA 用来平滑单帧测距跳变；CBF 滞回让约束在进入危险区域后，需要距离超过
`avoid_radius + hysteresis` 才解除。两者共同减少安全过滤器在边界附近频繁启停导致的左右抖动。

### 6. v6 微调 checkpoint 初始化

`swarm_rl/bootstrap_lidar_v6.py` 用于从 v5 best checkpoint 创建一个干净的 v6 起点：

- 保留模型网络权重。
- 清空 Adam 优化器状态。
- 将学习率重置为 `3e-5`。
- 将动作标准差重置为 `0.30`。
- 将 `train_step` 和 `env_steps` 重置为 0。
- 重置 best reward 记录。
- 写入独立 v6 实验目录，不修改原 v5 结果。

这样既能继承 v5 已经学到的导航能力，又能避免旧优化器动量和过大探索方差破坏微调初期稳定性。

### 7. 奖励和动作稳定性

v6 训练配置还包含以下稳定性约束：

- `continuous_tanh_scale=1.2`：限制连续动作均值范围。
- `BoundedActionWrapper`：在动作进入环境前按 `action_space` 硬裁剪。
- `kl_loss_coeff=0.05`：限制策略更新幅度。
- `anneal_collision_initial_ratio=0.125`：碰撞、平滑碰撞、障碍物碰撞和超速惩罚从最终权重的
  12.5% 开始。
- `anneal_collision_steps=5000000`：在 500 万环境步内线性恢复完整惩罚。

这些设置主要用于降低从旧策略微调时的退化风险。

## 测试覆盖

`bash run_tests.sh` 覆盖的重点包括：

- 九方向雷达和墙面距离计算。
- 多子射线扇区对细障碍物的检测。
- 目标和速度向机体系旋转。
- CBF-QP 对危险速度的限制。
- CBF 对安全横向速度的保留。
- 雷达 EMA 和 CBF 启停滞回。
- 动作空间硬裁剪。
- Gymnasium `reset(seed, options)` 兼容。
- v6 checkpoint 的步数、优化器和动作方差重置。
- 障碍物最小净空生成逻辑。

## 训练结果和调参记录

更完整的改进过程、问题分析、v4/v5/v6 参数变化和 v6 训练结果记录在
`TRAINING_IMPROVEMENTS.md`。

训练异常时建议优先查看这些指标：

- `rewraw_success`：是否真正到达目标。
- `rewraw_progress`：是否持续向目标推进。
- `rewraw_quadcol_obstacle`：是否被障碍物碰撞惩罚覆盖。
- 动作标准差和 KL：判断微调是否发生策略突变。

best checkpoint 只按 episode reward 保存。由于 v6 使用惩罚渐增，早期 best reward
可能受较低惩罚权重影响；最终选型建议同时比较成功率、碰撞率、到达时间和左右速度反向切换频率。

## 来源

核心仿真代码源自 [gym_art](https://github.com/amolchanov86/gym-art)，训练框架使用
[Sample Factory](https://github.com/alex-petrenko/sample-factory)。本仓库已整理为单无人机
`corridor` 与 `lidar body CBF` 两条实验链。
