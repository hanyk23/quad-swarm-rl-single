# Single Quadrotor Corridor and Lidar RL

本仓库用于训练和评估单无人机障碍物导航策略。项目基于 Sample Factory 和
`gym_art` 四旋翼动力学，当前保留两条独立实验链：使用局部距离采样的长走廊
课程学习，以及使用二维模拟激光雷达的机体系避障。`corridor` 表示场景形状，
不代表雷达传感器。

## 当前任务

仓库保留两组实验：

| 实验 | 用途 | 入口 |
| --- | --- | --- |
| `single_quad_corridor_curriculum_v7` | 40 x 10 m 长走廊课程学习，使用 `octomap`/SDF 局部距离采样 | `train_two_stages_corridor.sh` |
| `single_quad_obstacles_lidar_body_v2` | 机体系观测与机体系速度控制的雷达避障 | `train_single_quad_obstacles_lidar.sh` |

两组实验都使用单无人机、`o_random` 随机航点、柱状障碍物和 APPO，但障碍物观测
方式不同，checkpoint 不能混用。训练输出位于 `train_dir/`，该目录不会提交到 Git。

## 环境安装

推荐使用 Python 3.11：

```bash
conda create -n swarm-rl python=3.11
conda activate swarm-rl
pip install -e .
```

测试和可视化需要系统提供 OpenGL。无桌面服务器上可使用 `xvfb-run`。

## 两种观测

### Corridor

- corridor 是 40 x 10 m 的长条形场景，不是雷达方案。
- 障碍物观测类型为 `octomap`，实际提供无人机周围 3 x 3 采样点的 SDF 距离。
- `quads_obstacle_scan_resolution` 控制这 9 个采样点之间的间距。
- 控制接口使用普通 `velocity_yaw`。

### Lidar

- 雷达观测共 9 维：机身 yaw 坐标系中的 8 个等角度射线距离，以及当前位置到最近
  障碍物或墙面的 clearance。
- 射线会与柱体和房间边界求最近交点，不依赖旧版 SDF 采样分辨率。
- `velocity_yaw_body_avoid` 使用机身 yaw 坐标系水平速度观测和动作，再转换到世界坐标系跟踪。
- avoid 控制器会叠加 PID 风格的局部避障速度修正和低高度保护。
- `quads_obst_min_clearance` 控制生成柱体表面之间的最小净空。

## 训练

### 长走廊课程学习

```bash
bash train_two_stages_corridor.sh
```

阶段一使用障碍物密度为零的 `octomap` 观测进行预热，阶段二在长走廊柱状障碍环境中
继续使用 `octomap`，并从阶段一 checkpoint 续训。两个阶段共用
`single_quad_corridor_curriculum_v7` 实验目录。

### 机体系雷达避障

首次使用当前机体系观测语义训练：

```bash
bash train_single_quad_obstacles_lidar.sh retrain
```

继续现有阶段二 checkpoint：

```bash
bash train_single_quad_obstacles_lidar.sh resume
```

其他可用模式：

```text
stage1          续训阶段一
stage1-retrain  覆盖并重训阶段一
stage2          续训阶段二
two-stage       依次续训阶段一和阶段二
```

`body_v2` 以前的 checkpoint 使用不同的观测坐标系，不应与当前模型混用。

## 评估

评估长走廊模型：

```bash
bash test_single_quad_corridor.sh latest
bash test_single_quad_corridor.sh best
```

评估机体系雷达模型：

```bash
bash test_single_quad_obstacles_lidar.sh latest
bash test_single_quad_obstacles_lidar.sh best
```

评估脚本默认开启 chase 视角和二维投影地图。corridor 评估沿用 checkpoint 中的
`octomap` 配置；机体系雷达脚本还会显示附近障碍物与墙面点云。

## 测试

```bash
bash run_tests.sh
```

测试覆盖雷达射线与墙面距离、障碍物最小间距、机体系观测旋转，以及两条任务共同
使用的随机航点刷新逻辑。

## 目录结构

```text
gym_art/quadrotor_multi/              四旋翼动力学、碰撞、雷达、渲染与 o_random 场景
swarm_rl/env_wrappers/                Sample Factory 环境封装与参数
swarm_rl/models/                      策略编码器
swarm_rl/runs/single_quad/            当前实验配置
train_two_stages_corridor.sh          长走廊两阶段训练
train_single_quad_obstacles_lidar.sh  机体系雷达训练
test_single_quad_corridor.sh          长走廊评估
test_single_quad_obstacles_lidar.sh   机体系雷达评估
train_dir/                            本地 checkpoint 和日志，Git 忽略
```

## 关键训练配置

- corridor v7 阶段二：`quads_obstacle_obs_type=octomap`，`quads_obst_density=0.06`，
  障碍物尺寸随机范围 `0.40-0.45 m`。
- body v2 阶段一：少量小柱体预热 3M steps。
- body v2 阶段二：`quads_obstacle_obs_type=lidar`，`quads_obst_density=0.16`，
  继续训练到 25M steps。
- `o_random` 到达航点后立即生成新目标；首个目标和后续目标可使用不同成功奖励。
- best checkpoint 按 episode reward 保存。

训练异常时优先查看 `rewraw_success`、`rewraw_progress` 和
`rewraw_quadcol_obstacle`，判断成功信号是否被碰撞或速度惩罚覆盖。

## 来源

核心仿真代码源自 [gym_art](https://github.com/amolchanov86/gym-art)，训练框架使用
[Sample Factory](https://github.com/alex-petrenko/sample-factory)。本仓库已收敛为
单无人机 corridor 与 lidar 两条研究实验链，不再包含原项目的多机实验、论文绘图和
Sim2Real 导出工具。
