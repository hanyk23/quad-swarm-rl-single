# 深度相机避障训练

## 看当前模型原始效果

这个命令仍然给策略喂原来的 `octomap` 9 维观测，所以能看
`single_quad_velocity_nav_randomball_v3` 的真实当前效果；窗口左侧是原来的机载 FPV，右侧是俯视轨迹图。

```bash
cd /home/lzh/drone/quad-swarm-rl-single
/home/lzh/miniconda3/envs/swarm-rl/bin/python -m swarm_rl.vision.enjoy_depth_policy \
  --env=quadrotor_multi \
  --algo=APPO \
  --experiment=single_quad_velocity_nav_randomball_v3 \
  --train_dir=/home/lzh/drone/quad-swarm-rl-single/train_dir_velocity_nav \
  --load_checkpoint_kind=best \
  --device=gpu \
  --quads_obstacle_obs_type=octomap \
  --quads_camera_width=480 \
  --quads_camera_height=360 \
  --display=True \
  --viewer_fps=30 \
  --max_num_episodes=10
```

## 从当前模型 warm-start 训练深度相机策略

默认目标实验：

```text
single_quad_velocity_nav_depth_v6_smooth_dense
```

从当前 best checkpoint 初始化并训练。默认只继承模型权重，清空旧 optimizer 历史，并把学习率降到 `5e-6`。
v6 相比 v5 增加了障碍密度和目标球数量，同时进一步压低速度指令的抖动：

- 内部 velocity 控制器仍只接收 `xyz` 速度动作，但会自动让机头/相机朝向速度方向，低速时朝向当前目标。
- 最大水平速度、垂直速度、倾角和加速度更保守，并加入速度命令低通滤波，让飞行更稳。
- 障碍密度提高到 `0.18`，随机范围 `0.12~0.24`，柱子会明显更多。
- 目标球数量保持 `10`，并放宽目标球间距过滤，避免实际只生成三四个。
- 撞柱子惩罚和近障碍惩罚都会在 `1000000` env steps 内从 0 慢慢加到目标值。
- 进度奖励和目标球奖励更强，让策略先学会朝目标走，再在走的过程中避障。

```bash
cd /home/lzh/drone/quad-swarm-rl-single
bash train_local_velocity_depth_curriculum.sh
```

中断后续训：

```bash
cd /home/lzh/drone/quad-swarm-rl-single
bash train_local_velocity_depth_curriculum.sh resume
```

完全不加载旧模型、从零训练：

```bash
cd /home/lzh/drone/quad-swarm-rl-single
bash train_local_velocity_depth_curriculum.sh scratch
```

后续如果 v5 已经稳定，可以逐步打开 sim2real 噪声：

```bash
cd /home/lzh/drone/quad-swarm-rl-single
DEPTH_NOISE_STD=0.02 DEPTH_DROPOUT_PROB=0.01 bash train_local_velocity_depth_curriculum.sh resume
```

## 看深度策略效果

窗口左侧保持机载 FPV 视角，右侧显示房间、柱子、目标点、无人机当前位置和历史飞行轨迹。

看 best：

```bash
cd /home/lzh/drone/quad-swarm-rl-single
/home/lzh/miniconda3/envs/swarm-rl/bin/python -m swarm_rl.vision.enjoy_depth_policy \
  --env=quadrotor_multi \
  --algo=APPO \
  --experiment=single_quad_velocity_nav_depth_v6_smooth_dense \
  --train_dir=/home/lzh/drone/quad-swarm-rl-single/train_dir_velocity_nav \
  --load_checkpoint_kind=best \
  --device=gpu \
  --quads_obstacle_obs_type=depth \
  --quads_camera_width=480 \
  --quads_camera_height=360 \
  --display=True \
  --viewer_fps=30 \
  --max_num_episodes=10
```

看 latest：

```bash
cd /home/lzh/drone/quad-swarm-rl-single
/home/lzh/miniconda3/envs/swarm-rl/bin/python -m swarm_rl.vision.enjoy_depth_policy \
  --env=quadrotor_multi \
  --algo=APPO \
  --experiment=single_quad_velocity_nav_depth_v6_smooth_dense \
  --train_dir=/home/lzh/drone/quad-swarm-rl-single/train_dir_velocity_nav \
  --load_checkpoint_kind=latest \
  --device=gpu \
  --quads_obstacle_obs_type=depth \
  --quads_camera_width=480 \
  --quads_camera_height=360 \
  --display=True \
  --viewer_fps=30 \
  --max_num_episodes=10
```

## 接口约定

- 策略动作保持为 `xyz` 三轴速度：`--quads_control_mode=velocity`
- 训练里的相机朝向由控制器自动根据速度/目标转向：`--quads_velocity_yaw_mode=velocity_or_goal`
- 深度观测是 `3x3=9` 条机载相机射线，单位米，默认最大深度 `10m`
- `enjoy_depth_policy` 的显示窗口左侧是 FPV，右侧是轨迹视角；显示视角不改变策略输入
- 保持 9 维是为了和当前 octomap 模型结构一致，可以从旧 checkpoint 继续训练
