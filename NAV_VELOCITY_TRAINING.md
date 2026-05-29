# 单机避障训练与 YOLO 感知说明

## 当前方案

- 控制接口：`xyz` 三轴速度
- 障碍观测：`YOLO` 风格框特征，不再用原来的 `octomap`
- 训练时默认感知源：`oracle`
  - 含义：训练时直接从仿真几何生成与 YOLO 一致的框
  - 好处：训练稳定、速度快、观测格式和真机上的检测框一致
- 真机/实测时感知源：`detector`
  - 含义：真的跑 Ultralytics YOLO 模型做检测

这套路径适合迁移：

`仿真柱子几何 -> YOLO风格框 -> RL避障策略 -> 真机YOLO输出同格式框`

## 直接训练

当前训练脚本已经切到 `YOLO` 风格障碍观测，默认实验名：

```bash
single_quad_velocity_nav_yolo_v1
```

开新训练：

```bash
cd /home/lzh/drone/quad-swarm-rl-single
bash train_local_velocity_curriculum.sh
```

中断后续训：

```bash
cd /home/lzh/drone/quad-swarm-rl-single
bash train_local_velocity_curriculum.sh resume
```

模型目录：

```bash
/home/lzh/drone/quad-swarm-rl-single/train_dir_velocity_nav/single_quad_velocity_nav_yolo_v1
```

## 先看识别效果

先不用重新训 RL，也不用先有真 YOLO 权重。可以先看仿真相机里“YOLO 风格障碍框”的效果：

```bash
cd /home/lzh/drone/quad-swarm-rl-single
/home/lzh/miniconda3/envs/swarm-rl/bin/python -m swarm_rl.vision.demo_yolo_fpv \
  --episodes=2 \
  --max_steps=300 \
  --quads_yolo_source=oracle \
  --output_video=/home/lzh/drone/quad-swarm-rl-single/vision_outputs/yolo_fpv_demo.mp4
```

如果你想边跑边弹窗看：

```bash
cd /home/lzh/drone/quad-swarm-rl-single
/home/lzh/miniconda3/envs/swarm-rl/bin/python -m swarm_rl.vision.demo_yolo_fpv \
  --episodes=2 \
  --max_steps=300 \
  --quads_yolo_source=oracle \
  --display=True
```

## 看当前训练策略 + YOLO 框

如果你想看“正在训练出来的策略”而不是启发式飞行，用这个单窗口 viewer。

这个脚本比 `enjoy + 4个视角窗口` 轻很多，只显示机载第一视角并叠加检测框，帧率会明显更稳。

看当前最好的模型：

```bash
cd /home/lzh/drone/quad-swarm-rl-single
/home/lzh/miniconda3/envs/swarm-rl/bin/python -m swarm_rl.vision.enjoy_yolo_policy \
  --experiment=single_quad_velocity_nav_yolo_v1 \
  --train_dir=/home/lzh/drone/quad-swarm-rl-single/train_dir_velocity_nav \
  --load_checkpoint_kind=best \
  --device=gpu \
  --quads_yolo_source=oracle \
  --display=True \
  --max_num_episodes=10
```

看最新保存的模型：

```bash
cd /home/lzh/drone/quad-swarm-rl-single
/home/lzh/miniconda3/envs/swarm-rl/bin/python -m swarm_rl.vision.enjoy_yolo_policy \
  --experiment=single_quad_velocity_nav_yolo_v1 \
  --train_dir=/home/lzh/drone/quad-swarm-rl-single/train_dir_velocity_nav \
  --load_checkpoint_kind=latest \
  --device=gpu \
  --quads_yolo_source=oracle \
  --display=True \
  --max_num_episodes=10
```

如果你想顺手录视频：

```bash
cd /home/lzh/drone/quad-swarm-rl-single
/home/lzh/miniconda3/envs/swarm-rl/bin/python -m swarm_rl.vision.enjoy_yolo_policy \
  --experiment=single_quad_velocity_nav_yolo_v1 \
  --train_dir=/home/lzh/drone/quad-swarm-rl-single/train_dir_velocity_nav \
  --load_checkpoint_kind=best \
  --device=gpu \
  --quads_yolo_source=oracle \
  --output_video=/home/lzh/drone/quad-swarm-rl-single/vision_outputs/policy_yolo_view.mp4
```

## 导出 YOLO 数据集

从仿真第一视角导出图片和标签：

```bash
cd /home/lzh/drone/quad-swarm-rl-single
/home/lzh/miniconda3/envs/swarm-rl/bin/python -m swarm_rl.vision.export_yolo_dataset \
  --frames=4000 \
  --output_dir=/home/lzh/drone/quad-swarm-rl-single/data/yolo_obstacles_pillars
```

导出后会生成：

```bash
data/yolo_obstacles_pillars/data.yaml
data/yolo_obstacles_pillars/images/train
data/yolo_obstacles_pillars/images/val
data/yolo_obstacles_pillars/labels/train
data/yolo_obstacles_pillars/labels/val
```

## 训练真实 YOLO 检测器

如果环境里还没装 `ultralytics`：

```bash
cd /home/lzh/drone/quad-swarm-rl-single
/home/lzh/miniconda3/envs/swarm-rl/bin/python -m pip install ultralytics
```

训练柱子检测器：

```bash
cd /home/lzh/drone/quad-swarm-rl-single
/home/lzh/miniconda3/envs/swarm-rl/bin/python -m swarm_rl.vision.train_yolo_obstacles \
  --data=/home/lzh/drone/quad-swarm-rl-single/data/yolo_obstacles_pillars/data.yaml \
  --model=yolov8n.pt \
  --epochs=60 \
  --imgsz=640 \
  --project=/home/lzh/drone/quad-swarm-rl-single/train_dir_yolo \
  --name=pillar_detector_v1
```

训练完成后，权重大概率会在：

```bash
/home/lzh/drone/quad-swarm-rl-single/train_dir_yolo/pillar_detector_v1/weights/best.pt
```

## 看真实 YOLO 检测效果

有了自己的检测器权重后，可以把演示切到真实 detector：

```bash
cd /home/lzh/drone/quad-swarm-rl-single
/home/lzh/miniconda3/envs/swarm-rl/bin/python -m swarm_rl.vision.demo_yolo_fpv \
  --episodes=2 \
  --max_steps=300 \
  --quads_yolo_source=detector \
  --quads_yolo_model_path=/home/lzh/drone/quad-swarm-rl-single/train_dir_yolo/pillar_detector_v1/weights/best.pt \
  --output_video=/home/lzh/drone/quad-swarm-rl-single/vision_outputs/yolo_detector_demo.mp4
```

## 重新训练 RL 是否必要

需要。

原因不是“模型不够好”，而是输入已经换了：

- 旧模型吃的是 `octomap`
- 新模型吃的是 `YOLO` 框特征

观测空间已经变了，所以旧的避障策略不能直接续训到新输入上。建议直接从新的：

```bash
single_quad_velocity_nav_yolo_v1
```

重新开始训练。

## 真机迁移建议

- 仿真训练阶段：`--quads_obstacle_obs_type=yolo --quads_yolo_source=oracle`
- 检测器验证阶段：`--quads_yolo_source=detector`
- 真机部署阶段：把真实相机图像送进 YOLO，取出最多 `3` 个障碍框，编码成同样的 18 维特征：

```text
[cx, cy, w, h, conf, area] x 3
```

这样仿真训练和真机部署的 RL 输入结构就是一致的。
