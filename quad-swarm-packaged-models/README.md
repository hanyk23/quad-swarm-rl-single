# Packaged Models

This directory is the self-contained model bundle extracted from `/home/lzh/drone/quad-swarm-rl-single.zip`.

## Model

### `paper_single_quad_lidar_pid_best`

- Task: single quadrotor obstacle navigation with lidar observations.
- Source experiment: `single_quad_obstacles_lidar_wall/single_obstacles_lidar_wall_/00_single_obstacles_lidar_wall_see_0`.
- Latest checkpoint: `checkpoint_p0/checkpoint_000008520_8724480.pth`.
- Best checkpoint: `checkpoint_p0/best_000007350_7526400_reward_50.919.pth`.
- Runtime parameters are loaded from the extracted `config.json`.

## Evaluation

```bash
cd /home/lzh/drone/quad-swarm-rl-single
bash paper_enjoy_lidar_pid.sh
```

For headless loading tests:

```bash
cd /home/lzh/drone/quad-swarm-rl-single
DEVICE=cpu RENDER=False NO_RENDER=True MAX_NUM_EPISODES=1 bash paper_enjoy_lidar_pid.sh
```

## Continued Training

```bash
cd /home/lzh/drone/quad-swarm-rl-single
bash paper_train_lidar_pid.sh
```

Training copies this packaged model into `train_dir_paper_lidar_pid/paper_single_quad_lidar_pid_finetune` first, so the packaged checkpoints remain unchanged.
