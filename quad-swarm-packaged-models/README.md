# Packaged Models

This directory is the self-contained model bundle extracted from `/home/lzh/drone/quad-swarm-rl-single.zip`.
The runtime environment/control code in `gym_art/` and `swarm_rl/` has also been synchronized from the same archive, so the model runs against the same simulator/control stack used by the packaged checkpoints.

## Model

### `paper_single_quad_lidar_pid_best`

- Task: single quadrotor obstacle navigation with lidar observations.
- Source experiment: `single_quad_obstacles_lidar_wall/single_obstacles_lidar_wall_/00_single_obstacles_lidar_wall_see_0`.
- Latest checkpoint: `checkpoint_p0/checkpoint_000008520_8724480.pth`.
- Best checkpoint: `checkpoint_p0/best_000007350_7526400_reward_50.919.pth`.
- Runtime parameters are loaded from the extracted `config.json`.
- Runtime code source: zip `gym_art/` and `swarm_rl/`.

## Evaluation

Default evaluation uses the packaged latest checkpoint with PID obstacle-avoidance assist, chase + topdown views, and the projection obstacle map enabled.

```bash
cd /home/lzh/drone/quad-swarm-rl-single
bash paper_enjoy_lidar_pid.sh
```

To evaluate the original learned controller without PID assist:

```bash
cd /home/lzh/drone/quad-swarm-rl-single
CONTROL_TYPE=velocity_yaw bash paper_enjoy_lidar_pid.sh
```

For headless loading tests:

```bash
cd /home/lzh/drone/quad-swarm-rl-single
DEVICE=cpu RENDER=False NO_RENDER=True MAX_NUM_EPISODES=1 bash paper_enjoy_lidar_pid.sh
```

## Continued Training

This continues from the packaged model in a separate finetune experiment and keeps the package unchanged.

```bash
cd /home/lzh/drone/quad-swarm-rl-single
bash paper_train_lidar_pid.sh
```

Training copies this packaged model into `train_dir_paper_lidar_pid/paper_single_quad_lidar_pid_finetune` first, so the packaged checkpoints remain unchanged.
