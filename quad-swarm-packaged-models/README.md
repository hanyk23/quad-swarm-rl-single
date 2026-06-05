# Packaged Models

This directory is the self-contained model bundle for the paper-style single-quad lidar obstacle-navigation experiment.

## Model

### `paper_single_quad_lidar_pid_best`

- Task: single quadrotor obstacle navigation with lidar observations.
- Policy checkpoint: `checkpoint_p0/best_000007350_7526400_reward_50.919.pth`.
- Runtime controller: legacy velocity-yaw controller from the snapshot training code.
- The original external snapshot checkout is not required.

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

Training copies this packaged model into `train_dir_paper_lidar_pid/paper_single_quad_lidar_pid_finetune` first, so the packaged best checkpoint remains unchanged.
