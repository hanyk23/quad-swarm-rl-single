# Snapshot Lidar Obstacle Compatibility

Snapshot source:
`https://github.com/hanyk23/quad-swarm-rl-single/tree/snapshot/lidar-obstacles-20260529`

Local snapshot directory:
`/home/lzh/drone/quad-swarm-rl-single-snapshot-lidar-20260529`

This directory was populated from:

`/home/lzh/drone/quad-swarm-rl-single.zip`

The zip contains pretrained checkpoints. The main lidar obstacle model is:

`/home/lzh/drone/quad-swarm-rl-single-snapshot-lidar-20260529/train_dir/single_quad_obstacles_lidar_wall/single_obstacles_lidar_wall_/00_single_obstacles_lidar_wall_see_0/checkpoint_p0/best_000007350_7526400_reward_50.919.pth`

Expected experiment path:

`single_quad_obstacles_lidar_wall/single_obstacles_lidar_wall_/00_single_obstacles_lidar_wall_see_0/`

Continue training from the snapshot configuration:

```bash
cd /home/lzh/drone/quad-swarm-rl-single
bash train_snapshot_lidar_obstacles.sh
```

Watch the latest snapshot checkpoint:

```bash
cd /home/lzh/drone/quad-swarm-rl-single
bash enjoy_snapshot_lidar_obstacles.sh
```

Watch the best snapshot checkpoint:

```bash
cd /home/lzh/drone/quad-swarm-rl-single
CHECKPOINT_KIND=best bash enjoy_snapshot_lidar_obstacles.sh
```

Snapshot training output is saved under:

`/home/lzh/drone/quad-swarm-rl-single-snapshot-lidar-20260529/train_dir/`
