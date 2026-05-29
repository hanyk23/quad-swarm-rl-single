# Packaging Notes

Current recommended runtime:

```bash
cd /home/lzh/drone/quad-swarm-rl-single
CHECKPOINT_KIND=best bash enjoy_snapshot_lidar_pid.sh
```

Clean packaged models are stored outside the Git working tree:

`/home/lzh/drone/quad-swarm-packaged-models`

Model bundle names:

- `single_quad_lidar_original_best_20260529`
- `single_quad_lidar_pid_best_20260529`

The model bundle contains only:

- `config.json`
- `checkpoint_p0/best_000007350_7526400_reward_50.919.pth`
- a short `MODEL_CARD.md`

Training output directories are intentionally ignored by Git:

- `train_dir/`
- `train_dir_pid/`
- `train_dir_velocity_nav/`

Code commits for this package:

- Main workflow repo: `96dbf47`
- Snapshot runtime repo: `dcd6ea2`
