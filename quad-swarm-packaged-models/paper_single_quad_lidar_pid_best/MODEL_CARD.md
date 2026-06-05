# paper_single_quad_lidar_pid_best

- Task: single-quad lidar obstacle navigation.
- Source archive: `/home/lzh/drone/quad-swarm-rl-single.zip`.
- Source experiment: `train_dir/single_quad_obstacles_lidar_wall/single_obstacles_lidar_wall_/00_single_obstacles_lidar_wall_see_0`.
- Latest checkpoint: `checkpoint_p0/checkpoint_000008520_8724480.pth`.
- Best checkpoint: `checkpoint_p0/best_000007350_7526400_reward_50.919.pth`.
- Observation: `xyz_vxyz_R_omega_wall` + 9-ray lidar.
- Policy action interface: zip `quads_control_type=velocity_yaw`.
- Evaluation episode duration: `50.0s` by default, equivalent to 5000 control frames at 100 Hz.

The model is evaluated with the extracted zip `config.json`; launcher scripts only override path, rendering, device, and checkpoint selection.

Use `paper_enjoy_lidar_pid.sh` for evaluation and `paper_train_lidar_pid.sh` for continued training.
