# paper_single_quad_lidar_pid_best

- Task: single-quad lidar obstacle navigation.
- Checkpoint: `checkpoint_p0/best_000007350_7526400_reward_50.919.pth`.
- Checkpoint reward in filename: `50.919`.
- Observation: `xyz_vxyz_R_omega_wall` + 9-ray lidar.
- Policy action interface: legacy `velocity_yaw`.
- Evaluation episode duration: `50.0s` by default, equivalent to 5000 control frames at 100 Hz.

The model is evaluated with the same legacy velocity-yaw controller used during snapshot training:

- `quads_control_mode=legacy_velocity_yaw`
- `quads_control_type=velocity_yaw`
- `quads_velocity_yaw_max_speed=3.0`

Use `paper_enjoy_lidar_pid.sh` for evaluation and `paper_train_lidar_pid.sh` for continued training.
