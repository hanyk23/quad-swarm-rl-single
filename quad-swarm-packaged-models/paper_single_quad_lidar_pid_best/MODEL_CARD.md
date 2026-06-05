# paper_single_quad_lidar_pid_best

- Task: single-quad lidar obstacle navigation.
- Checkpoint: `checkpoint_p0/best_000007350_7526400_reward_50.919.pth`.
- Checkpoint reward in filename: `50.919`.
- Observation: `xyz_vxyz_R_omega_wall` + 9-ray lidar.
- Policy action interface: `velocity_yaw_avoid`.
- Evaluation episode duration: `50.0s` by default, equivalent to 5000 control frames at 100 Hz.

The model is evaluated with controller-level PID obstacle avoidance:

- `quads_obstacle_avoidance_distance=1.35`
- `quads_obstacle_avoidance_max_speed=0.95`
- `quads_obstacle_avoidance_pid_kp=1.25`
- `quads_obstacle_avoidance_pid_kd=0.10`

Use `paper_enjoy_lidar_pid.sh` for evaluation and `paper_train_lidar_pid.sh` for continued training.
