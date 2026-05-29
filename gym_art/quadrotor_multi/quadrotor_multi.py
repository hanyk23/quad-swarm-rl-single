import copy
import time
from collections import deque
from copy import deepcopy

import gymnasium as gym
import numpy as np

from gym_art.quadrotor_multi.aerodynamics.downwash import perform_downwash
from gym_art.quadrotor_multi.collisions.obstacles import perform_collision_with_obstacle
from gym_art.quadrotor_multi.collisions.quadrotors import calculate_collision_matrix, \
    calculate_drone_proximity_penalties, perform_collision_between_drones
from gym_art.quadrotor_multi.collisions.room import perform_collision_with_wall, perform_collision_with_ceiling
from gym_art.quadrotor_multi.obstacles.utils import get_cell_centers
from gym_art.quadrotor_multi.quad_utils import QUADS_OBS_REPR, QUADS_NEIGHBOR_OBS_TYPE

from gym_art.quadrotor_multi.obstacles.obstacles import MultiObstacles
from gym_art.quadrotor_multi.quadrotor_multi_visualization import Quadrotor3DSceneMulti
from gym_art.quadrotor_multi.quadrotor_single import QuadrotorSingle
from gym_art.quadrotor_multi.scenarios.mix import create_scenario
from swarm_rl.vision.yolo_obstacles import (
    CameraInfo,
    UltralyticsObstacleDetector,
    annotate_detections,
    encode_obstacle_detections,
    mask_obstacle_detections,
    oracle_obstacle_detections,
)
from swarm_rl.vision.depth_camera import (
    DEPTH_OBS_DIM,
    colorize_depth,
    encode_depth_observation,
    encode_lidar_observation,
    raycast_depth_map,
)

EPS = 1e-6


class QuadrotorEnvMulti(gym.Env):
    def __init__(self, num_agents, ep_time, rew_coeff, obs_repr,
                 # Neighbor
                 neighbor_visible_num, neighbor_obs_type, collision_hitbox_radius, collision_falloff_radius,

                 # Obstacle
                 use_obstacles, obst_density, obst_size, obst_spawn_area, obstacle_obs_type,
                 obstacle_safe_distance,

                 # Aerodynamics, Numba Speed Up, Scenarios, Room, Replay Buffer, Rendering
                 use_downwash, use_numba, quads_mode, room_dims, use_replay_buffer, quads_view_mode,
                 quads_render,

                 # Quadrotor Specific (Do Not Change)
                 dynamics_params, raw_control, raw_control_zero_middle,
                 dynamics_randomize_every, dynamics_change, dyn_sampler_1,
                 sense_noise, init_random_state,
                 sim_freq=200.0, sim_steps=2,
                 wall_safe_distance=0.8, wall_guard_distance=0.0, obstacle_guard_distance=0.0,
                 obstacle_guard_terminate=True, wall_guard_terminate=True,
                 terminate_on_obstacle_collision=False, terminate_on_wall_collision=False,
                 control_mode='raw', velocity_max_xy=2.0, velocity_max_z=1.0,
                 velocity_max_tilt_deg=35.0, velocity_max_acc_xy=6.0,
                 velocity_max_acc_z_up=4.0, velocity_max_acc_z_down=4.0,
                velocity_yaw_mode='keep', velocity_yaw_min_speed=0.15,
                velocity_yaw_rate_max=0.0, velocity_yaw_control_scale=1.0,
                velocity_command_smoothing_tau=0.0,
                 controller_obstacle_avoidance=False, obstacle_avoidance_distance=1.2,
                 obstacle_avoidance_max_speed=0.8, obstacle_avoidance_gain=1.2,
                 obstacle_avoidance_pid_kp=1.0, obstacle_avoidance_pid_ki=0.0,
                 obstacle_avoidance_pid_kd=0.0, obstacle_avoidance_pid_integral_limit=1.0,
                 goal_ball_capture_assist=False, goal_ball_capture_assist_distance=1.2,
                 goal_ball_capture_assist_speed=0.8, goal_ball_tangent_damping=0.25,
                 velocity_attitude_max_angle_deg=45.0, velocity_attitude_blend=1.0,
                 goal_z_range=(1.0, 3.0),
                 use_goal_ball=False, goal_ball_reward=0.0, goal_ball_radius=0.4, goal_ball_count=1,
                 goal_ball_velocity_reset=False, goal_ball_velocity_reset_ratio=0.0,
                 camera_hw=(320, 240), camera_fov=90.0, camera_pitch_deg=25.0, camera_drone_index=0,
                 depth_grid_hw=(3, 3), depth_min_distance=0.05, depth_max_distance=10.0,
                 lidar_num_rays=9,
                 depth_noise_std=0.03, depth_dropout_prob=0.02, depth_normalize=False,
                 yolo_source='oracle', yolo_model_path='', yolo_conf_threshold=0.25,
                 # Rendering
                 render_mode='human'
                 ):
        super().__init__()

        # Predefined Parameters
        self.num_agents = num_agents
        obs_self_size = QUADS_OBS_REPR[obs_repr]
        if neighbor_obs_type == 'none':
            self.num_use_neighbor_obs = 0
        elif neighbor_visible_num == -1:
            self.num_use_neighbor_obs = self.num_agents - 1
        else:
            self.num_use_neighbor_obs = neighbor_visible_num

        # Set to True means that sample_factory will treat it as a multi-agent vectorized environment even with
        # num_agents=1. More info, please look at sample-factory: envs/quadrotors/wrappers/reward_shaping.py
        self.is_multiagent = True
        self.room_dims = room_dims
        self.quads_view_mode = quads_view_mode
        self.camera_hw = tuple(int(v) for v in camera_hw)
        self.camera_fov = float(camera_fov)
        self.camera_pitch_deg = float(camera_pitch_deg)
        self.camera_drone_index = int(camera_drone_index)
        self.depth_grid_hw = tuple(int(v) for v in depth_grid_hw)
        if self.depth_grid_hw[0] * self.depth_grid_hw[1] != DEPTH_OBS_DIM:
            raise ValueError(
                f"Depth observation must be {DEPTH_OBS_DIM}D for the current model. "
                f"Got grid {self.depth_grid_hw[0]}x{self.depth_grid_hw[1]}."
            )
        self.depth_min_distance = float(depth_min_distance)
        self.depth_max_distance = float(depth_max_distance)
        self.depth_noise_std = float(depth_noise_std)
        self.depth_dropout_prob = float(depth_dropout_prob)
        self.depth_normalize = bool(depth_normalize)
        self.obstacle_obs_type = obstacle_obs_type
        self.obstacle_safe_distance = float(obstacle_safe_distance)
        self.obstacle_guard_distance = max(float(obstacle_guard_distance), 0.0)
        self.wall_safe_distance = float(wall_safe_distance)
        self.wall_guard_distance = max(float(wall_guard_distance), 0.0)
        self.guard_collision_raw_penalty = 0.05
        self.lidar_num_rays = max(1, int(lidar_num_rays))
        self.obstacle_guard_terminate = bool(obstacle_guard_terminate)
        self.wall_guard_terminate = bool(wall_guard_terminate)
        self.terminate_on_obstacle_collision = bool(terminate_on_obstacle_collision)
        self.terminate_on_wall_collision = bool(terminate_on_wall_collision)
        self.control_mode = str(control_mode)
        self.controller_obstacle_avoidance = bool(controller_obstacle_avoidance)
        self.obstacle_avoidance_distance = max(0.0, float(obstacle_avoidance_distance))
        self.obstacle_avoidance_max_speed = max(0.0, float(obstacle_avoidance_max_speed))
        self.obstacle_avoidance_gain = max(0.0, float(obstacle_avoidance_gain))
        self.obstacle_avoidance_pid_kp = max(0.0, float(obstacle_avoidance_pid_kp))
        self.obstacle_avoidance_pid_ki = max(0.0, float(obstacle_avoidance_pid_ki))
        self.obstacle_avoidance_pid_kd = max(0.0, float(obstacle_avoidance_pid_kd))
        self.obstacle_avoidance_pid_integral_limit = max(0.0, float(obstacle_avoidance_pid_integral_limit))
        self.goal_ball_capture_assist = bool(goal_ball_capture_assist)
        self.goal_ball_capture_assist_distance = max(0.0, float(goal_ball_capture_assist_distance))
        self.goal_ball_capture_assist_speed = max(0.0, float(goal_ball_capture_assist_speed))
        self.goal_ball_tangent_damping = float(np.clip(goal_ball_tangent_damping, 0.0, 1.0))
        self.yolo_source = yolo_source
        self.yolo_model_path = yolo_model_path
        self.yolo_conf_threshold = float(yolo_conf_threshold)
        self.yolo_detector = None
        self.latest_obstacle_detections = [[] for _ in range(self.num_agents)]
        self.latest_obstacle_obs = np.zeros((self.num_agents, 0), dtype=np.float32)

        # Generate All Quadrotors
        self.envs = []
        for i in range(self.num_agents):
            e = QuadrotorSingle(
                # Quad Parameters
                dynamics_params=dynamics_params, dynamics_change=dynamics_change,
                dynamics_randomize_every=dynamics_randomize_every, dyn_sampler_1=dyn_sampler_1,
                raw_control=raw_control, raw_control_zero_middle=raw_control_zero_middle, sense_noise=sense_noise,
                init_random_state=init_random_state, obs_repr=obs_repr, ep_time=ep_time, room_dims=room_dims,
                sim_freq=sim_freq, sim_steps=sim_steps,
                use_numba=use_numba, control_mode=control_mode, velocity_max_xy=velocity_max_xy,
                velocity_max_z=velocity_max_z, velocity_max_tilt_deg=velocity_max_tilt_deg,
                velocity_max_acc_xy=velocity_max_acc_xy,
                velocity_max_acc_z_up=velocity_max_acc_z_up,
                velocity_max_acc_z_down=velocity_max_acc_z_down,
                velocity_yaw_mode=velocity_yaw_mode,
                velocity_yaw_min_speed=velocity_yaw_min_speed,
                velocity_yaw_rate_max=velocity_yaw_rate_max,
                velocity_yaw_control_scale=velocity_yaw_control_scale,
                velocity_command_smoothing_tau=velocity_command_smoothing_tau,
                velocity_attitude_max_angle_deg=velocity_attitude_max_angle_deg,
                velocity_attitude_blend=velocity_attitude_blend,
                goal_z_range=goal_z_range,
                # Neighbor
                num_agents=num_agents,
                neighbor_obs_type=neighbor_obs_type, num_use_neighbor_obs=self.num_use_neighbor_obs,
                # Obstacle
                use_obstacles=use_obstacles, obstacle_obs_type=obstacle_obs_type,
                lidar_num_rays=self.lidar_num_rays,
            )
            self.envs.append(e)

        # Set Obs & Act
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

        # Aux variables
        self.quad_arm = self.envs[0].dynamics.arm
        self.quad_prop_radius = float(self.envs[0].dynamics.model.params["propellers"]["r"])
        self.quad_collision_radius = self.quad_arm + self.quad_prop_radius
        self.control_freq = self.envs[0].control_freq
        self.control_dt = 1.0 / self.control_freq
        self.pos = np.zeros([self.num_agents, 3])
        self.vel = np.zeros([self.num_agents, 3])
        self.omega = np.zeros([self.num_agents, 3])
        self.rel_pos = np.zeros((self.num_agents, self.num_agents, 3))
        self.rel_vel = np.zeros((self.num_agents, self.num_agents, 3))

        # Reward
        self.rew_coeff = dict(
            pos=1., effort=0.05, action_change=0., progress=0., crash=1., orient=1., yaw=0., rot=0.,
            attitude=0., spin=0.1, vel=0., vz=0., height_error=0., thrust=0., stagnation=0.,
            overspeed=0.,
            obst_proximity=0., wallcol_bin=0., wall_proximity=0., safe_flight=0., path_alignment=0.,
            obstacle_clearance_delta=0., wall_clearance_delta=0.,
            quadcol_bin=5., quadcol_bin_smooth_max=4., quadcol_bin_obst=5.
        )
        rew_coeff_orig = copy.deepcopy(self.rew_coeff)

        if rew_coeff is not None:
            assert isinstance(rew_coeff, dict)
            assert set(rew_coeff.keys()).issubset(set(self.rew_coeff.keys()))
            self.rew_coeff.update(rew_coeff)
        for key in self.rew_coeff.keys():
            self.rew_coeff[key] = float(self.rew_coeff[key])

        orig_keys = list(rew_coeff_orig.keys())
        # Checking to make sure we didn't provide some false rew_coeffs (for example by misspelling one of the params)
        assert np.all([key in orig_keys for key in self.rew_coeff.keys()])

        # Neighbors
        neighbor_obs_size = QUADS_NEIGHBOR_OBS_TYPE[neighbor_obs_type]

        self.clip_neighbor_space_length = self.num_use_neighbor_obs * neighbor_obs_size
        self.clip_neighbor_space_min_box = self.observation_space.low[
                                           obs_self_size:obs_self_size + self.clip_neighbor_space_length]
        self.clip_neighbor_space_max_box = self.observation_space.high[
                                           obs_self_size:obs_self_size + self.clip_neighbor_space_length]

        # Obstacles
        self.use_obstacles = use_obstacles
        self.obstacles = None
        self.num_obstacles = 0
        self.cell_centers = None
        if self.use_obstacles:
            self.prev_obst_quad_collisions = []
            self.obst_quad_collisions_per_episode = 0
            self.obst_quad_collisions_after_settle = 0
            self.curr_quad_col = []
            self.obst_density = obst_density
            self.obst_spawn_area = obst_spawn_area
            self.num_obstacles = int(obst_density * obst_spawn_area[0] * obst_spawn_area[1])
            self.obst_map = None
            self.obst_size = obst_size

            # Log more info
            self.distance_to_goal_3_5 = 0
            self.distance_to_goal_5 = 0

        # Goal-ball curriculum
        self.use_goal_ball = use_goal_ball
        self.goal_ball_reward = float(goal_ball_reward)
        self.goal_ball_radius = float(goal_ball_radius)
        self.goal_ball_count = max(0, int(goal_ball_count))
        self.goal_ball_velocity_reset = bool(goal_ball_velocity_reset)
        self.goal_ball_velocity_reset_ratio = float(np.clip(goal_ball_velocity_reset_ratio, 0.0, 1.0))
        self.goal_ball_targets = [None for _ in range(self.num_agents)]
        self.goal_ball_sequences = [[] for _ in range(self.num_agents)]
        self.goal_ball_active = np.zeros(self.num_agents, dtype=bool)
        self.final_goals = [None for _ in range(self.num_agents)]
        self.goal_ball_collected_per_episode = 0
        self.goal_ball_targets_total = 0

        # Scenarios
        self.quads_mode = quads_mode
        self.scenario = create_scenario(quads_mode=quads_mode, envs=self.envs, num_agents=num_agents,
                                        room_dims=room_dims)

        # Collisions
        # # Collisions: Neighbors
        self.collisions_per_episode = 0
        # # # Ignore collisions because of spawn
        self.collisions_after_settle = 0
        self.collisions_grace_period_steps = 1.5 * self.control_freq
        self.collisions_grace_period_seconds = 1.5
        self.prev_drone_collisions = []

        self.collisions_final_grace_period_steps = 5.0 * self.control_freq
        self.collisions_final_5s = 0
        self.safe_flight_streak_steps = np.zeros(self.num_agents, dtype=np.float32)
        self.max_safe_flight_streak_steps = np.zeros(self.num_agents, dtype=np.float32)
        self.prev_obstacle_clearance = np.full(self.num_agents, np.inf, dtype=np.float32)
        self.prev_wall_clearance = np.full(self.num_agents, np.inf, dtype=np.float32)
        self.obstacle_avoidance_error_prev = np.zeros(self.num_agents, dtype=np.float32)
        self.obstacle_avoidance_error_integral = np.zeros(self.num_agents, dtype=np.float32)
        self.obstacle_avoidance_pid_initialized = np.zeros(self.num_agents, dtype=bool)

        # # # Dense reward info
        self.collision_threshold = collision_hitbox_radius * self.quad_arm
        self.collision_falloff_threshold = collision_falloff_radius * self.quad_arm

        # # Collisions: Room
        self.collisions_room_per_episode = 0
        self.collisions_floor_per_episode = 0
        self.collisions_wall_per_episode = 0
        self.collisions_ceiling_per_episode = 0

        self.prev_crashed_walls = []
        self.prev_crashed_ceiling = []
        self.prev_crashed_room = []

        # Replay
        self.use_replay_buffer = use_replay_buffer
        # # only start using the buffer after the drones learn how to fly
        self.activate_replay_buffer = False
        # # since the same collisions happen during replay, we don't want to keep resaving the same event
        self.saved_in_replay_buffer = False
        self.last_step_unique_collisions = False
        self.crashes_in_recent_episodes = deque([], maxlen=100)
        self.crashes_last_episode = 0

        # Numba
        self.use_numba = use_numba

        # Aerodynamics
        self.use_downwash = use_downwash

        # Rendering
        # # set to true whenever we need to reset the OpenGL scene in render()
        self.render_mode =render_mode
        self.quads_render = quads_render
        self.scenes = []
        self.reset_scene = False
        self.simulation_start_time = 0
        self.frames_since_last_render = self.render_skip_frames = 0
        self.render_every_nth_frame = 1
        # # Use this to control rendering speed
        self.render_speed = 1.0
        self.quads_formation_size = 2.0
        self.all_collisions = {val: [0.0 for _ in range(self.num_agents)] for val in ['drone', 'ground', 'obstacle']}
        self.latest_fpv_frame = None

        # Log
        self.distance_to_goal = [[] for _ in range(len(self.envs))]
        self.reached_goal = [False for _ in range(len(self.envs))]

        # Log metric
        self.agent_col_agent = np.ones(self.num_agents)
        self.agent_col_obst = np.ones(self.num_agents)

        # Others
        self.apply_collision_force = True

    def all_dynamics(self):
        return tuple(e.dynamics for e in self.envs)

    def active_scenario(self):
        if hasattr(self.scenario, "scenario") and self.scenario.scenario is not None:
            return self.scenario.scenario
        return self.scenario

    def _current_camera_info(self, drone_index):
        drone_index = int(np.clip(drone_index, 0, self.num_agents - 1))
        eye, center, up = self.envs[drone_index].dynamics.look_at(degrees_down=self.camera_pitch_deg)
        return CameraInfo(
            width=self.camera_hw[0],
            height=self.camera_hw[1],
            fov_deg=self.camera_fov,
            eye=np.array(eye, dtype=np.float32),
            center=np.array(center, dtype=np.float32),
            up=np.array(up, dtype=np.float32),
        )

    def _get_yolo_detector(self):
        if self.yolo_detector is None:
            if not self.yolo_model_path:
                raise ValueError(
                    "quads_yolo_model_path is empty. Provide a trained YOLO weight file when quads_yolo_source=detector."
                )
            self.yolo_detector = UltralyticsObstacleDetector(
                model_path=self.yolo_model_path,
                conf_threshold=self.yolo_conf_threshold,
            )
        return self.yolo_detector

    def get_current_obstacle_detections(self, drone_index=None, source=None, image_rgb=None):
        if drone_index is None:
            drone_index = self.camera_drone_index
        drone_index = int(np.clip(drone_index, 0, self.num_agents - 1))
        source = source or self.yolo_source

        if not self.use_obstacles or self.obstacles is None or len(self.obstacles.pos_arr) == 0:
            return []

        if source == 'oracle':
            return oracle_obstacle_detections(
                obstacle_positions=self.obstacles.pos_arr,
                obstacle_size=self.obst_size,
                room_height=self.room_dims[2],
                camera=self._current_camera_info(drone_index),
            )

        if source == 'oracle_mask':
            if image_rgb is None:
                image_rgb = self.get_drone_fpv_image(drone_index=drone_index)
            return mask_obstacle_detections(image_rgb)

        if source == 'detector':
            detector = self._get_yolo_detector()
            if image_rgb is None:
                image_rgb = self.get_drone_fpv_image(drone_index=drone_index)
            return detector.detect(image_rgb)

        raise NotImplementedError(f"Unsupported YOLO source: {source}")

    def get_encoded_obstacle_observation(self, drone_index=None, source=None):
        detections = self.get_current_obstacle_detections(drone_index=drone_index, source=source)
        return encode_obstacle_detections(
            detections=detections,
            image_shape=(self.camera_hw[1], self.camera_hw[0]),
        )

    def get_depth_observation(self, drone_index=None):
        if drone_index is None:
            drone_index = self.camera_drone_index
        drone_index = int(np.clip(drone_index, 0, self.num_agents - 1))

        obstacle_positions = []
        if self.use_obstacles and self.obstacles is not None and len(self.obstacles.pos_arr) > 0:
            obstacle_positions = self.obstacles.pos_arr

        return encode_depth_observation(
            obstacle_positions=obstacle_positions,
            obstacle_size=self.obst_size if self.use_obstacles else 0.0,
            room_height=self.room_dims[2],
            camera=self._current_camera_info(drone_index),
            grid_width=self.depth_grid_hw[0],
            grid_height=self.depth_grid_hw[1],
            max_distance=self.depth_max_distance,
            min_distance=self.depth_min_distance,
            normalize=self.depth_normalize,
            noise_std=self.depth_noise_std,
            dropout_prob=self.depth_dropout_prob,
        )

    def get_lidar_observation(self, drone_index=None):
        if drone_index is None:
            drone_index = self.camera_drone_index
        drone_index = int(np.clip(drone_index, 0, self.num_agents - 1))

        obstacle_positions = []
        if self.use_obstacles and self.obstacles is not None and len(self.obstacles.pos_arr) > 0:
            obstacle_positions = self.obstacles.pos_arr

        camera = self._current_camera_info(drone_index)
        yaw = float(np.arctan2(camera.center[1] - camera.eye[1], camera.center[0] - camera.eye[0]))
        return encode_lidar_observation(
            obstacle_positions=obstacle_positions,
            obstacle_size=self.obst_size if self.use_obstacles else 0.0,
            room_dims=self.room_dims,
            origin=self.envs[drone_index].dynamics.pos,
            yaw=yaw,
            num_rays=self.lidar_num_rays,
            max_distance=self.depth_max_distance,
            min_distance=self.depth_min_distance,
            normalize=self.depth_normalize,
            noise_std=self.depth_noise_std,
            dropout_prob=self.depth_dropout_prob,
        )

    def get_depth_camera_image(self, drone_index=None, width=160, height=120):
        if drone_index is None:
            drone_index = self.camera_drone_index
        drone_index = int(np.clip(drone_index, 0, self.num_agents - 1))

        obstacle_positions = []
        if self.use_obstacles and self.obstacles is not None and len(self.obstacles.pos_arr) > 0:
            obstacle_positions = self.obstacles.pos_arr

        return raycast_depth_map(
            obstacle_positions=obstacle_positions,
            obstacle_size=self.obst_size if self.use_obstacles else 0.0,
            room_height=self.room_dims[2],
            camera=self._current_camera_info(drone_index),
            width=width,
            height=height,
            max_distance=self.depth_max_distance,
            min_distance=self.depth_min_distance,
        )

    def get_depth_camera_color_image(self, drone_index=None, width=160, height=120):
        depth = self.get_depth_camera_image(drone_index=drone_index, width=width, height=height)
        return colorize_depth(
            depth,
            max_distance=self.depth_max_distance,
            min_distance=self.depth_min_distance,
        )

    def get_annotated_fpv_image(self, drone_index=None, source=None):
        if drone_index is None:
            drone_index = self.camera_drone_index
        image_rgb = self.get_drone_fpv_image(drone_index=drone_index)
        detections = self.get_current_obstacle_detections(
            drone_index=drone_index, source=source, image_rgb=image_rgb
        )
        return annotate_detections(image_rgb, detections)

    def append_obstacle_observations(self, obs):
        obs_array = np.asarray(obs, dtype=np.float32)

        if not self.use_obstacles or self.obstacle_obs_type == 'none':
            self.latest_obstacle_detections = [[] for _ in range(self.num_agents)]
            self.latest_obstacle_obs = np.zeros((self.num_agents, 0), dtype=np.float32)
            return obs_array

        if self.obstacle_obs_type == 'octomap':
            obs_array = self.obstacles.step(obs=obs_array, quads_pos=self.pos)
            self.latest_obstacle_detections = [[] for _ in range(self.num_agents)]
            self.latest_obstacle_obs = np.zeros((self.num_agents, 0), dtype=np.float32)
            return obs_array

        if self.obstacle_obs_type == 'depth':
            depth_features = []
            for agent_id in range(self.num_agents):
                depth_features.append(self.get_depth_observation(drone_index=agent_id))

            self.latest_obstacle_detections = [[] for _ in range(self.num_agents)]
            self.latest_obstacle_obs = np.stack(depth_features).astype(np.float32)
            return np.concatenate((obs_array, self.latest_obstacle_obs), axis=1)

        if self.obstacle_obs_type == 'lidar':
            lidar_features = []
            for agent_id in range(self.num_agents):
                lidar_features.append(self.get_lidar_observation(drone_index=agent_id))

            self.latest_obstacle_detections = [[] for _ in range(self.num_agents)]
            self.latest_obstacle_obs = np.stack(lidar_features).astype(np.float32)
            return np.concatenate((obs_array, self.latest_obstacle_obs), axis=1)

        if self.obstacle_obs_type == 'yolo':
            yolo_features = []
            detections_per_agent = []
            for agent_id in range(self.num_agents):
                detections = self.get_current_obstacle_detections(drone_index=agent_id)
                detections_per_agent.append(detections)
                yolo_features.append(
                    encode_obstacle_detections(
                        detections=detections,
                        image_shape=(self.camera_hw[1], self.camera_hw[0]),
                    )
                )

            self.latest_obstacle_detections = detections_per_agent
            self.latest_obstacle_obs = np.stack(yolo_features).astype(np.float32)
            return np.concatenate((obs_array, self.latest_obstacle_obs), axis=1)

        raise NotImplementedError(f"Unsupported obstacle_obs_type: {self.obstacle_obs_type}")

    def metric_goal(self, env_id):
        if self.final_goals[env_id] is not None:
            return self.final_goals[env_id]
        return self.envs[env_id].goal

    def active_target(self, env_id):
        if self.goal_ball_active[env_id] and self.goal_ball_targets[env_id] is not None:
            return self.goal_ball_targets[env_id]
        return self.metric_goal(env_id)

    def goal_ball_min_xy_separation(self):
        return max(1.0, 0.5 * float(self.room_dims[0]))

    def sync_active_goals(self):
        for agent_id, env in enumerate(self.envs):
            if self.goal_ball_active[agent_id] and self.goal_ball_targets[agent_id] is not None:
                env.goal = self.goal_ball_targets[agent_id]
            elif self.final_goals[agent_id] is not None:
                env.goal = self.final_goals[agent_id]

    def has_active_goal_ball(self):
        return bool(np.any(self.goal_ball_active))

    def sample_goal_ball(self, start_point, final_goal):
        desired_xy = start_point[:2] + np.random.uniform(0.35, 0.65) * (final_goal[:2] - start_point[:2])
        desired_z = np.clip(
            0.5 * (start_point[2] + final_goal[2]), self.envs[0].goal_z_range[0], self.envs[0].goal_z_range[1]
        )
        min_separation = self.goal_ball_min_xy_separation()

        scenario = self.active_scenario()
        free_space = getattr(scenario, "free_space", None)
        obstacle_map = getattr(scenario, "obstacle_map", None)
        cell_centers = getattr(scenario, "cell_centers", None)
        if not free_space or obstacle_map is None or cell_centers is None:
            desired = np.array([desired_xy[0], desired_xy[1], desired_z])
            if np.linalg.norm(desired[:2] - start_point[:2]) >= min_separation:
                return desired
            return None

        width = obstacle_map.shape[0]
        candidates = []
        for row, col in free_space:
            index = row + (width * col)
            pos_x, pos_y = cell_centers[index]
            candidate = np.array([pos_x, pos_y, desired_z])
            if np.linalg.norm(candidate[:2] - start_point[:2]) < min_separation:
                continue
            if not self.segment_is_obstacle_clear(start_point, candidate):
                continue
            candidates.append(candidate)

        if len(candidates) == 0:
            desired = np.array([desired_xy[0], desired_xy[1], desired_z])
            if (
                np.linalg.norm(desired[:2] - start_point[:2]) >= min_separation
                and self.segment_is_obstacle_clear(start_point, desired)
            ):
                return desired
            return None

        candidates = np.array(candidates)
        best_idx = int(np.argmin(np.linalg.norm(candidates[:, :2] - desired_xy[None, :], axis=1)))
        return candidates[best_idx]

    def segment_is_obstacle_clear(self, start_point, end_point):
        if not self.use_obstacles or self.obstacles is None or len(self.obstacles.pos_arr) == 0:
            return True

        start_xy = np.asarray(start_point[:2], dtype=np.float64)
        end_xy = np.asarray(end_point[:2], dtype=np.float64)
        segment = end_xy - start_xy
        segment_len_sq = float(np.dot(segment, segment))
        if segment_len_sq < EPS:
            return True

        inflated_radius = (
            0.5 * float(self.obst_size)
            + float(self.quad_collision_radius)
            + max(0.15, float(self.obstacle_guard_distance))
        )
        for obstacle_pos in self.obstacles.pos_arr:
            obstacle_xy = np.asarray(obstacle_pos[:2], dtype=np.float64)
            t = float(np.dot(obstacle_xy - start_xy, segment) / segment_len_sq)
            t = np.clip(t, 0.0, 1.0)
            closest_xy = start_xy + t * segment
            if np.linalg.norm(obstacle_xy - closest_xy) <= inflated_radius:
                return False

        return True

    @staticmethod
    def point_segment_distance(point, start_point, end_point):
        point = np.asarray(point, dtype=np.float64)
        start_point = np.asarray(start_point, dtype=np.float64)
        end_point = np.asarray(end_point, dtype=np.float64)
        segment = end_point - start_point
        segment_len_sq = float(np.dot(segment, segment))
        if segment_len_sq < EPS:
            return float(np.linalg.norm(point - end_point))
        t = float(np.dot(point - start_point, segment) / segment_len_sq)
        t = np.clip(t, 0.0, 1.0)
        closest = start_point + t * segment
        return float(np.linalg.norm(point - closest))

    def continuous_obstacle_collision_detection(self, prev_pos_quads, pos_quads):
        collided_quads_id = []
        quad_obst_pair = {}
        if not self.use_obstacles or self.obstacles is None or len(self.obstacles.pos_arr) == 0:
            return np.array([], dtype=np.int64), quad_obst_pair

        collide_threshold = 0.5 * float(self.obst_size) + float(self.obstacles.quad_radius)
        for quad_id in range(len(pos_quads)):
            start_xy = np.asarray(prev_pos_quads[quad_id, :2], dtype=np.float64)
            end_xy = np.asarray(pos_quads[quad_id, :2], dtype=np.float64)
            for obst_id, obstacle_pos in enumerate(self.obstacles.pos_arr):
                obstacle_xy = np.asarray(obstacle_pos[:2], dtype=np.float64)
                if self.point_segment_distance(obstacle_xy, start_xy, end_xy) <= collide_threshold:
                    collided_quads_id.append(int(quad_id))
                    quad_obst_pair[int(quad_id)] = int(obst_id)
                    break

        return np.array(collided_quads_id, dtype=np.int64), quad_obst_pair

    def build_goal_ball_sequence(self, start_point, final_goal):
        if self.goal_ball_count <= 0:
            return []

        start_point = np.array(start_point, copy=True)
        final_goal = np.array(final_goal, copy=True)
        fractions = np.linspace(1.0, self.goal_ball_count, self.goal_ball_count) / (self.goal_ball_count + 1.0)
        fractions += np.random.uniform(-0.03, 0.03, size=fractions.shape)
        fractions = np.clip(fractions, 0.08, 0.92)
        fractions = np.sort(fractions)

        z_min, z_max = self.envs[0].goal_z_range
        cruise_z = np.clip(0.5 * (z_min + z_max), z_min, z_max)
        min_separation = self.goal_ball_min_xy_separation()

        def desired_point(frac):
            desired_xy = start_point[:2] + frac * (final_goal[:2] - start_point[:2])
            desired_z = cruise_z
            return np.array([desired_xy[0], desired_xy[1], desired_z])

        scenario = self.active_scenario()
        free_space = getattr(scenario, "free_space", None)
        obstacle_map = getattr(scenario, "obstacle_map", None)
        cell_centers = getattr(scenario, "cell_centers", None)
        if not free_space or obstacle_map is None or cell_centers is None:
            sequence = []
            previous_point = start_point
            for frac in fractions:
                desired = desired_point(frac)
                if np.linalg.norm(desired[:2] - start_point[:2]) < min_separation:
                    continue
                if np.linalg.norm(desired[:2] - previous_point[:2]) < min_separation:
                    continue
                sequence.append(desired)
                previous_point = desired
            return sequence

        width = obstacle_map.shape[0]
        candidates_xy = []
        for row, col in free_space:
            index = row + (width * col)
            pos_x, pos_y = cell_centers[index]
            candidates_xy.append([pos_x, pos_y])

        if len(candidates_xy) == 0:
            return [desired_point(frac) for frac in fractions]

        candidates_xy = np.array(candidates_xy)
        used_indices = set()
        sequence = []
        min_separation = self.goal_ball_min_xy_separation()
        previous_point = start_point

        for frac in fractions:
            desired = desired_point(frac)
            order = np.argsort(np.linalg.norm(candidates_xy - desired[:2], axis=1))
            chosen_point = None

            for idx in order:
                if int(idx) in used_indices:
                    continue

                candidate = np.array([candidates_xy[idx, 0], candidates_xy[idx, 1], desired[2]])
                if np.linalg.norm(candidate[:2] - start_point[:2]) < min_separation:
                    continue
                if np.linalg.norm(candidate[:2] - previous_point[:2]) < min_separation:
                    continue
                if not self.segment_is_obstacle_clear(previous_point, candidate):
                    continue

                chosen_point = candidate
                used_indices.add(int(idx))
                break

            if chosen_point is None:
                if np.linalg.norm(desired[:2] - previous_point[:2]) < min_separation:
                    continue
                if np.linalg.norm(desired[:2] - start_point[:2]) < min_separation:
                    continue
                if not self.segment_is_obstacle_clear(previous_point, desired):
                    continue
                chosen_point = desired

            sequence.append(chosen_point)
            previous_point = chosen_point

        filtered_sequence = []
        previous_point = start_point
        for point in sequence:
            if np.linalg.norm(point[:2] - previous_point[:2]) < min_separation:
                continue
            if np.linalg.norm(point[:2] - start_point[:2]) < min_separation:
                continue
            if not self.segment_is_obstacle_clear(previous_point, point):
                continue
            filtered_sequence.append(point)
            previous_point = point

        if len(filtered_sequence) == 0:
            fallback_goal_ball = self.sample_goal_ball(start_point=start_point, final_goal=final_goal)
            if fallback_goal_ball is not None:
                filtered_sequence.append(fallback_goal_ball)

        return filtered_sequence

    def reset_goal_ball_velocity(self, agent_id):
        if not self.goal_ball_velocity_reset:
            return

        dynamics = self.envs[agent_id].dynamics
        dynamics.vel[:] *= self.goal_ball_velocity_reset_ratio
        dynamics.acc[:] = 0.0
        dynamics.omega[:] = 0.0
        dynamics.thrust_cmds_damp[:] = 0.0
        dynamics.thrust_rot_damp[:] = 0.0
        self.vel[agent_id, :] = dynamics.vel

        controller = getattr(self.envs[agent_id], "controller", None)
        if controller is not None and hasattr(controller, "reset"):
            controller.reset(dynamics)
        if controller is not None and hasattr(controller, "action"):
            controller.action = None

    def controller_avoidance_velocity(self, agent_id):
        if (
            not self.controller_obstacle_avoidance
            or not self.use_obstacles
            or self.obstacles is None
            or len(self.obstacles.pos_arr) == 0
            or self.obstacle_avoidance_distance <= 0.0
            or self.obstacle_avoidance_max_speed <= 0.0
        ):
            return np.zeros(2, dtype=np.float64), np.inf

        pos_xy = self.envs[agent_id].dynamics.pos[:2]
        obstacle_xy = self.obstacles.pos_arr[:, :2]
        delta = pos_xy[None, :] - obstacle_xy
        center_dist = np.linalg.norm(delta, axis=1)
        closest_idx = int(np.argmin(center_dist))
        closest_center_dist = float(center_dist[closest_idx])
        clearance = closest_center_dist - 0.5 * float(self.obst_size) - float(self.obstacles.quad_radius)
        if clearance >= self.obstacle_avoidance_distance:
            self.obstacle_avoidance_error_prev[agent_id] = 0.0
            self.obstacle_avoidance_error_integral[agent_id] = 0.0
            self.obstacle_avoidance_pid_initialized[agent_id] = False
            return np.zeros(2, dtype=np.float64), clearance

        if closest_center_dist > EPS:
            away_dir = delta[closest_idx] / closest_center_dist
        else:
            vel_xy = self.envs[agent_id].dynamics.vel[:2]
            speed_xy = np.linalg.norm(vel_xy)
            away_dir = -vel_xy / speed_xy if speed_xy > EPS else np.array([1.0, 0.0], dtype=np.float64)

        clearance_error = max(0.0, self.obstacle_avoidance_distance - clearance)
        dt = max(float(self.control_dt), EPS)
        if self.obstacle_avoidance_pid_initialized[agent_id]:
            error_derivative = (clearance_error - float(self.obstacle_avoidance_error_prev[agent_id])) / dt
        else:
            error_derivative = 0.0
            self.obstacle_avoidance_pid_initialized[agent_id] = True

        error_integral = float(self.obstacle_avoidance_error_integral[agent_id]) + clearance_error * dt
        if self.obstacle_avoidance_pid_integral_limit > 0.0:
            error_integral = float(np.clip(
                error_integral,
                -self.obstacle_avoidance_pid_integral_limit,
                self.obstacle_avoidance_pid_integral_limit,
            ))

        self.obstacle_avoidance_error_prev[agent_id] = clearance_error
        self.obstacle_avoidance_error_integral[agent_id] = error_integral

        pid_speed = (
            self.obstacle_avoidance_pid_kp * clearance_error
            + self.obstacle_avoidance_pid_ki * error_integral
            + self.obstacle_avoidance_pid_kd * max(0.0, error_derivative)
        )
        closeness = np.clip(clearance_error / self.obstacle_avoidance_distance, 0.0, 1.0)
        shaped_speed = self.obstacle_avoidance_max_speed * (closeness ** self.obstacle_avoidance_gain)
        repel_speed = np.clip(max(pid_speed, shaped_speed), 0.0, self.obstacle_avoidance_max_speed)
        return repel_speed * away_dir, clearance

    def assist_goal_ball_capture_action(self, agent_id, action):
        if (
            not self.goal_ball_capture_assist
            or not self.goal_ball_active[agent_id]
            or self.goal_ball_targets[agent_id] is None
            or self.goal_ball_capture_assist_distance <= 0.0
            or len(action) < 2
        ):
            return action

        target_delta_xy = self.goal_ball_targets[agent_id][:2] - self.envs[agent_id].dynamics.pos[:2]
        target_dist_xy = float(np.linalg.norm(target_delta_xy))
        if target_dist_xy >= self.goal_ball_capture_assist_distance or target_dist_xy < EPS:
            return action

        target_dir = target_delta_xy / target_dist_xy
        action_xy = np.array(action[:2], dtype=np.float64, copy=True)
        forward_speed = float(np.dot(action_xy, target_dir))
        tangent_vel = action_xy - forward_speed * target_dir

        closeness = 1.0 - target_dist_xy / self.goal_ball_capture_assist_distance
        min_capture_speed = self.goal_ball_capture_assist_speed * (0.35 + 0.65 * closeness)
        assisted_forward = max(forward_speed, min_capture_speed)
        assisted_tangent = tangent_vel * (1.0 - self.goal_ball_tangent_damping * closeness)
        action[:2] = assisted_forward * target_dir + assisted_tangent
        return action

    def apply_controller_action_assist(self, agent_id, action):
        action = np.array(action, dtype=np.float32, copy=True)
        if self.control_mode != "velocity" or len(action) < 3:
            return action

        action = self.assist_goal_ball_capture_action(agent_id, action)
        avoid_vel, clearance = self.controller_avoidance_velocity(agent_id)
        if np.isfinite(clearance):
            action[:2] = action[:2] + avoid_vel

        controller = getattr(self.envs[agent_id], "controller", None)
        if controller is not None and hasattr(controller, "_clip_velocity_command"):
            action = controller._clip_velocity_command(action)
        return action.astype(np.float32)

    def get_rel_pos_vel_item(self, env_id, indices=None):
        i = env_id

        if indices is None:
            # if not specified explicitly, consider all neighbors
            indices = [j for j in range(self.num_agents) if j != i]

        cur_pos = self.pos[i]
        cur_vel = self.vel[i]
        pos_neighbor = np.stack([self.pos[j] for j in indices])
        vel_neighbor = np.stack([self.vel[j] for j in indices])
        pos_rel = pos_neighbor - cur_pos
        vel_rel = vel_neighbor - cur_vel
        return pos_rel, vel_rel

    def get_obs_neighbor_rel(self, env_id, closest_drones):
        i = env_id
        pos_neighbors_rel, vel_neighbors_rel = self.get_rel_pos_vel_item(env_id=i, indices=closest_drones[i])
        obs_neighbor_rel = np.concatenate((pos_neighbors_rel, vel_neighbors_rel), axis=1)
        return obs_neighbor_rel

    def extend_obs_space(self, obs, closest_drones):
        obs_neighbors = []
        for i in range(len(self.envs)):
            obs_neighbor_rel = self.get_obs_neighbor_rel(env_id=i, closest_drones=closest_drones)
            obs_neighbors.append(obs_neighbor_rel.reshape(-1))
        obs_neighbors = np.stack(obs_neighbors)

        # clip observation space of neighborhoods
        obs_neighbors = np.clip(
            obs_neighbors, a_min=self.clip_neighbor_space_min_box, a_max=self.clip_neighbor_space_max_box,
        )
        obs_ext = np.concatenate((obs, obs_neighbors), axis=1)
        return obs_ext

    def neighborhood_indices(self):
        """Return a list of closest drones for each drone in the swarm."""
        # indices of all the other drones except us
        indices = [[j for j in range(self.num_agents) if i != j] for i in range(self.num_agents)]
        indices = np.array(indices)

        if self.num_use_neighbor_obs == self.num_agents - 1:
            return indices
        elif 1 <= self.num_use_neighbor_obs < self.num_agents - 1:
            close_neighbor_indices = []

            for i in range(self.num_agents):
                rel_pos, rel_vel = self.get_rel_pos_vel_item(env_id=i, indices=indices[i])
                rel_dist = np.linalg.norm(rel_pos, axis=1)
                rel_dist = np.maximum(rel_dist, 0.01)
                rel_pos_unit = rel_pos / rel_dist[:, None]

                # new relative distance is a new metric that combines relative position and relative velocity
                # the smaller the new_rel_dist, the closer the drones
                new_rel_dist = rel_dist + np.sum(rel_pos_unit * rel_vel, axis=1)

                rel_pos_index = new_rel_dist.argsort()
                rel_pos_index = rel_pos_index[:self.num_use_neighbor_obs]
                close_neighbor_indices.append(indices[i][rel_pos_index])

            return close_neighbor_indices
        else:
            raise RuntimeError("Incorrect number of neigbors")

    def add_neighborhood_obs(self, obs):
        indices = self.neighborhood_indices()
        obs_ext = self.extend_obs_space(obs, closest_drones=indices)
        return obs_ext

    def can_drones_fly(self):
        """
        Here we count the average number of collisions with the walls and ground in the last N episodes
        Returns: True if drones are considered proficient at flying
        """
        res = abs(np.mean(self.crashes_in_recent_episodes)) < 1 and len(self.crashes_in_recent_episodes) >= 10
        return res

    def calculate_room_collision(self):
        floor_collisions = np.array([env.dynamics.crashed_floor for env in self.envs])
        wall_collisions = np.array([env.dynamics.crashed_wall for env in self.envs])
        ceiling_collisions = np.array([env.dynamics.crashed_ceiling for env in self.envs])

        floor_crash_list = np.where(floor_collisions >= 1)[0]

        cur_wall_crash_list = np.where(wall_collisions >= 1)[0]
        wall_crash_list = np.setdiff1d(cur_wall_crash_list, self.prev_crashed_walls)

        cur_ceiling_crash_list = np.where(ceiling_collisions >= 1)[0]
        ceiling_crash_list = np.setdiff1d(cur_ceiling_crash_list, self.prev_crashed_ceiling)

        return floor_crash_list, wall_crash_list, ceiling_crash_list

    def obstacle_clearance(self, agent_id):
        if not self.use_obstacles or self.obstacles is None or len(self.obstacles.pos_arr) == 0:
            return np.inf

        pos_xy = self.envs[agent_id].dynamics.pos[:2]
        obstacle_xy = self.obstacles.pos_arr[:, :2]
        center_dist = np.linalg.norm(obstacle_xy - pos_xy, axis=1)
        min_center_dist = float(np.min(center_dist))
        return min_center_dist - 0.5 * float(self.obst_size) - float(self.obstacles.quad_radius)

    def obstacle_proximity_reward_raw(self, agent_id):
        clearance = self.obstacle_clearance(agent_id)
        safe_distance = max(float(self.obstacle_safe_distance), 1e-6)
        if not np.isfinite(clearance) or clearance >= safe_distance:
            return 0.0, clearance

        # Smoothly penalize entering the safety bubble around pillars, with a cap so
        # the binary collision penalty remains the dominant crash signal.
        ratio = np.clip((safe_distance - clearance) / safe_distance, 0.0, 2.0)
        return -float(ratio * ratio), clearance

    def wall_clearance(self, agent_id):
        pos = self.envs[agent_id].dynamics.pos
        room_box = self.envs[agent_id].room_box
        wall_distances = (
            pos[0] - room_box[0][0],
            room_box[1][0] - pos[0],
            pos[1] - room_box[0][1],
            room_box[1][1] - pos[1],
        )
        return float(np.min(wall_distances))

    def wall_proximity_reward_raw(self, agent_id):
        clearance = self.wall_clearance(agent_id)
        safe_distance = max(float(self.wall_safe_distance), 1e-6)
        if not np.isfinite(clearance) or clearance >= safe_distance:
            return 0.0, clearance

        ratio = np.clip((safe_distance - clearance) / safe_distance, 0.0, 2.0)
        return -float(ratio * ratio), clearance

    def obst_generation_given_density(self, grid_size=1.0):
        obst_area_length, obst_area_width = int(self.obst_spawn_area[0]), int(self.obst_spawn_area[1])
        num_room_grids = obst_area_length * obst_area_width

        cell_centers = get_cell_centers(obst_area_length=obst_area_length, obst_area_width=obst_area_width,
                                        grid_size=grid_size)

        room_map = [i for i in range(0, num_room_grids)]

        min_free_cells = max(2 * self.num_agents, 1)
        num_obstacles = int(round(num_room_grids * self.obst_density))
        num_obstacles = int(np.clip(num_obstacles, 0, max(num_room_grids - min_free_cells, 0)))
        obst_index = np.random.choice(a=room_map, size=num_obstacles, replace=False)

        obst_pos_arr = []
        # 0: No Obst, 1: Obst
        obst_map = np.zeros([obst_area_length, obst_area_width])
        for obst_id in obst_index:
            rid, cid = obst_id // obst_area_width, obst_id - (obst_id // obst_area_width) * obst_area_width
            obst_map[rid, cid] = 1
            obst_item = list(cell_centers[rid + int(obst_area_length / grid_size) * cid])
            obst_item.append(self.room_dims[2] / 2.)
            obst_pos_arr.append(obst_item)

        return obst_map, obst_pos_arr, cell_centers

    def init_scene_multi(self):
        models = tuple(e.dynamics.model for e in self.envs)
        for i in range(len(self.quads_view_mode)):
            self.scenes.append(Quadrotor3DSceneMulti(
                models=models,
                w=480, h=360, resizable=True, viewpoint=self.quads_view_mode[i],
                obs_hw=self.camera_hw,
                room_dims=self.room_dims, num_agents=self.num_agents,
                render_speed=self.render_speed, formation_size=self.quads_formation_size, obstacles=self.obstacles,
                vis_vel_arrows=False, vis_acc_arrows=False, viz_traces=0, viz_trace_nth_step=1,
                num_obstacles=self.num_obstacles, scene_index=i, camera_fov_deg=self.camera_fov,
                camera_pitch_deg=self.camera_pitch_deg,
                camera_drone_index=self.camera_drone_index
            ))

    def reset(self, obst_density=None, obst_size=None):
        obs, rewards, dones, infos = [], [], [], []

        if obst_density:
            self.obst_density = obst_density
        if obst_size:
            self.obst_size = obst_size

        # Scenario reset
        if self.use_obstacles:
            self.obstacles = MultiObstacles(obstacle_size=self.obst_size, quad_radius=self.quad_collision_radius)
            self.obst_map, obst_pos_arr, cell_centers = self.obst_generation_given_density()
            self.obstacles.pos_arr = copy.deepcopy(np.array(obst_pos_arr))
            self.num_obstacles = len(obst_pos_arr)
            self.cell_centers = cell_centers
            self.scenario.reset(obst_map=self.obst_map, cell_centers=cell_centers)
        else:
            self.num_obstacles = 0
            self.scenario.reset()

        # Replay buffer
        if self.use_replay_buffer and not self.activate_replay_buffer:
            self.crashes_in_recent_episodes.append(self.crashes_last_episode)
            self.activate_replay_buffer = self.can_drones_fly()
            self.crashes_last_episode = 0

        self.goal_ball_targets = [None for _ in range(self.num_agents)]
        self.goal_ball_sequences = [[] for _ in range(self.num_agents)]
        self.goal_ball_active = np.zeros(self.num_agents, dtype=bool)
        self.final_goals = [None for _ in range(self.num_agents)]
        self.goal_ball_collected_per_episode = 0
        self.goal_ball_targets_total = 0

        for i, e in enumerate(self.envs):
            self.final_goals[i] = np.array(self.scenario.goals[i], copy=True)
            if self.scenario.spawn_points is None:
                e.spawn_point = np.array(self.scenario.goals[i], copy=True)
            else:
                e.spawn_point = np.array(self.scenario.spawn_points[i], copy=True)

            e.goal = self.final_goals[i]
            if self.use_goal_ball and self.goal_ball_count > 0:
                goal_sequence = self.build_goal_ball_sequence(start_point=e.spawn_point, final_goal=self.final_goals[i])
                self.goal_ball_sequences[i] = [np.array(goal_ball, copy=True) for goal_ball in goal_sequence]
                self.goal_ball_targets_total += len(self.goal_ball_sequences[i])
                if len(self.goal_ball_sequences[i]) > 0:
                    self.goal_ball_targets[i] = np.array(self.goal_ball_sequences[i][0], copy=True)
                    self.goal_ball_active[i] = True
                    e.goal = self.goal_ball_targets[i]
            e.rew_coeff = self.rew_coeff

            observation = e.reset()
            obs.append(observation)
            self.pos[i, :] = e.dynamics.pos

        # Neighbors
        if self.num_use_neighbor_obs > 0:
            obs = self.add_neighborhood_obs(obs)

        # Obstacles
        if self.use_obstacles:
            quads_pos = np.array([e.dynamics.pos for e in self.envs])
            if self.obstacle_obs_type == 'octomap':
                obs = self.obstacles.reset(obs=obs, quads_pos=quads_pos, pos_arr=obst_pos_arr)
                self.latest_obstacle_obs = np.zeros((self.num_agents, 0), dtype=np.float32)
            else:
                self.obstacles.pos_arr = copy.deepcopy(np.array(obst_pos_arr))
                obs = self.append_obstacle_observations(obs)
            self.obst_quad_collisions_per_episode = self.obst_quad_collisions_after_settle = 0
            self.prev_obst_quad_collisions = []
            self.distance_to_goal_3_5 = 0
            self.distance_to_goal_5 = 0

        # Collision
        # # Collision: Neighbor
        self.collisions_per_episode = self.collisions_after_settle = self.collisions_final_5s = 0
        self.prev_drone_collisions = []
        self.safe_flight_streak_steps[:] = 0.0
        self.max_safe_flight_streak_steps[:] = 0.0
        self.prev_obstacle_clearance[:] = np.inf
        self.prev_wall_clearance[:] = np.inf
        self.obstacle_avoidance_error_prev[:] = 0.0
        self.obstacle_avoidance_error_integral[:] = 0.0
        self.obstacle_avoidance_pid_initialized[:] = False

        # # Collision: Room
        self.collisions_room_per_episode = 0
        self.collisions_floor_per_episode = self.collisions_wall_per_episode = self.collisions_ceiling_per_episode = 0
        self.prev_crashed_walls = []
        self.prev_crashed_ceiling = []
        self.prev_crashed_room = []

        # Log
        # # Final Distance (1s / 3s / 5s)
        self.distance_to_goal = [[] for _ in range(len(self.envs))]
        self.agent_col_agent = np.ones(self.num_agents)
        self.agent_col_obst = np.ones(self.num_agents)
        self.reached_goal = [False for _ in range(len(self.envs))]

        # Rendering / Camera state
        self.reset_scene = True
        self.quads_formation_size = self.scenario.formation_size
        self.all_collisions = {val: [0.0 for _ in range(len(self.envs))] for val in ['drone', 'ground', 'obstacle']}
        self.latest_fpv_frame = None

        return obs

    def step(self, actions):
        obs, rewards, dones, infos = [], [], [], []
        goal_ball_episode_finished = False
        prev_pos = np.array(self.pos, copy=True)

        if self.use_goal_ball:
            self.sync_active_goals()

        for i, a in enumerate(actions):
            self.envs[i].rew_coeff = self.rew_coeff
            a = self.apply_controller_action_assist(i, a)

            observation, reward, done, info = self.envs[i].step(a)
            obs.append(observation)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

            self.pos[i, :] = self.envs[i].dynamics.pos

        # 1. Calculate collisions: 1) between drones 2) with obstacles 3) with room
        # 1) Collisions between drones
        drone_col_matrix, curr_drone_collisions, distance_matrix = \
            calculate_collision_matrix(positions=self.pos, collision_threshold=self.collision_threshold)

        # # Filter curr_drone_collisions
        curr_drone_collisions = curr_drone_collisions.astype(int)
        curr_drone_collisions = np.delete(curr_drone_collisions, np.unique(
            np.where(curr_drone_collisions == [-1000, -1000])[0]), axis=0)

        old_quad_collision = set(map(tuple, self.prev_drone_collisions))
        new_quad_collision = np.array([x for x in curr_drone_collisions if tuple(x) not in old_quad_collision])

        self.last_step_unique_collisions = np.setdiff1d(curr_drone_collisions, self.prev_drone_collisions)

        # # Filter distance_matrix; Only contains quadrotor pairs with distance <= self.collision_threshold
        near_quad_ids = np.where(distance_matrix[:, 2] <= self.collision_falloff_threshold)
        distance_matrix = distance_matrix[near_quad_ids]

        # Collision between 2 drones counts as a single collision
        # # Calculate collisions (i) All collisions (ii) collisions after grace period
        collisions_curr_tick = len(self.last_step_unique_collisions) // 2
        self.collisions_per_episode += collisions_curr_tick

        if collisions_curr_tick > 0 and self.envs[0].tick >= self.collisions_grace_period_steps:
            self.collisions_after_settle += collisions_curr_tick
            for agent_id in self.last_step_unique_collisions:
                self.agent_col_agent[agent_id] = 0
        if collisions_curr_tick > 0 and self.envs[0].time_remain <= self.collisions_final_grace_period_steps:
            self.collisions_final_5s += collisions_curr_tick

        # # Aux: Neighbor Collisions
        self.prev_drone_collisions = curr_drone_collisions

        # 2) Collisions with obstacles
        obstacle_collision_terminated = np.zeros(self.num_agents, dtype=bool)
        obstacle_guard_terminated = np.zeros(self.num_agents, dtype=bool)
        if self.use_obstacles:
            rew_obst_quad_collisions_raw = np.zeros(self.num_agents)
            point_obst_quad_col_matrix, point_quad_obst_pair = self.obstacles.collision_detection(pos_quads=self.pos)
            segment_obst_quad_col_matrix, segment_quad_obst_pair = self.continuous_obstacle_collision_detection(
                prev_pos_quads=prev_pos, pos_quads=self.pos
            )
            obst_quad_col_matrix = np.unique(
                np.concatenate([point_obst_quad_col_matrix.astype(np.int64), segment_obst_quad_col_matrix])
            )
            quad_obst_pair = dict(segment_quad_obst_pair)
            quad_obst_pair.update(point_quad_obst_pair)
            # We assume drone can only collide with one obstacle at the same time.
            # Given this setting, in theory, the gap between obstacles should >= 0.1 (drone diameter: 0.46*2 = 0.92)
            self.curr_quad_col = np.setdiff1d(obst_quad_col_matrix, self.prev_obst_quad_collisions)
            collisions_obst_curr_tick = len(self.curr_quad_col)
            self.obst_quad_collisions_per_episode += collisions_obst_curr_tick

            if collisions_obst_curr_tick > 0 and self.envs[0].tick >= self.collisions_grace_period_steps:
                self.obst_quad_collisions_after_settle += collisions_obst_curr_tick
                for qid in self.curr_quad_col:
                    q_rel_dist = np.linalg.norm(self.metric_goal(qid) - self.envs[qid].dynamics.pos)
                    if q_rel_dist > 3.5:
                        self.distance_to_goal_3_5 += 1
                    if q_rel_dist > 5.0:
                        self.distance_to_goal_5 += 1
                    # Used for log agent_success
                    self.agent_col_obst[qid] = 0

            # # Aux: Obstacle Collisions
            self.prev_obst_quad_collisions = obst_quad_col_matrix

            if len(obst_quad_col_matrix) > 0:
                # Penalize only the first contact for this collision event, not every
                # frame while the drone is still brushing the same pillar.
                rew_obst_quad_collisions_raw[self.curr_quad_col] = -1.0
                if self.terminate_on_obstacle_collision:
                    obstacle_collision_terminated[obst_quad_col_matrix] = True
        else:
            obst_quad_col_matrix = np.array([], dtype=np.int64)
            rew_obst_quad_collisions_raw = np.zeros(self.num_agents)

        # 3) Collisions with room
        floor_crash_list, wall_crash_list, ceiling_crash_list = self.calculate_room_collision()
        room_crash_raw = np.unique(np.concatenate([floor_crash_list, wall_crash_list, ceiling_crash_list]))
        room_crash_list = np.setdiff1d(room_crash_raw, self.prev_crashed_room)
        floor_collision_terminated = np.zeros(self.num_agents, dtype=bool)
        ceiling_collision_terminated = np.zeros(self.num_agents, dtype=bool)
        wall_collision_terminated = np.zeros(self.num_agents, dtype=bool)
        if len(floor_crash_list) > 0:
            floor_collision_terminated[floor_crash_list] = True
        if len(ceiling_crash_list) > 0:
            ceiling_collision_terminated[ceiling_crash_list] = True
        if len(wall_crash_list) > 0 and self.terminate_on_wall_collision:
            wall_collision_terminated[wall_crash_list] = True
        # # Aux: Room Collisions
        self.prev_crashed_walls = wall_crash_list
        self.prev_crashed_ceiling = ceiling_crash_list
        self.prev_crashed_room = room_crash_raw

        # 2. Calculate rewards and infos for collision
        # 1) Between drones
        rew_collisions_raw = np.zeros(self.num_agents)
        if self.last_step_unique_collisions.any():
            rew_collisions_raw[self.last_step_unique_collisions] = -1.0
        rew_collisions = self.rew_coeff["quadcol_bin"] * rew_collisions_raw

        # penalties for being too close to other drones
        if len(distance_matrix) > 0:
            rew_proximity = -1.0 * calculate_drone_proximity_penalties(
                distance_matrix=distance_matrix, collision_falloff_threshold=self.collision_falloff_threshold,
                dt=self.control_dt, max_penalty=self.rew_coeff["quadcol_bin_smooth_max"], num_agents=self.num_agents,
            )
        else:
            rew_proximity = np.zeros(self.num_agents)

        # 2) With obstacles
        rew_collisions_obst_quad = np.zeros(self.num_agents)
        if self.use_obstacles:
            rew_obst_proximity_raw = np.zeros(self.num_agents)
            obst_clearance = np.full(self.num_agents, np.inf)
            for agent_id in range(self.num_agents):
                rew_obst_proximity_raw[agent_id], obst_clearance[agent_id] = \
                    self.obstacle_proximity_reward_raw(agent_id)
            if self.obstacle_guard_distance > 0.0:
                obstacle_guard_terminated = np.logical_and(
                    np.isfinite(obst_clearance),
                    obst_clearance <= self.obstacle_guard_distance,
                )
                guard_only = np.logical_and(obstacle_guard_terminated, ~obstacle_collision_terminated)
                rew_obst_quad_collisions_raw[guard_only] = -self.guard_collision_raw_penalty
            rew_collisions_obst_quad = self.rew_coeff["quadcol_bin_obst"] * rew_obst_quad_collisions_raw
            rew_obst_proximity = self.control_dt * self.rew_coeff["obst_proximity"] * rew_obst_proximity_raw
        else:
            rew_obst_proximity_raw = np.zeros(self.num_agents)
            rew_obst_proximity = np.zeros(self.num_agents)
            obst_clearance = np.full(self.num_agents, np.inf)

        # 3) With horizontal room walls
        rew_wall_collisions_raw = np.zeros(self.num_agents)
        wall_guard_terminated = np.zeros(self.num_agents, dtype=bool)
        rew_wall_proximity_raw = np.zeros(self.num_agents)
        wall_clearance = np.full(self.num_agents, np.inf)
        for agent_id in range(self.num_agents):
            rew_wall_proximity_raw[agent_id], wall_clearance[agent_id] = \
                self.wall_proximity_reward_raw(agent_id)
        if len(wall_crash_list) > 0:
            rew_wall_collisions_raw[room_crash_list] = -1.0
        if self.wall_guard_distance > 0.0:
            wall_guard_terminated = np.logical_and(
                np.isfinite(wall_clearance),
                wall_clearance <= self.wall_guard_distance,
            )
            guard_only = np.logical_and(wall_guard_terminated, ~wall_collision_terminated)
            rew_wall_collisions_raw[guard_only] = -self.guard_collision_raw_penalty
        rew_wall_collisions = self.rew_coeff["wallcol_bin"] * rew_wall_collisions_raw
        rew_wall_proximity = self.control_dt * self.rew_coeff["wall_proximity"] * rew_wall_proximity_raw

        rew_obst_clearance_delta_raw = np.zeros(self.num_agents)
        rew_wall_clearance_delta_raw = np.zeros(self.num_agents)
        for agent_id in range(self.num_agents):
            prev_obst_clearance = self.prev_obstacle_clearance[agent_id]
            if np.isfinite(prev_obst_clearance) and np.isfinite(obst_clearance[agent_id]):
                rew_obst_clearance_delta_raw[agent_id] = np.clip(
                    obst_clearance[agent_id] - prev_obst_clearance, -1.0, 1.0
                )

            prev_wall_clearance = self.prev_wall_clearance[agent_id]
            if np.isfinite(prev_wall_clearance) and np.isfinite(wall_clearance[agent_id]):
                rew_wall_clearance_delta_raw[agent_id] = np.clip(
                    wall_clearance[agent_id] - prev_wall_clearance, -1.0, 1.0
                )
        rew_obst_clearance_delta = (
            self.control_dt * self.rew_coeff["obstacle_clearance_delta"] * rew_obst_clearance_delta_raw
        )
        rew_wall_clearance_delta = (
            self.control_dt * self.rew_coeff["wall_clearance_delta"] * rew_wall_clearance_delta_raw
        )

        # Reward uninterrupted flight in the safe part of the room. The small streak
        # multiplier makes longer collision-free runs more valuable than brief pauses.
        rew_safe_flight_raw = np.zeros(self.num_agents)
        rew_safe_flight = np.zeros(self.num_agents)
        rew_path_alignment_raw = np.zeros(self.num_agents)
        rew_path_alignment = np.zeros(self.num_agents)
        room_crash_flags = np.zeros(self.num_agents, dtype=bool)
        room_crash_flags[room_crash_list] = True
        for agent_id in range(self.num_agents):
            obstacle_margin_ok = (
                not self.use_obstacles
                or not np.isfinite(obst_clearance[agent_id])
                or obst_clearance[agent_id] >= 0.5 * self.obstacle_safe_distance
            )
            wall_margin_ok = (
                not np.isfinite(wall_clearance[agent_id])
                or wall_clearance[agent_id] >= 0.5 * self.wall_safe_distance
            )
            collision_now = (
                rew_collisions_raw[agent_id] < 0.0
                or rew_obst_quad_collisions_raw[agent_id] < 0.0
                or rew_wall_collisions_raw[agent_id] < 0.0
                or room_crash_flags[agent_id]
            )
            can_reward_safe = (
                self.envs[0].tick >= self.collisions_grace_period_steps
                and not collision_now
                and obstacle_margin_ok
                and wall_margin_ok
            )
            if can_reward_safe:
                self.safe_flight_streak_steps[agent_id] += 1.0
                self.max_safe_flight_streak_steps[agent_id] = max(
                    self.max_safe_flight_streak_steps[agent_id],
                    self.safe_flight_streak_steps[agent_id],
                )
                streak_seconds = self.safe_flight_streak_steps[agent_id] * self.control_dt
                rew_safe_flight_raw[agent_id] = 1.0 + min(streak_seconds / 4.0, 1.0)
            else:
                self.safe_flight_streak_steps[agent_id] = 0.0
            rew_safe_flight[agent_id] = self.control_dt * self.rew_coeff["safe_flight"] * rew_safe_flight_raw[agent_id]

            # In open space, discourage orbiting around the target or continuing
            # sideways after an avoidance turn. Near obstacles/walls this is disabled
            # so the policy can still sidestep to avoid a pillar.
            if can_reward_safe:
                goal_delta_xy = self.active_target(agent_id)[:2] - self.envs[agent_id].dynamics.pos[:2]
                goal_dist_xy = float(np.linalg.norm(goal_delta_xy))
                vel_xy = self.envs[agent_id].dynamics.vel[:2]
                speed_limit = max(float(self.envs[agent_id].velocity_max_xy), 1e-6)
                if goal_dist_xy > 0.75:
                    goal_dir = goal_delta_xy / goal_dist_xy
                    forward_speed = float(np.dot(vel_xy, goal_dir))
                    lateral_speed = float(goal_dir[0] * vel_xy[1] - goal_dir[1] * vel_xy[0])
                    lateral_ratio = abs(lateral_speed) / speed_limit
                    away_ratio = max(0.0, -forward_speed) / speed_limit
                    rew_path_alignment_raw[agent_id] = -(lateral_ratio * lateral_ratio + away_ratio * away_ratio)
                    rew_path_alignment[agent_id] = (
                        self.control_dt * self.rew_coeff["path_alignment"] * rew_path_alignment_raw[agent_id]
                    )

        if self.envs[0].tick >= self.collisions_grace_period_steps:
            self.collisions_room_per_episode += len(room_crash_list)
            self.collisions_floor_per_episode += len(floor_crash_list)
            self.collisions_wall_per_episode += len(wall_crash_list)
            self.collisions_ceiling_per_episode += len(ceiling_crash_list)

        # Reward & Info
        goal_changed_ids = []
        for i in range(self.num_agents):
            rewards[i] += rew_collisions[i]
            rewards[i] += rew_proximity[i]
            rewards[i] += rew_wall_collisions[i]
            rewards[i] += rew_wall_proximity[i]
            rewards[i] += rew_wall_clearance_delta[i]
            rewards[i] += rew_safe_flight[i]
            rewards[i] += rew_path_alignment[i]

            infos[i]["rewards"]["rew_quadcol"] = rew_collisions[i]
            infos[i]["rewards"]["rew_proximity"] = rew_proximity[i]
            infos[i]["rewards"]["rewraw_quadcol"] = rew_collisions_raw[i]
            infos[i]["rewards"]["rew_wall_collision"] = rew_wall_collisions[i]
            infos[i]["rewards"]["rewraw_wall_collision"] = rew_wall_collisions_raw[i]
            infos[i]["rewards"]["rew_wall_proximity"] = rew_wall_proximity[i]
            infos[i]["rewards"]["rewraw_wall_proximity"] = rew_wall_proximity_raw[i]
            infos[i]["rewards"]["rew_wall_clearance_delta"] = rew_wall_clearance_delta[i]
            infos[i]["rewards"]["rewraw_wall_clearance_delta"] = rew_wall_clearance_delta_raw[i]
            infos[i]["rewards"]["rew_safe_flight"] = rew_safe_flight[i]
            infos[i]["rewards"]["rewraw_safe_flight"] = self.control_dt * rew_safe_flight_raw[i]
            infos[i]["rewards"]["rew_path_alignment"] = rew_path_alignment[i]
            infos[i]["rewards"]["rewraw_path_alignment"] = self.control_dt * rew_path_alignment_raw[i]
            infos[i]["rewards"]["rew_main"] += rew_safe_flight[i]
            infos[i]["rewards"]["rewraw_main"] += self.control_dt * rew_safe_flight_raw[i]
            infos[i]["rewards"]["rew_main"] += rew_path_alignment[i]
            infos[i]["rewards"]["rewraw_main"] += self.control_dt * rew_path_alignment_raw[i]
            infos[i]["rewards"]["rew_main"] += rew_wall_clearance_delta[i]
            infos[i]["rewards"]["rewraw_main"] += self.control_dt * rew_wall_clearance_delta_raw[i]
            infos[i]["wall_clearance"] = wall_clearance[i]
            infos[i]["floor_collision_terminated"] = bool(floor_collision_terminated[i])
            infos[i]["wall_collision_terminated"] = bool(wall_collision_terminated[i])
            infos[i]["ceiling_collision_terminated"] = bool(ceiling_collision_terminated[i])
            infos[i]["wall_guard_terminated"] = bool(wall_guard_terminated[i])
            infos[i]["safe_flight_streak_s"] = float(self.safe_flight_streak_steps[i] * self.control_dt)
            infos[i]["rewards"]["rew_goal_ball"] = 0.0
            infos[i]["rewards"]["rewraw_goal_ball"] = 0.0

            if self.use_obstacles:
                if obstacle_guard_terminated[i]:
                    self.agent_col_obst[i] = 0
                rewards[i] += rew_collisions_obst_quad[i]
                rewards[i] += rew_obst_proximity[i]
                rewards[i] += rew_obst_clearance_delta[i]
                infos[i]["rewards"]["rew_quadcol_obstacle"] = rew_collisions_obst_quad[i]
                infos[i]["rewards"]["rewraw_quadcol_obstacle"] = rew_obst_quad_collisions_raw[i]
                infos[i]["rewards"]["rew_obst_proximity"] = rew_obst_proximity[i]
                infos[i]["rewards"]["rewraw_obst_proximity"] = rew_obst_proximity_raw[i]
                infos[i]["rewards"]["rew_obst_clearance_delta"] = rew_obst_clearance_delta[i]
                infos[i]["rewards"]["rewraw_obst_clearance_delta"] = rew_obst_clearance_delta_raw[i]
                infos[i]["rewards"]["rew_main"] += rew_obst_clearance_delta[i]
                infos[i]["rewards"]["rewraw_main"] += self.control_dt * rew_obst_clearance_delta_raw[i]
                infos[i]["obstacle_clearance"] = obst_clearance[i]
                infos[i]["obstacle_collision_terminated"] = bool(obstacle_collision_terminated[i])
                infos[i]["obstacle_guard_terminated"] = bool(obstacle_guard_terminated[i])
            if self.goal_ball_active[i]:
                dist_to_ball = np.linalg.norm(self.goal_ball_targets[i] - self.envs[i].dynamics.pos)
                segment_dist_to_ball = self.point_segment_distance(
                    self.goal_ball_targets[i], prev_pos[i], self.envs[i].dynamics.pos
                )
                if min(dist_to_ball, segment_dist_to_ball) <= self.goal_ball_radius:
                    rewards[i] += self.goal_ball_reward
                    infos[i]["rewards"]["rew_goal_ball"] = self.goal_ball_reward
                    infos[i]["rewards"]["rewraw_goal_ball"] = self.goal_ball_reward
                    infos[i]["rewards"]["rew_main"] += self.goal_ball_reward
                    infos[i]["rewards"]["rewraw_main"] += self.goal_ball_reward
                    self.goal_ball_collected_per_episode += 1
                    if len(self.goal_ball_sequences[i]) > 0:
                        self.goal_ball_sequences[i].pop(0)

                    if len(self.goal_ball_sequences[i]) > 0:
                        self.goal_ball_targets[i] = np.array(self.goal_ball_sequences[i][0], copy=True)
                        self.goal_ball_active[i] = True
                        self.envs[i].goal = self.goal_ball_targets[i]
                    else:
                        self.goal_ball_targets[i] = None
                        self.goal_ball_active[i] = False
                        self.envs[i].goal = self.final_goals[i]
                    self.reset_goal_ball_velocity(i)
                    goal_changed_ids.append(i)
                    if self.goal_ball_targets_total > 0 and self.goal_ball_collected_per_episode >= self.goal_ball_targets_total:
                        goal_ball_episode_finished = True

            metric_goal_dist = np.linalg.norm(self.active_target(i) - self.envs[i].dynamics.pos)
            self.distance_to_goal[i].append(self.envs[0].dt * metric_goal_dist)
            if len(self.distance_to_goal[i]) >= 5 and \
                    np.mean(self.distance_to_goal[i][-5:]) / self.envs[0].dt < self.scenario.approch_goal_metric \
                    and not self.reached_goal[i]:
                self.reached_goal[i] = True

        self.prev_obstacle_clearance[:] = obst_clearance
        self.prev_wall_clearance[:] = wall_clearance

        # 3. Applying random forces: 1) aerodynamics 2) between drones 3) obstacles 4) room
        self_state_update_flag = False

        # # 1) aerodynamics
        if self.use_downwash:
            envs_dynamics = [env.dynamics for env in self.envs]
            applied_downwash_list = perform_downwash(drones_dyn=envs_dynamics, dt=self.control_dt)
            downwash_agents_list = np.where(applied_downwash_list == 1)[0]
            if len(downwash_agents_list) > 0:
                self_state_update_flag = True

        # # 2) Drones
        if self.apply_collision_force:
            if len(new_quad_collision) > 0:
                self_state_update_flag = True
                for val in new_quad_collision:
                    dyn1, dyn2 = self.envs[val[0]].dynamics, self.envs[val[1]].dynamics
                    dyn1.vel, dyn1.omega, dyn2.vel, dyn2.omega = perform_collision_between_drones(
                        pos1=dyn1.pos, vel1=dyn1.vel, omega1=dyn1.omega, pos2=dyn2.pos, vel2=dyn2.vel, omega2=dyn2.omega)

            # # 3) Obstacles
            if self.use_obstacles:
                if len(obst_quad_col_matrix) > 0:
                    self_state_update_flag = True
                    for val in obst_quad_col_matrix:
                        if obstacle_collision_terminated[int(val)]:
                            continue
                        obstacle_id = quad_obst_pair[int(val)]
                        obstacle_pos = self.obstacles.pos_arr[int(obstacle_id)]
                        perform_collision_with_obstacle(drone_dyn=self.envs[int(val)].dynamics,
                                                        obstacle_pos=obstacle_pos,
                                                        obstacle_size=self.obst_size,
                                                        quad_radius=self.obstacles.quad_radius,
                                                        prev_pos=prev_pos[int(val)])

            # # 4) Room
            if len(wall_crash_list) > 0 or len(ceiling_crash_list) > 0:
                self_state_update_flag = True

                for val in wall_crash_list:
                    if wall_collision_terminated[val]:
                        continue
                    perform_collision_with_wall(drone_dyn=self.envs[val].dynamics, room_box=self.envs[0].room_box)

                for val in ceiling_crash_list:
                    if ceiling_collision_terminated[val]:
                        continue
                    perform_collision_with_ceiling(drone_dyn=self.envs[val].dynamics)

        # 4. Run the scenario passed to self.quads_mode. While reward-ball
        # navigation is active, keep the scenario from replacing env.goal with
        # its timed final target and changing the policy input mid-flight.
        if not goal_ball_episode_finished and not (self.use_goal_ball and self.has_active_goal_ball()):
            self.scenario.step()
        if self.use_goal_ball:
            self.sync_active_goals()

        # 5. Collect final observations
        # Collect positions after physical interaction
        for i in range(self.num_agents):
            self.pos[i, :] = self.envs[i].dynamics.pos
            self.vel[i, :] = self.envs[i].dynamics.vel

        if self_state_update_flag:
            obs = [e.state_vector(e) for e in self.envs]
        elif len(goal_changed_ids) > 0:
            for goal_changed_id in goal_changed_ids:
                obs[goal_changed_id] = self.envs[goal_changed_id].state_vector(self.envs[goal_changed_id])

        # Concatenate observations of neighbor drones
        if self.num_use_neighbor_obs > 0:
            obs = self.add_neighborhood_obs(obs)

        # Concatenate obstacle observations
        if self.use_obstacles:
            obs = self.append_obstacle_observations(obs)
            if self.obstacle_obs_type == 'yolo':
                for i in range(self.num_agents):
                    infos[i]["yolo_detections"] = self.latest_obstacle_detections[i]
                    infos[i]["yolo_observation"] = self.latest_obstacle_obs[i]
            elif self.obstacle_obs_type == 'depth':
                for i in range(self.num_agents):
                    infos[i]["depth_observation"] = self.latest_obstacle_obs[i]
            elif self.obstacle_obs_type == 'lidar':
                for i in range(self.num_agents):
                    infos[i]["lidar_observation"] = self.latest_obstacle_obs[i]

        # 6. Update info for replay buffer
        # Once agent learns how to take off, activate the replay buffer
        if self.use_replay_buffer and not self.activate_replay_buffer:
            self.crashes_last_episode += infos[0]["rewards"]["rew_crash"]

        # Rendering / Camera overlays
        ground_collisions = [1.0 if env.dynamics.on_floor else 0.0 for env in self.envs]
        if self.use_obstacles:
            obst_coll = [1.0 if i < 0 else 0.0 for i in rew_obst_quad_collisions_raw]
        else:
            obst_coll = [0.0 for _ in range(self.num_agents)]
        self.all_collisions = {'drone': drone_col_matrix, 'ground': ground_collisions,
                               'obstacle': obst_coll}

        # 7. DONES
        if any(dones):
            scenario_name = self.scenario.name()[9:]
            for i in range(len(infos)):
                if self.saved_in_replay_buffer:
                    infos[i]['episode_extra_stats'] = {
                        'num_collisions_replay': self.collisions_per_episode,
                        'num_collisions_obst_replay': self.obst_quad_collisions_per_episode,
                    }
                else:
                    self.distance_to_goal = np.array(self.distance_to_goal)
                    self.reached_goal = np.array(self.reached_goal)
                    infos[i]['episode_extra_stats'] = {
                        'num_collisions': self.collisions_per_episode,
                        'num_collisions_with_room': self.collisions_room_per_episode,
                        'num_collisions_with_floor': self.collisions_floor_per_episode,
                        'num_collisions_with_wall': self.collisions_wall_per_episode,
                        'num_collisions_with_ceiling': self.collisions_ceiling_per_episode,
                        'goal_ball_collected': self.goal_ball_collected_per_episode,
                        'goal_ball_targets_total': self.goal_ball_targets_total,
                        'num_obstacles': self.num_obstacles,
                        'obstacle_density': self.obst_density if self.use_obstacles else 0.0,
                        'obstacle_size': self.obst_size if self.use_obstacles else 0.0,
                        'max_safe_flight_streak_s': float(
                            np.max(self.max_safe_flight_streak_steps) * self.control_dt
                        ),
                        f'{scenario_name}/goal_ball_collected': self.goal_ball_collected_per_episode,
                        f'{scenario_name}/goal_ball_targets_total': self.goal_ball_targets_total,
                        'num_collisions_after_settle': self.collisions_after_settle,
                        f'{scenario_name}/num_collisions': self.collisions_after_settle,

                        'num_collisions_final_5_s': self.collisions_final_5s,
                        f'{scenario_name}/num_collisions_final_5_s': self.collisions_final_5s,

                        'distance_to_goal_1s': (1.0 / self.envs[0].dt) * np.mean(
                            self.distance_to_goal[i, int(-1 * self.control_freq):]),
                        'distance_to_goal_3s': (1.0 / self.envs[0].dt) * np.mean(
                            self.distance_to_goal[i, int(-3 * self.control_freq):]),
                        'distance_to_goal_5s': (1.0 / self.envs[0].dt) * np.mean(
                            self.distance_to_goal[i, int(-5 * self.control_freq):]),

                        f'{scenario_name}/distance_to_goal_1s': (1.0 / self.envs[0].dt) * np.mean(
                            self.distance_to_goal[i, int(-1 * self.control_freq):]),
                        f'{scenario_name}/distance_to_goal_3s': (1.0 / self.envs[0].dt) * np.mean(
                            self.distance_to_goal[i, int(-3 * self.control_freq):]),
                        f'{scenario_name}/distance_to_goal_5s': (1.0 / self.envs[0].dt) * np.mean(
                            self.distance_to_goal[i, int(-5 * self.control_freq):]),
                    }

                    if self.use_obstacles:
                        infos[i]['episode_extra_stats']['num_collisions_obst_quad'] = \
                            self.obst_quad_collisions_per_episode
                        infos[i]['episode_extra_stats']['num_collisions_obst_quad_after_settle'] = \
                            self.obst_quad_collisions_after_settle
                        infos[i]['episode_extra_stats'][f'{scenario_name}/num_collisions_obst'] = \
                            self.obst_quad_collisions_per_episode

                        infos[i]['episode_extra_stats']['num_collisions_obst_quad_3_5'] = \
                            self.distance_to_goal_3_5
                        infos[i]['episode_extra_stats'][f'{scenario_name}/num_collisions_obst_quad_3_5'] = \
                            self.distance_to_goal_3_5

                        infos[i]['episode_extra_stats']['num_collisions_obst_quad_5'] = \
                            self.distance_to_goal_5
                        infos[i]['episode_extra_stats'][f'{scenario_name}/num_collisions_obst_quad_5'] = \
                            self.distance_to_goal_5

            if not self.saved_in_replay_buffer:
                # agent_success_rate: base_success_rate, based on per agent
                # 0: collision; 1: no collision
                agent_col_flag_list = np.logical_and(self.agent_col_agent, self.agent_col_obst)
                agent_success_flag_list = np.logical_and(agent_col_flag_list, self.reached_goal)
                agent_success_ratio = 1.0 * np.sum(agent_success_flag_list) / self.num_agents

                # agent_deadlock_rate
                # Doesn't approach to the goal while no collisions with other objects
                agent_deadlock_list = np.logical_and(agent_col_flag_list, 1 - self.reached_goal)
                agent_deadlock_ratio = 1.0 * np.sum(agent_deadlock_list) / self.num_agents

                # agent_col_rate
                # Collide with other drones and obstacles
                agent_col_ratio = 1.0 - np.sum(agent_col_flag_list) / self.num_agents

                # agent_neighbor_col_rate
                agent_neighbor_col_ratio = 1.0 - np.sum(self.agent_col_agent) / self.num_agents
                # agent_obst_col_rate
                agent_obst_col_ratio = 1.0 - np.sum(self.agent_col_obst) / self.num_agents

                for i in range(len(infos)):
                    # agent_success_rate
                    infos[i]['episode_extra_stats']['metric/agent_success_rate'] = agent_success_ratio
                    infos[i]['episode_extra_stats'][f'{scenario_name}/agent_success_rate'] = agent_success_ratio
                    # agent_deadlock_rate
                    infos[i]['episode_extra_stats']['metric/agent_deadlock_rate'] = agent_deadlock_ratio
                    infos[i]['episode_extra_stats'][f'{scenario_name}/agent_deadlock_rate'] = agent_deadlock_ratio
                    # agent_col_rate
                    infos[i]['episode_extra_stats']['metric/agent_col_rate'] = agent_col_ratio
                    infos[i]['episode_extra_stats'][f'{scenario_name}/agent_col_rate'] = agent_col_ratio
                    # agent_neighbor_col_rate
                    infos[i]['episode_extra_stats']['metric/agent_neighbor_col_rate'] = agent_neighbor_col_ratio
                    infos[i]['episode_extra_stats'][f'{scenario_name}/agent_neighbor_col_rate'] = agent_neighbor_col_ratio
                    # agent_obst_col_rate
                    infos[i]['episode_extra_stats']['metric/agent_obst_col_rate'] = agent_obst_col_ratio
                    infos[i]['episode_extra_stats'][f'{scenario_name}/agent_obst_col_rate'] = agent_obst_col_ratio

            obs = self.reset()
            # terminate the episode for all "sub-envs"
            dones = [True] * len(dones)

        if self.use_goal_ball and self.goal_ball_targets_total > 0:
            if self.goal_ball_collected_per_episode >= self.goal_ball_targets_total:
                dones = [True] * len(dones)

        return obs, rewards, dones, infos

    def _prepare_scenes_for_render(self):
        models = tuple(e.dynamics.model for e in self.envs)

        if len(self.scenes) == 0:
            self.init_scene_multi()

        if self.reset_scene:
            for i in range(len(self.scenes)):
                self.scenes[i].update_models(models)
                self.scenes[i].formation_size = self.quads_formation_size
                self.scenes[i].update_env(self.room_dims)

                self.scenes[i].reset(tuple(e.goal for e in self.envs), self.all_dynamics(), self.obstacles,
                                     self.all_collisions)

            self.reset_scene = False

        if self.quads_mode == "mix":
            for i in range(len(self.scenes)):
                self.scenes[i].formation_size = self.scenario.scenario.formation_size
        else:
            for i in range(len(self.scenes)):
                self.scenes[i].formation_size = self.scenario.formation_size

    def get_drone_fpv_image(self, drone_index=None):
        if drone_index is None:
            drone_index = self.camera_drone_index

        drone_index = int(np.clip(drone_index, 0, self.num_agents - 1))
        self.camera_drone_index = drone_index

        self._prepare_scenes_for_render()

        if len(self.scenes) == 0:
            raise RuntimeError("No render scene is available for FPV camera capture")

        scene = self.scenes[0]
        scene.camera_drone_index = drone_index
        goals = tuple(e.goal for e in self.envs)
        self.latest_fpv_frame = scene.render_obs(
            all_dynamics=self.all_dynamics(), goals=goals, collisions=self.all_collisions, obstacles=self.obstacles
        )
        return self.latest_fpv_frame

    def get_drone_fpv_camera_info(self):
        camera = self._current_camera_info(self.camera_drone_index)
        return {
            "width": self.camera_hw[0],
            "height": self.camera_hw[1],
            "fov_deg": self.camera_fov,
            "pitch_deg": self.camera_pitch_deg,
            "drone_index": self.camera_drone_index,
            "eye": camera.eye.copy(),
            "center": camera.center.copy(),
            "up": camera.up.copy(),
        }

    def render(self, verbose=False):
        self._prepare_scenes_for_render()

        self.frames_since_last_render += 1

        if self.render_skip_frames > 0:
            self.render_skip_frames -= 1
            return None

        # this is to handle the 1st step of the simulation that will typically be very slow
        if self.simulation_start_time > 0:
            simulation_time = time.time() - self.simulation_start_time
        else:
            simulation_time = 0

        realtime_control_period = 1 / self.control_freq

        render_start = time.time()
        goals = tuple(e.goal for e in self.envs)
        frames = []
        first_spawn = None
        for i in range(len(self.scenes)):
            frame, first_spawn = self.scenes[i].render_chase(all_dynamics=self.all_dynamics(), goals=goals,
                                                             collisions=self.all_collisions,
                                                             mode=self.render_mode, obstacles=self.obstacles,
                                                             first_spawn=first_spawn)
            frames.append(frame)
        # Update the formation size of the scenario
        if self.quads_mode == "mix":
            for i in range(len(self.scenes)):
                self.scenario.scenario.update_formation_size(self.scenes[i].formation_size)
        else:
            for i in range(len(self.scenes)):
                self.scenario.update_formation_size(self.scenes[i].formation_size)

        render_time = time.time() - render_start

        desired_time_between_frames = realtime_control_period * self.frames_since_last_render / self.render_speed
        time_to_sleep = desired_time_between_frames - simulation_time - render_time

        # wait so we don't simulate/render faster than realtime
        if self.render_mode == "human" and time_to_sleep > 0:
            time.sleep(time_to_sleep)

        if simulation_time + render_time > desired_time_between_frames:
            self.render_every_nth_frame += 1
            if verbose:
                print(f"Last render + simulation time {render_time + simulation_time:.3f}")
                print(f"Rendering does not keep up, rendering every {self.render_every_nth_frame} frames")
        elif simulation_time + render_time < realtime_control_period * (
                self.frames_since_last_render - 1) / self.render_speed:
            self.render_every_nth_frame -= 1
            if verbose:
                print(f"We can increase rendering framerate, rendering every {self.render_every_nth_frame} frames")

        if self.render_every_nth_frame > 5:
            self.render_every_nth_frame = 5
            if self.envs[0].tick % 20 == 0:
                print(f"Rendering cannot keep up! Rendering every {self.render_every_nth_frame} frames")

        self.render_skip_frames = self.render_every_nth_frame - 1
        self.frames_since_last_render = 0

        self.simulation_start_time = time.time()

        if self.render_mode == "rgb_array":
            return frame

    def __deepcopy__(self, memo):
        """OpenGL scene can't be copied naively."""

        cls = self.__class__
        copied_env = cls.__new__(cls)
        memo[id(self)] = copied_env

        # this will actually break the reward shaping functionality in PBT, but we need to fix it in SampleFactory, not here
        skip_copying = {"scene", "reward_shaping_interface"}

        for k, v in self.__dict__.items():
            if k not in skip_copying:
                setattr(copied_env, k, deepcopy(v, memo))

        # warning! deep-copied env has its scene uninitialized! We need to reuse one from the existing env
        # to avoid creating tons of windows
        copied_env.scene = None

        return copied_env
