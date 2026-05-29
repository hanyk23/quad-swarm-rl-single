import copy
import numpy as np

from gym_art.quadrotor_multi.obstacles.utils import get_surround_sdfs, get_surround_sdfs_with_bounds, collision_detection


class MultiObstacles:
    def __init__(self, obstacle_size=1.0, quad_radius=0.046, resolution=0.1,
                 room_dims=None, include_room_bounds=False):
        self.size = obstacle_size
        self.obstacle_radius = obstacle_size / 2.0
        self.quad_radius = quad_radius
        self.pos_arr = []
        self.resolution = resolution
        self.room_dims = np.array(room_dims[:2], dtype=np.float64) if room_dims is not None else np.zeros(2)
        self.include_room_bounds = include_room_bounds

    def _empty_sdf_obs(self, quads_pos):
        return 100 * np.ones((len(quads_pos), 9))

    def reset(self, obs, quads_pos, pos_arr):
        self.pos_arr = copy.deepcopy(np.array(pos_arr))

        quads_sdf_obs = self._empty_sdf_obs(quads_pos)
        if self.include_room_bounds:
            obst_poses = self.pos_arr[:, :2] if self.pos_arr.size > 0 else np.zeros((0, 2))
            quads_sdf_obs = get_surround_sdfs_with_bounds(
                quad_poses=quads_pos[:, :2], obst_poses=obst_poses, quads_sdf_obs=quads_sdf_obs,
                obst_radius=self.obstacle_radius, resolution=self.resolution, room_dims=self.room_dims)
        elif self.pos_arr.size > 0:
            quads_sdf_obs = get_surround_sdfs(quad_poses=quads_pos[:, :2], obst_poses=self.pos_arr[:, :2],
                                              quads_sdf_obs=quads_sdf_obs, obst_radius=self.obstacle_radius,
                                              resolution=self.resolution)

        obs = np.concatenate((obs, quads_sdf_obs), axis=1)

        return obs

    def step(self, obs, quads_pos):
        quads_sdf_obs = self._empty_sdf_obs(quads_pos)
        if self.include_room_bounds:
            obst_poses = self.pos_arr[:, :2] if self.pos_arr.size > 0 else np.zeros((0, 2))
            quads_sdf_obs = get_surround_sdfs_with_bounds(
                quad_poses=quads_pos[:, :2], obst_poses=obst_poses, quads_sdf_obs=quads_sdf_obs,
                obst_radius=self.obstacle_radius, resolution=self.resolution, room_dims=self.room_dims)
        elif self.pos_arr.size > 0:
            quads_sdf_obs = get_surround_sdfs(quad_poses=quads_pos[:, :2], obst_poses=self.pos_arr[:, :2],
                                              quads_sdf_obs=quads_sdf_obs, obst_radius=self.obstacle_radius,
                                              resolution=self.resolution)

        obs = np.concatenate((obs, quads_sdf_obs), axis=1)

        return obs

    def collision_detection(self, pos_quads):
        if self.pos_arr.size == 0:
            return np.array([], dtype=np.int64), {}

        quad_collisions = collision_detection(quad_poses=pos_quads[:, :2], obst_poses=self.pos_arr[:, :2],
                                              obst_radius=self.obstacle_radius, quad_radius=self.quad_radius)

        collided_quads_id = np.where(quad_collisions > -1)[0]
        collided_obstacles_id = quad_collisions[collided_quads_id]
        quad_obst_pair = {}
        for i, key in enumerate(collided_quads_id):
            quad_obst_pair[key] = int(collided_obstacles_id[i])

        return collided_quads_id, quad_obst_pair
