import numpy as np
from numba import njit


def sample_spaced_obstacle_indices(candidate_positions, num_obstacles, min_center_distance, max_attempts=200):
    candidate_positions = np.asarray(candidate_positions, dtype=np.float64)
    num_candidates = len(candidate_positions)
    if num_obstacles < 0 or num_obstacles > num_candidates:
        raise ValueError(f"Cannot sample {num_obstacles} obstacles from {num_candidates} candidate positions")
    if num_obstacles == 0:
        return np.array([], dtype=np.int64)
    if min_center_distance <= 0.0:
        return np.random.choice(num_candidates, size=num_obstacles, replace=False)

    min_distance_sq = float(min_center_distance) ** 2
    best = []
    for _ in range(max_attempts):
        selected = []
        for candidate_idx in np.random.permutation(num_candidates):
            candidate = candidate_positions[candidate_idx]
            if all(np.sum((candidate - candidate_positions[idx]) ** 2) >= min_distance_sq for idx in selected):
                selected.append(int(candidate_idx))
                if len(selected) == num_obstacles:
                    return np.asarray(selected, dtype=np.int64)
        if len(selected) > len(best):
            best = selected

    raise ValueError(
        f"Could only place {len(best)} of {num_obstacles} obstacles with "
        f"minimum center distance {min_center_distance:.3f}"
    )


@njit
def get_surround_sdfs(quad_poses, obst_poses, quads_sdf_obs, obst_radius, resolution=0.1):
    # Shape of quads_sdf_obs: (quad_num, 9)

    sdf_map = np.array([-1., -1., -1., 0., 0., 0., 1., 1., 1.])
    sdf_map *= resolution

    for i, q_pos in enumerate(quad_poses):
        q_pos_x, q_pos_y = q_pos[0], q_pos[1]

        for g_i, g_x in enumerate([q_pos_x - resolution, q_pos_x, q_pos_x + resolution]):
            for g_j, g_y in enumerate([q_pos_y - resolution, q_pos_y, q_pos_y + resolution]):
                grid_pos = np.array([g_x, g_y])

                min_dist = 100.0
                for o_pos in obst_poses:
                    dist = np.linalg.norm(grid_pos - o_pos) - obst_radius
                    if dist < min_dist:
                        min_dist = dist

                g_id = g_i * 3 + g_j
                quads_sdf_obs[i, g_id] = min_dist

    return quads_sdf_obs


@njit
def get_surround_sdfs_with_bounds(quad_poses, obst_poses, quads_sdf_obs, obst_radius, resolution, room_dims):
    # Shape of quads_sdf_obs: (quad_num, 9)

    for i, q_pos in enumerate(quad_poses):
        q_pos_x, q_pos_y = q_pos[0], q_pos[1]

        for g_i, g_x in enumerate([q_pos_x - resolution, q_pos_x, q_pos_x + resolution]):
            for g_j, g_y in enumerate([q_pos_y - resolution, q_pos_y, q_pos_y + resolution]):
                grid_pos = np.array([g_x, g_y])

                min_dist = 100.0
                for o_pos in obst_poses:
                    dist = np.linalg.norm(grid_pos - o_pos) - obst_radius
                    if dist < min_dist:
                        min_dist = dist

                half_length = room_dims[0] / 2.0
                half_width = room_dims[1] / 2.0
                wall_dist = min(
                    grid_pos[0] + half_length,
                    half_length - grid_pos[0],
                    grid_pos[1] + half_width,
                    half_width - grid_pos[1],
                )
                if wall_dist < min_dist:
                    min_dist = wall_dist

                g_id = g_i * 3 + g_j
                quads_sdf_obs[i, g_id] = min_dist

    return quads_sdf_obs


@njit
def get_lidar_rays_with_bounds(quad_poses, obst_poses, lidar_obs, obst_radius, room_dims, ray_rotations):
    # Shape of lidar_obs: (quad_num, 9). First 8 entries are ray hit distances;
    # the last entry is local clearance at the quad position. Ray directions are
    # defined in the quad body-yaw frame and rotated into world XY here.
    ray_dirs = np.array([
        [1.0, 0.0],
        [0.70710678, 0.70710678],
        [0.0, 1.0],
        [-0.70710678, 0.70710678],
        [-1.0, 0.0],
        [-0.70710678, -0.70710678],
        [0.0, -1.0],
        [0.70710678, -0.70710678],
    ])
    half_length = room_dims[0] / 2.0
    half_width = room_dims[1] / 2.0

    for i, q_pos in enumerate(quad_poses):
        q_pos_x, q_pos_y = q_pos[0], q_pos[1]

        clearance = min(
            q_pos_x + half_length,
            half_length - q_pos_x,
            q_pos_y + half_width,
            half_width - q_pos_y,
        )
        for o_pos in obst_poses:
            dist = np.linalg.norm(q_pos - o_pos) - obst_radius
            if dist < clearance:
                clearance = dist

        for r_i in range(8):
            body_x, body_y = ray_dirs[r_i, 0], ray_dirs[r_i, 1]
            d_x = ray_rotations[i, 0, 0] * body_x + ray_rotations[i, 0, 1] * body_y
            d_y = ray_rotations[i, 1, 0] * body_x + ray_rotations[i, 1, 1] * body_y
            ray_dist = 100.0

            if d_x > 1e-8:
                wall_t = (half_length - q_pos_x) / d_x
                if wall_t >= 0.0 and wall_t < ray_dist:
                    ray_dist = wall_t
            elif d_x < -1e-8:
                wall_t = (-half_length - q_pos_x) / d_x
                if wall_t >= 0.0 and wall_t < ray_dist:
                    ray_dist = wall_t

            if d_y > 1e-8:
                wall_t = (half_width - q_pos_y) / d_y
                if wall_t >= 0.0 and wall_t < ray_dist:
                    ray_dist = wall_t
            elif d_y < -1e-8:
                wall_t = (-half_width - q_pos_y) / d_y
                if wall_t >= 0.0 and wall_t < ray_dist:
                    ray_dist = wall_t

            for o_pos in obst_poses:
                rel_x = q_pos_x - o_pos[0]
                rel_y = q_pos_y - o_pos[1]
                b = rel_x * d_x + rel_y * d_y
                c = rel_x * rel_x + rel_y * rel_y - obst_radius * obst_radius
                disc = b * b - c
                if c <= 0.0:
                    hit_t = 0.0
                elif disc >= 0.0:
                    hit_t = -b - np.sqrt(disc)
                else:
                    hit_t = -1.0

                if hit_t >= 0.0 and hit_t < ray_dist:
                    ray_dist = hit_t

            lidar_obs[i, r_i] = ray_dist

        lidar_obs[i, 8] = clearance

    return lidar_obs


@njit
def collision_detection(quad_poses, obst_poses, obst_radius, quad_radius):
    quad_num = len(quad_poses)
    collide_threshold = quad_radius + obst_radius
    # Get distance matrix b/w quad and obst
    quad_collisions = -1 * np.ones(quad_num)
    for i, q_pos in enumerate(quad_poses):
        for j, o_pos in enumerate(obst_poses):
            dist = np.linalg.norm(q_pos - o_pos)
            if dist <= collide_threshold:
                quad_collisions[i] = j
                break

    return quad_collisions


@njit
def get_cell_centers(obst_area_length, obst_area_width, grid_size=1.):
    count = 0
    i_len = obst_area_length / grid_size
    j_len = obst_area_width / grid_size
    cell_centers = np.zeros((int(i_len * j_len), 2))
    for i in np.arange(0, obst_area_length, grid_size):
        for j in np.arange(obst_area_width - grid_size, -grid_size, -grid_size):
            cell_centers[count][0] = i + (grid_size / 2) - obst_area_length // 2
            cell_centers[count][1] = j + (grid_size / 2) - obst_area_width // 2
            count += 1

    return cell_centers


if __name__ == "__main__":
    from gym_art.quadrotor_multi.obstacles.test.unit_test import unit_test

    unit_test()
