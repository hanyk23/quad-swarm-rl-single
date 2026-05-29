import numpy as np

from gym_art.quadrotor_multi.quad_utils import EPS
def obstacle_collision_normal(pos, vel, obstacle_pos, prev_pos=None):
    if prev_pos is not None:
        segment_xy = pos[:2] - prev_pos[:2]
        segment_len_sq = float(np.dot(segment_xy, segment_xy))
        if segment_len_sq > EPS:
            obstacle_xy = obstacle_pos[:2]
            t = float(np.dot(obstacle_xy - prev_pos[:2], segment_xy) / segment_len_sq)
            t = np.clip(t, 0.0, 1.0)
            closest_xy = prev_pos[:2] + t * segment_xy
            collision_vec_xy = closest_xy - obstacle_xy
            coll_norm_mag = np.linalg.norm(collision_vec_xy)
            if coll_norm_mag > EPS:
                return collision_vec_xy / coll_norm_mag, coll_norm_mag

    collision_vec_xy = pos[:2] - obstacle_pos[:2]
    coll_norm_mag = np.linalg.norm(collision_vec_xy)

    if coll_norm_mag > EPS:
        return collision_vec_xy / coll_norm_mag, coll_norm_mag

    vel_xy = vel[:2]
    vel_xy_mag = np.linalg.norm(vel_xy)
    if vel_xy_mag > EPS:
        return -vel_xy / vel_xy_mag, 0.0

    return np.array([1.0, 0.0]), 0.0


def perform_collision_with_obstacle(drone_dyn, obstacle_pos, obstacle_size, quad_radius, prev_pos=None):
    obstacle_radius = obstacle_size / 2.0
    collision_radius = obstacle_radius + quad_radius

    collision_norm_xy, dist_xy = obstacle_collision_normal(
        pos=drone_dyn.pos, vel=drone_dyn.vel, obstacle_pos=obstacle_pos, prev_pos=prev_pos
    )
    penetration = collision_radius - dist_xy

    if penetration >= -EPS:
        # Move the quadrotor back outside the pillar, with a tiny slop to avoid sticking on the next step.
        separation_slop = max(1e-3, 0.05 * quad_radius)
        drone_dyn.pos[:2] += collision_norm_xy * (max(penetration, 0.0) + separation_slop)

    vel_xy = np.array(drone_dyn.vel[:2], copy=True)
    normal_speed = float(np.dot(vel_xy, collision_norm_xy))
    tangential_vel = vel_xy - normal_speed * collision_norm_xy

    # Physically motivated response:
    # - reverse the inward normal component with restitution
    # - damp the tangential slip with a friction-like coefficient
    # - avoid any random angular kick so the vehicle does not spin off chaotically
    restitution = 0.25
    tangential_damping = 0.85

    if normal_speed < 0.0:
        normal_after = -restitution * normal_speed
    else:
        normal_after = normal_speed

    tangential_after = tangential_damping * tangential_vel
    drone_dyn.vel[:2] = normal_after * collision_norm_xy + tangential_after

    # Mild angular damping from impact dissipation.
    drone_dyn.omega *= 0.9
