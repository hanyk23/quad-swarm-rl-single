from __future__ import annotations

from typing import Iterable

import numpy as np

from swarm_rl.vision.yolo_obstacles import CameraInfo, _camera_axes


DEPTH_GRID_WIDTH = 3
DEPTH_GRID_HEIGHT = 3
DEPTH_OBS_DIM = DEPTH_GRID_WIDTH * DEPTH_GRID_HEIGHT
LIDAR_OBS_DIM = 9


def camera_ray_directions(camera: CameraInfo, width: int, height: int) -> np.ndarray:
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError("Depth camera width and height must be positive")

    forward, right, true_up = _camera_axes(camera)
    aspect = float(camera.width) / float(camera.height)
    tan_half_fov = float(np.tan(np.radians(camera.fov_deg) / 2.0))

    cols = (np.arange(width, dtype=np.float32) + 0.5) / float(width)
    rows = (np.arange(height, dtype=np.float32) + 0.5) / float(height)
    x_ndc = 2.0 * cols - 1.0
    y_ndc = 1.0 - 2.0 * rows

    x_grid, y_grid = np.meshgrid(x_ndc, y_ndc)
    dirs = (
        forward[None, None, :]
        + (x_grid[:, :, None] * tan_half_fov * aspect) * right[None, None, :]
        + (y_grid[:, :, None] * tan_half_fov) * true_up[None, None, :]
    ).astype(np.float32)
    norms = np.linalg.norm(dirs, axis=2, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return dirs / norms


def raycast_depth_map(
    obstacle_positions: Iterable[np.ndarray],
    obstacle_size: float,
    room_height: float,
    camera: CameraInfo,
    width: int,
    height: int,
    max_distance: float = 10.0,
    min_distance: float = 0.05,
) -> np.ndarray:
    """Raycast vertical cylindrical pillars and return metric depth in meters."""
    width = int(width)
    height = int(height)
    max_distance = float(max_distance)
    min_distance = float(min_distance)

    depth = np.full((height, width), max_distance, dtype=np.float32)
    obstacle_positions = [np.asarray(pos, dtype=np.float32) for pos in obstacle_positions]
    if len(obstacle_positions) == 0:
        return depth

    dirs = camera_ray_directions(camera, width=width, height=height)
    origin = np.asarray(camera.eye, dtype=np.float32)
    radius = float(obstacle_size) / 2.0
    z_min = 0.0
    z_max = float(room_height)

    dx = dirs[:, :, 0]
    dy = dirs[:, :, 1]
    dz = dirs[:, :, 2]
    a = dx * dx + dy * dy
    valid_a = a > 1e-8

    for obstacle in obstacle_positions:
        ox = float(origin[0] - obstacle[0])
        oy = float(origin[1] - obstacle[1])
        b = 2.0 * (ox * dx + oy * dy)
        c = ox * ox + oy * oy - radius * radius
        disc = b * b - 4.0 * a * c

        valid_disc = np.logical_and(valid_a, disc >= 0.0)
        if not np.any(valid_disc):
            continue

        sqrt_disc = np.zeros_like(depth)
        sqrt_disc[valid_disc] = np.sqrt(disc[valid_disc])

        for t_candidate in ((-b - sqrt_disc) / (2.0 * a + 1e-12), (-b + sqrt_disc) / (2.0 * a + 1e-12)):
            z_hit = origin[2] + t_candidate * dz
            valid_hit = np.logical_and.reduce(
                (
                    valid_disc,
                    t_candidate >= min_distance,
                    t_candidate <= max_distance,
                    z_hit >= z_min,
                    z_hit <= z_max,
                )
            )
            depth = np.where(valid_hit, np.minimum(depth, t_candidate), depth)

    return np.clip(depth, min_distance, max_distance).astype(np.float32)


def encode_depth_observation(
    obstacle_positions: Iterable[np.ndarray],
    obstacle_size: float,
    room_height: float,
    camera: CameraInfo,
    grid_width: int = DEPTH_GRID_WIDTH,
    grid_height: int = DEPTH_GRID_HEIGHT,
    max_distance: float = 10.0,
    min_distance: float = 0.05,
    normalize: bool = False,
    noise_std: float = 0.0,
    dropout_prob: float = 0.0,
    rng=None,
) -> np.ndarray:
    depth = raycast_depth_map(
        obstacle_positions=obstacle_positions,
        obstacle_size=obstacle_size,
        room_height=room_height,
        camera=camera,
        width=grid_width,
        height=grid_height,
        max_distance=max_distance,
        min_distance=min_distance,
    )

    if rng is None:
        rng = np.random

    if noise_std > 0.0:
        depth = depth + rng.normal(0.0, float(noise_std), size=depth.shape).astype(np.float32)

    if dropout_prob > 0.0:
        dropout = rng.random(size=depth.shape) < float(dropout_prob)
        depth = np.where(dropout, max_distance, depth)

    depth = np.clip(depth, min_distance, max_distance)

    if normalize:
        denom = max(max_distance - min_distance, 1e-6)
        depth = (depth - min_distance) / denom

    return depth.reshape(-1).astype(np.float32)


def encode_lidar_observation(
    obstacle_positions: Iterable[np.ndarray],
    obstacle_size: float,
    room_dims,
    origin: np.ndarray,
    yaw: float = 0.0,
    num_rays: int = LIDAR_OBS_DIM,
    max_distance: float = 10.0,
    min_distance: float = 0.05,
    normalize: bool = False,
    noise_std: float = 0.0,
    dropout_prob: float = 0.0,
    rng=None,
) -> np.ndarray:
    """Return metric 360-degree horizontal lidar ranges around the drone."""
    num_rays = int(num_rays)
    max_distance = float(max_distance)
    min_distance = float(min_distance)
    origin = np.asarray(origin, dtype=np.float32)
    room_dims = np.asarray(room_dims, dtype=np.float32)
    obstacle_positions = [np.asarray(pos, dtype=np.float32) for pos in obstacle_positions]

    angles = float(yaw) + np.linspace(0.0, 2.0 * np.pi, num_rays, endpoint=False, dtype=np.float32)
    dirs = np.stack((np.cos(angles), np.sin(angles)), axis=1).astype(np.float32)
    ranges = np.full(num_rays, max_distance, dtype=np.float32)
    radius = float(obstacle_size) / 2.0
    origin_xy = origin[:2].astype(np.float32)

    for obstacle in obstacle_positions:
        rel = origin_xy - obstacle[:2]
        b = 2.0 * (rel[0] * dirs[:, 0] + rel[1] * dirs[:, 1])
        c = rel[0] * rel[0] + rel[1] * rel[1] - radius * radius
        disc = b * b - 4.0 * c
        valid = disc >= 0.0
        if not np.any(valid):
            continue
        sqrt_disc = np.zeros_like(ranges)
        sqrt_disc[valid] = np.sqrt(disc[valid])
        for t_candidate in ((-b - sqrt_disc) / 2.0, (-b + sqrt_disc) / 2.0):
            hit = np.logical_and.reduce((
                valid,
                t_candidate >= min_distance,
                t_candidate <= max_distance,
            ))
            ranges = np.where(hit, np.minimum(ranges, t_candidate), ranges)

    half_x = float(room_dims[0]) / 2.0
    half_y = float(room_dims[1]) / 2.0
    ox, oy = float(origin_xy[0]), float(origin_xy[1])
    dx = dirs[:, 0]
    dy = dirs[:, 1]

    wall_candidates = []
    eps = 1e-8
    wall_candidates.append(np.where(dx > eps, (half_x - ox) / dx, max_distance))
    wall_candidates.append(np.where(dx < -eps, (-half_x - ox) / dx, max_distance))
    wall_candidates.append(np.where(dy > eps, (half_y - oy) / dy, max_distance))
    wall_candidates.append(np.where(dy < -eps, (-half_y - oy) / dy, max_distance))
    for candidate in wall_candidates:
        hit = np.logical_and(candidate >= min_distance, candidate <= max_distance)
        ranges = np.where(hit, np.minimum(ranges, candidate), ranges)

    if rng is None:
        rng = np.random
    if noise_std > 0.0:
        ranges = ranges + rng.normal(0.0, float(noise_std), size=ranges.shape).astype(np.float32)
    if dropout_prob > 0.0:
        dropout = rng.random(size=ranges.shape) < float(dropout_prob)
        ranges = np.where(dropout, max_distance, ranges)

    ranges = np.clip(ranges, min_distance, max_distance)
    if normalize:
        denom = max(max_distance - min_distance, 1e-6)
        ranges = (ranges - min_distance) / denom
    return ranges.astype(np.float32)


def colorize_depth(depth: np.ndarray, max_distance: float = 10.0, min_distance: float = 0.05) -> np.ndarray:
    import cv2

    depth = np.asarray(depth, dtype=np.float32)
    denom = max(float(max_distance) - float(min_distance), 1e-6)
    normalized = np.clip((depth - float(min_distance)) / denom, 0.0, 1.0)
    inv = (255.0 * (1.0 - normalized)).astype(np.uint8)
    color_map = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
    return cv2.applyColorMap(inv, color_map)
