from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

YOLO_MAX_DETECTIONS = 15
YOLO_FEATURES_PER_DETECTION = 6
YOLO_OBS_DIM = YOLO_MAX_DETECTIONS * YOLO_FEATURES_PER_DETECTION


@dataclass
class CameraInfo:
    width: int
    height: int
    fov_deg: float
    eye: np.ndarray
    center: np.ndarray
    up: np.ndarray


def _normalize(vec: np.ndarray) -> tuple[np.ndarray, float]:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return np.array(vec, copy=True), 0.0
    return vec / norm, norm


def _camera_axes(camera: CameraInfo) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward, _ = _normalize(camera.center - camera.eye)
    right, _ = _normalize(np.cross(forward, camera.up))
    true_up, _ = _normalize(np.cross(right, forward))
    return forward, right, true_up


def project_world_point(
    point_world: np.ndarray,
    camera: CameraInfo,
    clip_margin_ndc: Optional[float] = 6.0,
) -> Optional[np.ndarray]:
    forward, right, true_up = _camera_axes(camera)
    rel = point_world - camera.eye
    depth = float(np.dot(rel, forward))

    if depth <= 1e-4:
        return None

    x_cam = float(np.dot(rel, right))
    y_cam = float(np.dot(rel, true_up))

    aspect = float(camera.width) / float(camera.height)
    tan_half_fov = np.tan(np.radians(camera.fov_deg) / 2.0)

    if tan_half_fov <= 0.0:
        return None

    x_ndc = x_cam / (depth * tan_half_fov * aspect)
    y_ndc = y_cam / (depth * tan_half_fov)

    if clip_margin_ndc is not None:
        if abs(x_ndc) > clip_margin_ndc or abs(y_ndc) > clip_margin_ndc:
            return None

    x_px = 0.5 * (x_ndc + 1.0) * camera.width
    y_px = 0.5 * (1.0 - y_ndc) * camera.height

    return np.array([x_px, y_px, depth], dtype=np.float32)


def _sample_cylinder_points(
    center_xy: np.ndarray,
    radius: float,
    room_height: float,
) -> np.ndarray:
    """
    Lightweight cylinder sampling for training.
    Enough to make bbox cover the pillar, but not too slow.
    """
    angle_samples = np.linspace(
        0.0,
        2.0 * np.pi,
        num=16,
        endpoint=False,
        dtype=np.float32,
    )

    z_samples = np.linspace(
        0.0,
        float(room_height),
        num=5,
        dtype=np.float32,
    )

    points = []

    for z in z_samples:
        for angle in angle_samples:
            x = float(center_xy[0]) + float(radius) * float(np.cos(angle))
            y = float(center_xy[1]) + float(radius) * float(np.sin(angle))
            points.append([x, y, float(z)])

    points.append([float(center_xy[0]), float(center_xy[1]), 0.0])
    points.append([float(center_xy[0]), float(center_xy[1]), float(room_height)])

    return np.asarray(points, dtype=np.float32)


def _clip_bbox_to_image(
    x1_raw: float,
    y1_raw: float,
    x2_raw: float,
    y2_raw: float,
    width: int,
    height: int,
    padding_ratio: float = 0.12,
) -> Optional[np.ndarray]:
    """
    Expand bbox slightly and clip to image.
    """
    if x2_raw < 0.0 or y2_raw < 0.0:
        return None
    if x1_raw > width - 1.0 or y1_raw > height - 1.0:
        return None

    box_w = max(0.0, x2_raw - x1_raw)
    box_h = max(0.0, y2_raw - y1_raw)

    pad_x = padding_ratio * box_w
    pad_y = padding_ratio * box_h

    x1 = float(np.clip(x1_raw - pad_x, 0.0, width - 1.0))
    y1 = float(np.clip(y1_raw - pad_y, 0.0, height - 1.0))
    x2 = float(np.clip(x2_raw + pad_x, 0.0, width - 1.0))
    y2 = float(np.clip(y2_raw + pad_y, 0.0, height - 1.0))

    if x2 - x1 < 2.0 or y2 - y1 < 2.0:
        return None

    return np.array([x1, y1, x2, y2], dtype=np.float32)


def oracle_obstacle_detections(
    obstacle_positions: Iterable[np.ndarray],
    obstacle_size: float,
    room_height: float,
    camera: CameraInfo,
    max_detections: int = YOLO_MAX_DETECTIONS,
) -> List[dict]:
    detections: List[dict] = []

    radius = float(obstacle_size) / 2.0
    width = int(camera.width)
    height = int(camera.height)

    for obstacle in obstacle_positions:
        obstacle = np.asarray(obstacle, dtype=np.float32)
        center_xy = obstacle[:2]

        points_world = _sample_cylinder_points(
            center_xy=center_xy,
            radius=radius,
            room_height=float(room_height),
        )

        projected = [
            project_world_point(point, camera, clip_margin_ndc=6.0)
            for point in points_world
        ]
        projected = [point for point in projected if point is not None]

        # Too few projected points usually means the obstacle is behind camera
        # or barely visible. Ignore it to avoid tiny unstable boxes.
        if len(projected) < 6:
            continue

        projected = np.stack(projected, axis=0)

        x1_raw = float(np.min(projected[:, 0]))
        y1_raw = float(np.min(projected[:, 1]))
        x2_raw = float(np.max(projected[:, 0]))
        y2_raw = float(np.max(projected[:, 1]))

        bbox = _clip_bbox_to_image(
            x1_raw=x1_raw,
            y1_raw=y1_raw,
            x2_raw=x2_raw,
            y2_raw=y2_raw,
            width=width,
            height=height,
            padding_ratio=0.12,
        )

        if bbox is None:
            continue

        x1, y1, x2, y2 = bbox
        area_norm = ((x2 - x1) * (y2 - y1)) / float(max(width * height, 1))

        detections.append(
            {
                "bbox_xyxy": bbox,
                "conf": 1.0,
                "class_id": 0,
                "label": "obstacle",
                "area_norm": float(area_norm),
            }
        )

    detections.sort(key=lambda item: item["area_norm"], reverse=True)
    return detections[:max_detections]


def encode_obstacle_detections(
    detections: Iterable[dict],
    image_shape: tuple[int, int] | list[int],
    max_detections: int = YOLO_MAX_DETECTIONS,
) -> np.ndarray:
    height, width = int(image_shape[0]), int(image_shape[1])
    encoded = np.zeros(YOLO_OBS_DIM, dtype=np.float32)
    detections = list(detections)[:max_detections]

    for index, detection in enumerate(detections):
        x1, y1, x2, y2 = detection["bbox_xyxy"]
        cx = ((x1 + x2) / 2.0) / max(width, 1)
        cy = ((y1 + y2) / 2.0) / max(height, 1)
        w = (x2 - x1) / max(width, 1)
        h = (y2 - y1) / max(height, 1)
        conf = float(detection.get("conf", 1.0))
        area = w * h

        start = index * YOLO_FEATURES_PER_DETECTION
        encoded[start:start + YOLO_FEATURES_PER_DETECTION] = np.array(
            [cx, cy, w, h, conf, area], dtype=np.float32
        )

    return encoded


def detections_to_yolo_labels(
    detections: Iterable[dict],
    image_shape: tuple[int, int] | list[int],
    class_id: int = 0,
) -> List[str]:
    height, width = int(image_shape[0]), int(image_shape[1])
    labels = []

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox_xyxy"]
        cx = ((x1 + x2) / 2.0) / max(width, 1)
        cy = ((y1 + y2) / 2.0) / max(height, 1)
        w = (x2 - x1) / max(width, 1)
        h = (y2 - y1) / max(height, 1)
        labels.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return labels


def annotate_detections(image_rgb: np.ndarray, detections: Iterable[dict]) -> np.ndarray:
    import cv2

    annotated = np.array(image_rgb, copy=True)

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox_xyxy"].astype(int)
        conf = float(detection.get("conf", 1.0))
        label = detection.get("label", "obstacle")

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"{label}:{conf:.2f}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return annotated


def mask_obstacle_detections(
    image_rgb: np.ndarray,
    max_detections: int = YOLO_MAX_DETECTIONS,
    min_area_px: int = 40,
) -> List[dict]:
    import cv2

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    lower_green = np.array([35, 35, 25], dtype=np.uint8)
    upper_green = np.array([90, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections: List[dict] = []
    height, width = image_rgb.shape[:2]

    for contour in contours:
        area_px = float(cv2.contourArea(contour))
        if area_px < float(min_area_px):
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w < 2 or h < 2:
            continue

        pad_x = 0.08 * w
        pad_y = 0.08 * h

        x1 = float(np.clip(x - pad_x, 0, width - 1))
        y1 = float(np.clip(y - pad_y, 0, height - 1))
        x2 = float(np.clip(x + w + pad_x, 0, width - 1))
        y2 = float(np.clip(y + h + pad_y, 0, height - 1))

        area_norm = ((x2 - x1) * (y2 - y1)) / float(max(width * height, 1))

        detections.append(
            {
                "bbox_xyxy": np.array([x1, y1, x2, y2], dtype=np.float32),
                "conf": 1.0,
                "class_id": 0,
                "label": "obstacle",
                "area_norm": float(area_norm),
            }
        )

    detections.sort(key=lambda item: item["area_norm"], reverse=True)
    return detections[:max_detections]


class UltralyticsObstacleDetector:
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        max_detections: int = YOLO_MAX_DETECTIONS,
        class_id: Optional[int] = 0,
    ):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is not installed. Run `python -m pip install ultralytics` "
                "in your swarm-rl env first."
            ) from exc

        self.model_path = str(model_path)
        self.conf_threshold = float(conf_threshold)
        self.max_detections = int(max_detections)
        self.class_id = class_id
        self.model = YOLO(self.model_path)

    def detect(self, image_rgb: np.ndarray) -> List[dict]:
        results = self.model.predict(
            source=image_rgb,
            conf=self.conf_threshold,
            verbose=False,
            max_det=self.max_detections,
        )

        detections: List[dict] = []

        if len(results) == 0:
            return detections

        boxes = results[0].boxes

        if boxes is None:
            return detections

        for box in boxes:
            class_id = int(box.cls.item()) if box.cls is not None else 0

            if self.class_id is not None and class_id != self.class_id:
                continue

            xyxy = box.xyxy[0].detach().cpu().numpy().astype(np.float32)
            conf = float(box.conf.item()) if box.conf is not None else 0.0

            detections.append(
                {
                    "bbox_xyxy": xyxy,
                    "conf": conf,
                    "class_id": class_id,
                    "label": "obstacle",
                    "area_norm": float(
                        max(0.0, (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]))
                        / max(1.0, image_rgb.shape[0] * image_rgb.shape[1])
                    ),
                }
            )

        detections.sort(key=lambda item: item["conf"], reverse=True)
        return detections[: self.max_detections]


def default_yolo_data_yaml(dataset_dir: Path) -> str:
    dataset_dir = Path(dataset_dir).resolve()

    yaml_text = (
        f"path: {dataset_dir.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: obstacle\n"
    )

    return yaml_text