from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from swarm_rl.vision.demo_utils import REPO_ROOT, build_vision_parser, heuristic_goal_velocity, make_base_env
from swarm_rl.vision.yolo_obstacles import default_yolo_data_yaml, detections_to_yolo_labels


def ensure_dataset_dirs(output_dir: Path):
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def main():
    parser = build_vision_parser("Export simulator FPV images and oracle obstacle boxes in YOLO format.")
    parser.add_argument("--frames", default=4000, type=int)
    parser.add_argument("--val_ratio", default=0.1, type=float)
    parser.add_argument("--save_empty_prob", default=0.15, type=float)
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "data" / "yolo_obstacles_pillars"), type=str)
    parser.add_argument("--heuristic_speed_xy", default=2.2, type=float)
    parser.add_argument("--heuristic_speed_z", default=0.6, type=float)
    args = parser.parse_args()

    args.quads_yolo_source = "oracle"
    env = make_base_env(args, render_mode="rgb_array")

    output_dir = Path(args.output_dir).resolve()
    ensure_dataset_dirs(output_dir)

    saved_frames = 0
    env.reset()

    while saved_frames < args.frames:
        try:
            frame_rgb = env.get_drone_fpv_image(drone_index=args.quads_camera_drone_index)
        except Exception as exc:
            if "Cannot connect to" in str(exc):
                raise RuntimeError(
                    "FPV rendering needs a desktop/X display. Run dataset export from your local desktop session."
                ) from exc
            raise
        detections = env.get_current_obstacle_detections(
            drone_index=args.quads_camera_drone_index,
            source="oracle",
        )

        keep_frame = len(detections) > 0 or np.random.rand() < args.save_empty_prob
        if keep_frame:
            split = "val" if np.random.rand() < args.val_ratio else "train"
            stem = f"{saved_frames:06d}"
            image_path = output_dir / "images" / split / f"{stem}.jpg"
            label_path = output_dir / "labels" / split / f"{stem}.txt"

            cv2.imwrite(image_path.as_posix(), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            label_lines = detections_to_yolo_labels(
                detections=detections,
                image_shape=(args.quads_camera_height, args.quads_camera_width),
            )
            label_path.write_text("\n".join(label_lines), encoding="utf-8")
            saved_frames += 1

            if saved_frames % 200 == 0:
                print(f"Saved {saved_frames}/{args.frames} frames to {output_dir}")

        actions = [
            heuristic_goal_velocity(
                env,
                drone_index=agent_id,
                speed_xy=args.heuristic_speed_xy,
                speed_z=args.heuristic_speed_z,
            )
            for agent_id in range(env.num_agents)
        ]
        _, _, dones, _ = env.step(actions)
        if any(dones):
            env.reset()

    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text(default_yolo_data_yaml(output_dir), encoding="utf-8")
    print(f"YOLO dataset exported to: {output_dir}")
    print(f"Data config written to: {data_yaml}")


if __name__ == "__main__":
    main()
