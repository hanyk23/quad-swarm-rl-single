from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from sample_factory.utils.utils import str2bool

from swarm_rl.vision.demo_utils import REPO_ROOT, build_vision_parser, heuristic_goal_velocity, make_base_env


def main():
    parser = build_vision_parser("Render the drone FPV camera with YOLO-style obstacle detections.")
    parser.add_argument("--episodes", default=2, type=int)
    parser.add_argument("--max_steps", default=300, type=int)
    parser.add_argument("--output_video", default=str(REPO_ROOT / "vision_outputs" / "yolo_fpv_demo.mp4"), type=str)
    parser.add_argument("--display", default=False, type=str2bool)
    parser.add_argument("--heuristic_speed_xy", default=2.2, type=float)
    parser.add_argument("--heuristic_speed_z", default=0.6, type=float)
    args = parser.parse_args()

    env = make_base_env(args, render_mode="rgb_array")
    output_video = Path(args.output_video).resolve()
    output_video.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        output_video.as_posix(),
        cv2.VideoWriter_fourcc(*"mp4v"),
        20.0,
        (int(args.quads_camera_width), int(args.quads_camera_height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_video}")

    total_frames = 0
    total_detections = 0

    try:
        for episode_idx in range(args.episodes):
            env.reset()

            for step_idx in range(args.max_steps):
                try:
                    frame_rgb = env.get_drone_fpv_image(drone_index=args.quads_camera_drone_index)
                except Exception as exc:
                    if "Cannot connect to" in str(exc):
                        raise RuntimeError(
                            "FPV rendering needs a desktop/X display. Run this command from your local desktop session."
                        ) from exc
                    raise
                detections = env.get_current_obstacle_detections(
                    drone_index=args.quads_camera_drone_index,
                    source=args.quads_yolo_source,
                )
                annotated = np.array(frame_rgb, copy=True)
                for detection in detections:
                    x1, y1, x2, y2 = detection["bbox_xyxy"].astype(int)
                    conf = float(detection.get("conf", 1.0))
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        annotated,
                        f"obstacle {conf:.2f}",
                        (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )

                cv2.putText(
                    annotated,
                    f"source={args.quads_yolo_source} episode={episode_idx} step={step_idx} detections={len(detections)}",
                    (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                writer.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
                total_frames += 1
                total_detections += len(detections)

                if args.display:
                    cv2.imshow("YOLO FPV Demo", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

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
                    break
    finally:
        writer.release()
        if args.display:
            cv2.destroyAllWindows()

    mean_detections = total_detections / max(total_frames, 1)
    print(f"Saved YOLO FPV demo video to: {output_video}")
    print(f"Frames recorded: {total_frames}")
    print(f"Average detections per frame: {mean_detections:.3f}")


if __name__ == "__main__":
    main()
