from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train an obstacle detector with Ultralytics YOLO.")
    parser.add_argument("--data", required=True, type=str, help="Path to YOLO data.yaml")
    parser.add_argument("--model", default="yolov8n.pt", type=str, help="Base YOLO checkpoint or model yaml")
    parser.add_argument("--epochs", default=60, type=int)
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument("--batch", default=16, type=int)
    parser.add_argument("--project", default="train_dir_yolo", type=str)
    parser.add_argument("--name", default="pillar_detector_v1", type=str)
    parser.add_argument("--device", default="", type=str, help='Examples: "", "cpu", "0"')
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics is not installed. Run `python -m pip install ultralytics` in the swarm-rl environment first."
        ) from exc

    data_yaml = Path(args.data).resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(f"YOLO data config not found: {data_yaml}")

    model = YOLO(args.model)
    results = model.train(
        data=data_yaml.as_posix(),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=args.device,
        verbose=True,
    )

    save_dir = getattr(results, "save_dir", None)
    if save_dir is None and hasattr(model, "trainer"):
        save_dir = getattr(model.trainer, "save_dir", None)

    if save_dir is not None:
        print(f"Training finished. Outputs are under: {save_dir}")
    else:
        print("Training finished.")


if __name__ == "__main__":
    main()
