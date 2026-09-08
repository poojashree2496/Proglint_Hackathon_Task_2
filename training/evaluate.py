"""
Evaluate a trained model on the validation and test splits: precision,
recall, mAP50, mAP50-95, confusion matrix, PR/F1 curves (all produced by
Ultralytics' own validator, which already generates these plots — this
script just runs it against both splits and prints a clean summary table),
plus latency/FPS timing measured independently of Ultralytics' benchmarking
utilities so it reflects this project's actual single-image inference path.

Usage:
    python training/evaluate.py --weights training/runs/surveillance_human_detector/weights/best.pt
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from scripts.common import TRAINING_ROOT, load_config, resolve_device


def measure_latency_fps(model: YOLO, imgsz: int, device: str, n_warmup=10, n_measure=50) -> dict:
    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    for _ in range(n_warmup):
        model.predict(dummy, device=device, verbose=False, imgsz=imgsz)

    latencies = []
    for _ in range(n_measure):
        start = time.perf_counter()
        model.predict(dummy, device=device, verbose=False, imgsz=imgsz)
        latencies.append(time.perf_counter() - start)

    latencies = np.array(latencies)
    return {
        "mean_latency_ms": float(latencies.mean() * 1000),
        "p95_latency_ms": float(np.percentile(latencies, 95) * 1000),
        "fps": float(1.0 / latencies.mean()),
    }


def evaluate_split(model: YOLO, data_yaml: Path, split: str, imgsz: int, conf: float, iou: float, device: str):
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        plots=True,
        save_json=True,
        project=str(TRAINING_ROOT / "runs" / "eval"),
        name=split,
        exist_ok=True,
    )

    box = metrics.box
    return {
        "split": split,
        "precision": float(box.mp),
        "recall": float(box.mr),
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "f1": float(2 * box.mp * box.mr / max(1e-9, box.mp + box.mr)),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate the custom human detector.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(config["project"]["device"])
    inf_cfg = config["inference"]

    data_yaml = args.data or (TRAINING_ROOT / "data" / "final" / "data.yaml")
    model = YOLO(str(args.weights))

    print(f"Weights: {args.weights}")
    print(f"Data:    {data_yaml}")
    print(f"Device:  {device}\n")

    results = []
    for split in ("val", "test"):
        print(f"--- Evaluating split: {split} ---")
        results.append(evaluate_split(
            model, data_yaml, split,
            imgsz=config["model"]["imgsz"],
            conf=inf_cfg["conf_threshold"],
            iou=inf_cfg["iou_threshold"],
            device=device,
        ))

    print("\n--- Latency / FPS (single-image inference) ---")
    timing = measure_latency_fps(model, config["model"]["imgsz"], device)

    print("\n================ SUMMARY ================")
    header = f"{'Split':<8}{'Precision':>12}{'Recall':>10}{'F1':>8}{'mAP50':>10}{'mAP50-95':>12}"
    print(header)
    print("-" * len(header))
    for row in results:
        print(
            f"{row['split']:<8}{row['precision']:>12.4f}{row['recall']:>10.4f}"
            f"{row['f1']:>8.4f}{row['map50']:>10.4f}{row['map50_95']:>12.4f}"
        )
    print("-" * len(header))
    print(f"Mean latency : {timing['mean_latency_ms']:.2f} ms")
    print(f"P95 latency  : {timing['p95_latency_ms']:.2f} ms")
    print(f"FPS          : {timing['fps']:.2f}")
    print("==========================================")
    print(
        "\nPlots (PR curve, F1 curve, confusion matrix) saved under "
        f"{TRAINING_ROOT / 'runs' / 'eval'}"
    )


if __name__ == "__main__":
    main()
