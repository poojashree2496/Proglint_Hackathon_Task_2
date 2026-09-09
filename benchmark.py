"""
Head-to-head comparison: stock yolo11s.pt vs. the custom-trained model.

Runs both models against the same held-out test split and against
labelled subset slices for the three conditions this project cares
about most — crowded scenes, low-light scenes, and heavily-occluded
people — then prints one comparison table.

Slice selection: rather than requiring separate hand-labelled slice
datasets, this script derives the three conditions directly from the test
split using cheap, explainable heuristics:
  * crowded      -> images with >= 8 ground-truth boxes
  * low_light    -> images whose mean grayscale pixel value is < 60
  * occluded     -> images where at least one pair of ground-truth boxes
                     has IoU > 0.3 (a proxy for people overlapping)
Small-person detection is measured as recall restricted to GT boxes whose
area is below 32x32px (COCO's own "small object" convention).

Usage:
    python training/benchmark.py --custom-weights training/runs/surveillance_human_detector/weights/best.pt
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from scripts.common import TRAINING_ROOT, load_config, resolve_device


def iou_xyxy(a, b):
    xa1, ya1, xa2, ya2 = a
    xb1, yb1, xb2, yb2 = b
    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_ground_truth(image_path: Path, label_path: Path):
    image = cv2.imread(str(image_path))
    if image is None:
        return None, []
    h, w = image.shape[:2]
    boxes = []
    if label_path.exists():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            _, cx, cy, bw, bh = (float(x) for x in parts)
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h
            boxes.append((x1, y1, x2, y2))
    return image, boxes


def classify_slices(image, boxes) -> dict:
    gray_mean = float(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean())
    is_crowded = len(boxes) >= 8
    is_low_light = gray_mean < 60.0

    is_occluded = False
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if iou_xyxy(boxes[i], boxes[j]) > 0.3:
                is_occluded = True
                break
        if is_occluded:
            break

    return {"crowded": is_crowded, "low_light": is_low_light, "occluded": is_occluded, "all": True}


def match_detections(pred_boxes, gt_boxes, iou_thresh=0.5):
    matched_gt = set()
    tp = 0
    for pred in pred_boxes:
        best_iou, best_j = 0.0, -1
        for j, gt in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            score = iou_xyxy(pred, gt)
            if score > best_iou:
                best_iou, best_j = score, j
        if best_iou >= iou_thresh:
            matched_gt.add(best_j)
            tp += 1
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


def run_model_on_test_set(model: YOLO, test_img_dir: Path, test_lbl_dir: Path, imgsz, conf, iou, device):
    slice_stats = {name: {"tp": 0, "fp": 0, "fn": 0, "small_tp": 0, "small_gt": 0} for name in
                   ("all", "crowded", "low_light", "occluded")}

    image_paths = sorted(p for p in test_img_dir.iterdir() if p.is_file())
    latencies = []

    for image_path in image_paths:
        label_path = test_lbl_dir / f"{image_path.stem}.txt"
        image, gt_boxes = load_ground_truth(image_path, label_path)
        if image is None:
            continue

        start = time.perf_counter()
        result = model.predict(image, imgsz=imgsz, conf=conf, iou=iou, device=device, verbose=False)[0]
        latencies.append(time.perf_counter() - start)

        pred_boxes = result.boxes.xyxy.cpu().numpy().tolist() if result.boxes is not None else []

        slices = classify_slices(image, gt_boxes)
        tp, fp, fn = match_detections(pred_boxes, gt_boxes)

        small_gt = [b for b in gt_boxes if (b[2] - b[0]) * (b[3] - b[1]) < 32 * 32]
        small_tp, _, _ = match_detections(pred_boxes, small_gt)

        for name, active in slices.items():
            if not active:
                continue
            slice_stats[name]["tp"] += tp
            slice_stats[name]["fp"] += fp
            slice_stats[name]["fn"] += fn
            slice_stats[name]["small_tp"] += small_tp
            slice_stats[name]["small_gt"] += len(small_gt)

    fps = 1.0 / (sum(latencies) / len(latencies)) if latencies else 0.0

    summary = {}
    for name, stats in slice_stats.items():
        precision = stats["tp"] / max(1, stats["tp"] + stats["fp"])
        recall = stats["tp"] / max(1, stats["tp"] + stats["fn"])
        small_recall = stats["small_tp"] / max(1, stats["small_gt"])
        summary[name] = {"precision": precision, "recall": recall, "small_recall": small_recall}
    summary["fps"] = fps
    return summary


def main():
    parser = argparse.ArgumentParser(description="Benchmark stock YOLO11s vs. the custom human detector.")
    parser.add_argument("--baseline-weights", default="yolo11s.pt")
    parser.add_argument("--custom-weights", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path, default=None, help="Defaults to training/data/final")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(config["project"]["device"])
    inf_cfg = config["inference"]
    imgsz = config["model"]["imgsz"]

    data_dir = args.data_dir or (TRAINING_ROOT / "data" / "final")
    test_img_dir = data_dir / "images" / "test"
    test_lbl_dir = data_dir / "labels" / "test"
    if not test_img_dir.exists():
        raise FileNotFoundError(f"{test_img_dir} not found — run prepare_dataset.py first.")

    print("Loading models...")
    baseline = YOLO(args.baseline_weights)
    custom = YOLO(str(args.custom_weights))

    print("Running baseline (yolo11s.pt)...")
    baseline_results = run_model_on_test_set(
        baseline, test_img_dir, test_lbl_dir, imgsz, inf_cfg["conf_threshold"], inf_cfg["iou_threshold"], device
    )

    print("Running custom model...")
    custom_results = run_model_on_test_set(
        custom, test_img_dir, test_lbl_dir, imgsz, inf_cfg["conf_threshold"], inf_cfg["iou_threshold"], device
    )

    print("\n================= BENCHMARK: yolo11s.pt vs custom =================")
    header = f"{'Condition':<12}{'Metric':<14}{'yolo11s.pt':>14}{'custom':>14}{'delta':>10}"
    print(header)
    print("-" * len(header))
    for condition in ("all", "crowded", "low_light", "occluded"):
        for metric in ("precision", "recall", "small_recall"):
            b = baseline_results[condition][metric]
            c = custom_results[condition][metric]
            print(f"{condition:<12}{metric:<14}{b:>14.4f}{c:>14.4f}{c - b:>+10.4f}")
    print("-" * len(header))
    print(f"{'FPS':<12}{'':<14}{baseline_results['fps']:>14.2f}{custom_results['fps']:>14.2f}"
          f"{custom_results['fps'] - baseline_results['fps']:>+10.2f}")
    print("=====================================================================")


if __name__ == "__main__":
    main()
