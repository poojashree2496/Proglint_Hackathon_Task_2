"""
Train the custom surveillance human-detector.

Base architecture: YOLO11s (see README "Architecture choice" for the
n/s/m/l/x tradeoff analysis — 's' is the balance point of accuracy vs. the
real-time-with-BoT-SORT requirement this project has).

All hyperparameters live in config.yaml so this script stays a thin,
readable wrapper around Ultralytics' trainer rather than a pile of CLI
flags — see config.yaml's inline comments for what each value does and why.

Usage:
    python training/train.py
    python training/train.py --config training/config.yaml --resume
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

from scripts.common import TRAINING_ROOT, load_config, resolve_device, set_seed


def main():
    parser = argparse.ArgumentParser(description="Train the custom human detector.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data", type=Path, default=None, help="Override data.yaml path.")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["project"]["seed"])
    device = resolve_device(config["project"]["device"])

    data_yaml = args.data or (TRAINING_ROOT / "data" / "final" / "data.yaml")
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"{data_yaml} not found. Run merge_datasets.py then prepare_dataset.py first."
        )

    model_cfg = config["model"]
    aug_cfg = config["augmentation"]

    model = YOLO(model_cfg["base"])

    print(f"Training on device: {device}")
    print(f"Data config: {data_yaml}")

    model.train(
        data=str(data_yaml),
        imgsz=model_cfg["imgsz"],
        epochs=model_cfg["epochs"],
        patience=model_cfg["patience"],
        batch=model_cfg["batch"],
        optimizer=model_cfg["optimizer"],
        lr0=model_cfg["lr0"],
        lrf=model_cfg["lrf"],
        cos_lr=model_cfg["cos_lr"],
        warmup_epochs=model_cfg["warmup_epochs"],
        warmup_momentum=model_cfg["warmup_momentum"],
        momentum=model_cfg["momentum"],
        weight_decay=model_cfg["weight_decay"],
        label_smoothing=model_cfg["label_smoothing"],
        box=model_cfg["box"],
        cls=model_cfg["cls"],
        dfl=model_cfg["dfl"],
        amp=model_cfg["amp"],
        close_mosaic=model_cfg["close_mosaic"],
        multi_scale=model_cfg["multi_scale"],
        deterministic=model_cfg["deterministic"],
        device=device,
        resume=args.resume,
        seed=config["project"]["seed"],
        project=str(TRAINING_ROOT / "runs"),
        name="surveillance_human_detector",
        exist_ok=True,
        # single-class task: no benefit from mixing classes, keep it explicit
        single_cls=True,
        # augmentation knobs (see config.yaml for rationale per parameter)
        hsv_h=aug_cfg["hsv_h"],
        hsv_s=aug_cfg["hsv_s"],
        hsv_v=aug_cfg["hsv_v"],
        degrees=aug_cfg["degrees"],
        translate=aug_cfg["translate"],
        scale=aug_cfg["scale"],
        shear=aug_cfg["shear"],
        perspective=aug_cfg["perspective"],
        flipud=aug_cfg["flipud"],
        fliplr=aug_cfg["fliplr"],
        mosaic=aug_cfg["mosaic"],
        mixup=aug_cfg["mixup"],
        copy_paste=aug_cfg["copy_paste"],
        erasing=aug_cfg["erasing"],
        plots=True,
        val=True,
    )

    best_path = TRAINING_ROOT / "runs" / "surveillance_human_detector" / "weights" / "best.pt"
    print(f"\nTraining complete. Best weights: {best_path}")
    print("Next: python training/scripts/export_model.py --weights "
          f"\"{best_path}\"")


if __name__ == "__main__":
    main()
