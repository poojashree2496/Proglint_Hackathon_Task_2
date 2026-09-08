"""
Clean the merged dataset and produce the final train/val/test split that
Ultralytics trains on.

Steps (in order):
  1. Validate every label file: parse-check, drop degenerate/out-of-range
     boxes, drop boxes below the configured min size (label noise from
     dense-crowd datasets is dominated by 2-3px "boxes" around a distant
     head that are unlabelable and just teach the model to false-positive).
  2. Perceptual-hash every image and drop near-duplicates (common across
     MOT's consecutive video frames and multiple datasets sourcing the same
     public photos).
  3. Drop images that are too blurry to be usable ground truth (Laplacian
     variance below threshold) — this specifically targets motion-blurred
     MOT frames that are blurred enough to make the person unrecognizable
     even to a human labeller, as opposed to the *desirable* motion blur we
     later synthesize on purpose in augment_weather.py.
  4. Re-split the surviving pool into train/val/test using the ratios in
     config.yaml, writing a `data.yaml` Ultralytics can train against
     directly.

Usage:
    python training/scripts/prepare_dataset.py
"""

import random
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image
from tqdm import tqdm

from common import TRAINING_ROOT, ensure_dir, load_config, set_seed, yolo_box_valid


def validate_and_clean_label(label_path: Path, min_box_area_px: int, min_box_side_px: int,
                              img_w: int, img_h: int) -> list:
    if not label_path.exists():
        return []

    kept = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            cls, cx, cy, w, h = (float(x) for x in parts)
        except ValueError:
            continue

        if not yolo_box_valid(cx, cy, w, h):
            continue

        box_w_px = w * img_w
        box_h_px = h * img_h
        if box_w_px < min_box_side_px or box_h_px < min_box_side_px:
            continue
        if box_w_px * box_h_px < min_box_area_px:
            continue

        kept.append((int(cls), cx, cy, w, h))
    return kept


def compute_phash(image_path: Path):
    try:
        with Image.open(image_path) as im:
            return imagehash.phash(im.convert("L"))
    except Exception:
        return None


def is_too_blurry(image_path: Path, variance_threshold: float) -> bool:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return True
    variance = cv2.Laplacian(image, cv2.CV_64F).var()
    return variance < variance_threshold


def prepare(config: dict):
    prep_cfg = config["preprocessing"]
    merged_dir = TRAINING_ROOT / "data" / "merged"
    final_dir = ensure_dir(TRAINING_ROOT / "data" / "final")

    all_pairs = []
    for split in ("train", "val"):
        img_dir = merged_dir / "images" / split
        lbl_dir = merged_dir / "labels" / split
        if not img_dir.exists():
            continue
        for image_path in img_dir.iterdir():
            if image_path.is_file():
                all_pairs.append((image_path, lbl_dir / f"{image_path.stem}.txt"))

    print(f"Starting from {len(all_pairs)} merged image/label pairs.")

    # ---- Step 1 + 3: label validation and blur filtering -----------------
    surviving = []
    dropped_blur, dropped_empty_label, dropped_unreadable = 0, 0, 0

    for image_path, label_path in tqdm(all_pairs, desc="Validating labels + blur"):
        image = cv2.imread(str(image_path))
        if image is None:
            dropped_unreadable += 1
            continue
        img_h, img_w = image.shape[:2]

        boxes = validate_and_clean_label(
            label_path, prep_cfg["min_box_area_px"], prep_cfg["min_box_side_px"], img_w, img_h
        )
        if not boxes:
            dropped_empty_label += 1
            continue

        if is_too_blurry(image_path, prep_cfg["blur_variance_threshold"]):
            dropped_blur += 1
            continue

        surviving.append((image_path, boxes))

    print(f"  dropped (unreadable image): {dropped_unreadable}")
    print(f"  dropped (no valid boxes after cleaning): {dropped_empty_label}")
    print(f"  dropped (too blurry): {dropped_blur}")
    print(f"  surviving: {len(surviving)}")

    # ---- Step 2: perceptual-hash de-duplication ---------------------------
    print("Hashing images for de-duplication...")
    seen_hashes = []
    deduped = []
    dropped_dupe = 0
    threshold = prep_cfg["duplicate_hash_threshold"]

    for image_path, boxes in tqdm(surviving, desc="De-duplicating"):
        phash = compute_phash(image_path)
        if phash is None:
            deduped.append((image_path, boxes))
            continue

        is_dupe = False
        for existing in seen_hashes:
            if phash - existing <= threshold:
                is_dupe = True
                break

        if is_dupe:
            dropped_dupe += 1
            continue

        seen_hashes.append(phash)
        deduped.append((image_path, boxes))

    print(f"  dropped (near-duplicate): {dropped_dupe}")
    print(f"  final pool: {len(deduped)}")

    # ---- Step 4: re-split and write out ------------------------------------
    random.shuffle(deduped)
    n = len(deduped)
    n_train = int(n * prep_cfg["train_split"])
    n_val = int(n * prep_cfg["val_split"])

    splits = {
        "train": deduped[:n_train],
        "val": deduped[n_train:n_train + n_val],
        "test": deduped[n_train + n_val:],
    }

    for split, items in splits.items():
        img_out = ensure_dir(final_dir / "images" / split)
        lbl_out = ensure_dir(final_dir / "labels" / split)
        for image_path, boxes in tqdm(items, desc=f"Writing '{split}'"):
            dest_img = img_out / image_path.name
            dest_img.write_bytes(image_path.read_bytes())
            lines = [f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cls, cx, cy, w, h in boxes]
            (lbl_out / f"{image_path.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        print(f"  {split}: {len(items)} images")

    data_yaml_path = final_dir / "data.yaml"
    data_yaml_path.write_text(
        "path: {}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: person\n".format(final_dir.as_posix()),
        encoding="utf-8",
    )
    print(f"\nWrote {data_yaml_path}")
    return data_yaml_path


def main():
    config = load_config()
    set_seed(config["project"]["seed"])
    prepare(config)


if __name__ == "__main__":
    main()
