"""
Merge every already-converted dataset (training/data/converted/<name>/) into
one pooled directory (training/data/merged/), applying each dataset's
`weight` from config.yaml by duplicating (or subsampling) its file list so
the final training mix reflects the intended emphasis rather than raw
per-dataset image counts (CrowdHuman and WiderPerson should dominate the
mix; MOT/COCO/OpenImages/PennFudan are supporting material).

Filenames are prefixed with the source dataset name to avoid collisions
across datasets that happen to reuse image ids (e.g. "000001.jpg").

Usage:
    python training/scripts/merge_datasets.py
"""

import random
import shutil
from pathlib import Path

from common import TRAINING_ROOT, ensure_dir, load_config, set_seed


def _list_pairs(dataset_dir: Path, split: str):
    img_dir = dataset_dir / "images" / split
    lbl_dir = dataset_dir / "labels" / split
    if not img_dir.exists():
        return []
    pairs = []
    for image_path in img_dir.iterdir():
        if not image_path.is_file():
            continue
        label_path = lbl_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            pairs.append((image_path, label_path))
    return pairs


def merge(config: dict):
    merged_dir = ensure_dir(TRAINING_ROOT / "data" / "merged")
    for split in ("train", "val"):
        ensure_dir(merged_dir / "images" / split)
        ensure_dir(merged_dir / "labels" / split)

    total_written = 0
    per_dataset_counts = {}

    for entry in config["datasets"]:
        name = entry["name"]
        weight = float(entry.get("weight", 1.0))
        dataset_dir = TRAINING_ROOT.parent / entry["converted_dir"] if not Path(entry["converted_dir"]).is_absolute() \
            else Path(entry["converted_dir"])
        dataset_dir = (TRAINING_ROOT.parent / entry["converted_dir"]).resolve()

        written_this_dataset = 0
        for split in ("train", "val"):
            pairs = _list_pairs(dataset_dir, split)
            if not pairs:
                continue

            if weight >= 1.0:
                # Oversample by duplicating (with distinct output filenames)
                # for the fractional remainder, so a weight like 1.3 means
                # "every image once, plus 30% of them again".
                whole, frac = int(weight), weight - int(weight)
                selections = pairs * whole
                if frac > 0:
                    selections += random.sample(pairs, k=int(len(pairs) * frac))
            else:
                selections = random.sample(pairs, k=max(1, int(len(pairs) * weight)))

            for idx, (image_path, label_path) in enumerate(selections):
                out_name = f"{name}_{split}_{idx:06d}_{image_path.stem}"
                dest_img = merged_dir / "images" / split / f"{out_name}{image_path.suffix}"
                dest_lbl = merged_dir / "labels" / split / f"{out_name}.txt"
                if not dest_img.exists():
                    shutil.copyfile(image_path, dest_img)
                if not dest_lbl.exists():
                    shutil.copyfile(label_path, dest_lbl)
                written_this_dataset += 1

        per_dataset_counts[name] = written_this_dataset
        total_written += written_this_dataset
        print(f"  merged {name}: {written_this_dataset} image/label pairs (weight={weight})")

    print(f"\nTotal merged image/label pairs: {total_written}")
    print("Per-dataset contribution:")
    for name, count in per_dataset_counts.items():
        pct = 100.0 * count / max(1, total_written)
        print(f"  {name:>20s}: {count:>7d}  ({pct:5.1f}%)")

    return merged_dir


def main():
    config = load_config()
    set_seed(config["project"]["seed"])
    merge(config)


if __name__ == "__main__":
    main()
