"""
Bake weather/lighting augmentations into a copy of the TRAIN split only.

Ultralytics' built-in augmenter (mosaic/mixup/hsv/copy-paste/flip/etc., all
configured in config.yaml -> `augmentation`, applied live during training)
has no concept of fog, rain, low-light, or shadow. Those need to be baked
into image files up front via Albumentations. This script reads
training/data/final/{images,labels}/train, and for a configurable fraction
of images writes a weather-augmented duplicate (same label file, since
these are pixel-only transforms that don't move boxes) back into the same
train folder — so Ultralytics' own augmenter still applies on top at
training time.

Why each effect is included:
  * motion_blur  - real CCTV footage compresses to fixed frame rates; fast
                   walkers/runners smear. Without training examples of this,
                   the model under-detects running/fast-moving people.
  * gaussian_blur - cheap analog/IP cameras and video compression softness.
  * fog          - outdoor cameras in haze/fog lose contrast; without this
                   the model over-relies on sharp edges that fog removes.
  * rain         - rain streaks partially occlude people and add noise
                   texture the model must learn to see through.
  * low_light    - night-time CCTV is underexposed and noisy; this is one
                   of the most common real-world failure modes for
                   pretrained daylight-biased detectors.
  * shadow       - harsh outdoor sun casts hard shadows across bodies,
                   which can look like an occlusion boundary or a second
                   "leg" if never seen in training.
  * noise        - sensor noise from low-cost/analog CCTV hardware.

Usage:
    python training/scripts/augment_weather.py
"""

import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from tqdm import tqdm

from common import TRAINING_ROOT, load_config, set_seed


def build_transform(name: str) -> A.Compose:
    if name == "motion_blur":
        return A.Compose([A.MotionBlur(blur_limit=(9, 21), p=1.0)])
    if name == "gaussian_blur":
        return A.Compose([A.GaussianBlur(blur_limit=(5, 11), p=1.0)])
    if name == "fog":
        return A.Compose([A.RandomFog(fog_coef_lower=0.15, fog_coef_upper=0.45, alpha_coef=0.08, p=1.0)])
    if name == "rain":
        return A.Compose([A.RandomRain(
            slant_lower=-8, slant_upper=8, drop_length=14, drop_width=1,
            blur_value=3, brightness_coefficient=0.85, p=1.0,
        )])
    if name == "low_light":
        return A.Compose([
            A.RandomBrightnessContrast(brightness_limit=(-0.55, -0.30), contrast_limit=(-0.2, 0.1), p=1.0),
            A.GaussNoise(var_limit=(15.0, 45.0), p=0.8),
        ])
    if name == "shadow":
        return A.Compose([A.RandomShadow(
            shadow_roi=(0, 0.3, 1, 1), num_shadows_lower=1, num_shadows_upper=3,
            shadow_dimension=5, p=1.0,
        )])
    if name == "noise":
        return A.Compose([A.GaussNoise(var_limit=(10.0, 60.0), p=1.0)])
    raise ValueError(f"Unknown weather transform: {name}")


def augment(config: dict):
    aug_cfg = config["augmentation"]
    final_dir = TRAINING_ROOT / "data" / "final"
    img_dir = final_dir / "images" / "train"
    lbl_dir = final_dir / "labels" / "train"

    if not img_dir.exists():
        raise FileNotFoundError(f"{img_dir} not found — run prepare_dataset.py first.")

    effects = {
        "motion_blur": aug_cfg["motion_blur_prob"],
        "gaussian_blur": aug_cfg["gaussian_blur_prob"],
        "fog": aug_cfg["fog_prob"],
        "rain": aug_cfg["rain_prob"],
        "low_light": aug_cfg["low_light_prob"],
        "shadow": aug_cfg["shadow_prob"],
        "noise": aug_cfg["noise_prob"],
    }
    transforms = {name: build_transform(name) for name in effects}

    image_paths = sorted(p for p in img_dir.iterdir() if p.is_file())
    written = 0

    for image_path in tqdm(image_paths, desc="Baking weather augmentations"):
        for effect_name, prob in effects.items():
            if random.random() >= prob:
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                continue

            transformed = transforms[effect_name](image=image)["image"]
            out_stem = f"{image_path.stem}_{effect_name}"
            out_img_path = img_dir / f"{out_stem}{image_path.suffix}"
            cv2.imwrite(str(out_img_path), transformed)

            label_path = lbl_dir / f"{image_path.stem}.txt"
            if label_path.exists():
                (lbl_dir / f"{out_stem}.txt").write_text(
                    label_path.read_text(encoding="utf-8"), encoding="utf-8"
                )
            written += 1

    print(f"Wrote {written} weather-augmented train images (labels copied unchanged).")


def main():
    config = load_config()
    set_seed(config["project"]["seed"])
    augment(config)


if __name__ == "__main__":
    main()
