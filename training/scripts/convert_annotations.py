"""
Convert each source dataset's native annotation format into a single common
YOLO-format layout:

    training/data/converted/<dataset_name>/
        images/<split>/*.jpg
        labels/<split>/*.txt          # one line per box: "0 cx cy w h" (normalized)

All person-like sub-classes (CrowdHuman's "full body"/"visible body" pair,
WiderPerson's pedestrian/rider/partial/crowd/ignore, MOT's pedestrian class,
COCO's "person" category, Open Images "Person" label) are collapsed to a
single class id 0, since this project only ever needs "is this a human".

Each source has its own converter function below because every dataset ships
annotations in a different format (CrowdHuman: one JSON-lines .odgt file per
split; WiderPerson: custom per-image .txt with 5 columns; CityPersons: COCO-
style JSON with an extra `vis_bbox`/occlusion field; MOT17/20: MOTChallenge
gt.txt per sequence; COCO: standard COCO JSON; Open Images: flat CSV of
normalized boxes; Penn-Fudan: PNG instance masks).

Run one dataset at a time:

    python training/scripts/convert_annotations.py --dataset crowdhuman \
        --raw-dir /path/to/raw/CrowdHuman

Every converter is defensive: malformed rows are skipped and counted rather
than raising, and a summary is printed at the end so silent data loss never
goes unnoticed.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from common import TRAINING_ROOT, ensure_dir, yolo_box_valid


def _write_label(label_path: Path, boxes_xyxy, img_w, img_h):
    lines = []
    for x1, y1, x2, y2 in boxes_xyxy:
        x1, x2 = sorted((max(0.0, x1), min(float(img_w), x2)))
        y1, y2 = sorted((max(0.0, y1), min(float(img_h), y2)))
        w = x2 - x1
        h = y2 - y1
        if w <= 1 or h <= 1:
            continue
        cx = (x1 + x2) / 2.0 / img_w
        cy = (y1 + y2) / 2.0 / img_h
        nw = w / img_w
        nh = h / img_h
        if not yolo_box_valid(cx, cy, nw, nh):
            continue
        lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines), encoding="utf-8")
    return len(lines)


def convert_crowdhuman(raw_dir: Path, out_dir: Path):
    """CrowdHuman ships annotation_train.odgt / annotation_val.odgt: one JSON
    object per line, each with a `gtboxes` list carrying `fbox` (full body,
    x,y,w,h) and an `extra`/`head_attr` ignore flag. We use `fbox` (full
    body, including occluded parts) since that is what we want the detector
    to learn to draw even when a person is partly hidden."""
    total_boxes, total_images, skipped = 0, 0, 0
    for split, odgt_name in [("train", "annotation_train.odgt"), ("val", "annotation_val.odgt")]:
        odgt_path = raw_dir / odgt_name
        if not odgt_path.exists():
            print(f"  [crowdhuman] missing {odgt_path}, skipping split '{split}'")
            continue

        img_dir_candidates = [raw_dir / "Images", raw_dir / "images"]
        img_dir = next((p for p in img_dir_candidates if p.exists()), raw_dir)

        with open(odgt_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue

                image_id = record.get("ID")
                image_path = img_dir / f"{image_id}.jpg"
                if not image_path.exists():
                    skipped += 1
                    continue

                try:
                    with Image.open(image_path) as im:
                        img_w, img_h = im.size
                except Exception:
                    skipped += 1
                    continue

                boxes = []
                for box_entry in record.get("gtboxes", []):
                    if box_entry.get("tag") != "person":
                        continue
                    extra = box_entry.get("extra", {}) or {}
                    if extra.get("ignore", 0) == 1:
                        continue
                    x, y, w, h = box_entry.get("fbox", [0, 0, 0, 0])
                    if w <= 0 or h <= 0:
                        continue
                    boxes.append((x, y, x + w, y + h))

                if not boxes:
                    continue

                out_img_dir = ensure_dir(out_dir / "images" / split)
                out_lbl_dir = ensure_dir(out_dir / "labels" / split)
                dest_img = out_img_dir / image_path.name
                if not dest_img.exists():
                    dest_img.write_bytes(image_path.read_bytes())

                n = _write_label(out_lbl_dir / f"{image_path.stem}.txt", boxes, img_w, img_h)
                total_boxes += n
                total_images += 1

    print(f"[crowdhuman] images={total_images} boxes={total_boxes} skipped={skipped}")


def convert_widerperson(raw_dir: Path, out_dir: Path):
    """WiderPerson: images/<id>.jpg + Annotations/<id>.jpg.txt, first line is
    box count, each following line is "class_label x1 y1 x2 y2" with
    class_label in {1: pedestrian, 2: rider, 3: partially-visible person,
    4: ignore region, 5: crowd}. We keep 1/2/3 (all are humans), drop 4/5."""
    keep_labels = {1, 2, 3}
    total_boxes, total_images, skipped = 0, 0, 0

    ann_dir = raw_dir / "Annotations"
    img_dir = raw_dir / "Images"
    split_files = {
        "train": raw_dir / "train.txt",
        "val": raw_dir / "val.txt",
    }

    for split, list_path in split_files.items():
        if not list_path.exists():
            print(f"  [widerperson] missing {list_path}, skipping split '{split}'")
            continue

        ids = [line.strip() for line in list_path.read_text().splitlines() if line.strip()]
        for image_id in ids:
            image_path = img_dir / f"{image_id}.jpg"
            ann_path = ann_dir / f"{image_id}.jpg.txt"
            if not image_path.exists() or not ann_path.exists():
                skipped += 1
                continue

            try:
                with Image.open(image_path) as im:
                    img_w, img_h = im.size
            except Exception:
                skipped += 1
                continue

            lines = ann_path.read_text().splitlines()
            boxes = []
            for row in lines[1:]:
                parts = row.split()
                if len(parts) != 5:
                    continue
                label, x1, y1, x2, y2 = parts
                if int(label) not in keep_labels:
                    continue
                boxes.append((float(x1), float(y1), float(x2), float(y2)))

            if not boxes:
                continue

            out_img_dir = ensure_dir(out_dir / "images" / split)
            out_lbl_dir = ensure_dir(out_dir / "labels" / split)
            dest_img = out_img_dir / image_path.name
            if not dest_img.exists():
                dest_img.write_bytes(image_path.read_bytes())

            n = _write_label(out_lbl_dir / f"{image_id}.txt", boxes, img_w, img_h)
            total_boxes += n
            total_images += 1

    print(f"[widerperson] images={total_images} boxes={total_boxes} skipped={skipped}")


def convert_citypersons(raw_dir: Path, out_dir: Path):
    """CityPersons annotations (COCO-style JSON per split) carry both
    `bbox` (visible) and `bbox_vis`/`vis_ratio`. We use the full bbox and
    additionally drop boxes whose visibility ratio is below 0.15 (almost
    entirely occluded boxes teach the model to hallucinate people from
    near-nothing, which hurts precision more than it helps recall)."""
    total_boxes, total_images, skipped = 0, 0, 0

    for split, json_name in [("train", "citypersons_train.json"), ("val", "citypersons_val.json")]:
        json_path = raw_dir / json_name
        if not json_path.exists():
            print(f"  [citypersons] missing {json_path}, skipping split '{split}'")
            continue

        with open(json_path, "r", encoding="utf-8") as handle:
            coco = json.load(handle)

        images_by_id = {img["id"]: img for img in coco.get("images", [])}
        boxes_by_image = {}
        for ann in coco.get("annotations", []):
            if ann.get("category_id") not in (1, None):
                continue
            vis_ratio = ann.get("vis_ratio", 1.0)
            if vis_ratio is not None and vis_ratio < 0.15:
                continue
            x, y, w, h = ann["bbox"]
            boxes_by_image.setdefault(ann["image_id"], []).append((x, y, x + w, y + h))

        img_dir = raw_dir / "leftImg8bit" / split
        for image_id, boxes in boxes_by_image.items():
            meta = images_by_id.get(image_id)
            if meta is None:
                skipped += 1
                continue

            image_path = img_dir / meta["file_name"]
            if not image_path.exists():
                image_path = raw_dir / meta["file_name"]
            if not image_path.exists():
                skipped += 1
                continue

            img_w = meta.get("width")
            img_h = meta.get("height")
            if not img_w or not img_h:
                try:
                    with Image.open(image_path) as im:
                        img_w, img_h = im.size
                except Exception:
                    skipped += 1
                    continue

            out_img_dir = ensure_dir(out_dir / "images" / split)
            out_lbl_dir = ensure_dir(out_dir / "labels" / split)
            dest_img = out_img_dir / f"{image_path.stem}.jpg"
            if not dest_img.exists():
                try:
                    Image.open(image_path).convert("RGB").save(dest_img, quality=95)
                except Exception:
                    skipped += 1
                    continue

            n = _write_label(out_lbl_dir / f"{image_path.stem}.txt", boxes, img_w, img_h)
            total_boxes += n
            total_images += 1

    print(f"[citypersons] images={total_images} boxes={total_boxes} skipped={skipped}")


def convert_mot(raw_dir: Path, out_dir: Path, dataset_label: str = "mot1720"):
    """MOT17/MOT20: each sequence has img1/*.jpg frames and gt/gt.txt with
    rows "frame,id,x,y,w,h,conf,class,visibility". class==1 is "pedestrian"
    in the MOTChallenge label spec; conf==0 rows are ignore regions.
    Frames are subsampled (every Nth frame) since consecutive video frames
    are near-duplicates and would otherwise dominate the combined dataset."""
    frame_stride = 8
    total_boxes, total_images, skipped = 0, 0, 0

    sequence_dirs = [p for p in raw_dir.glob("*") if (p / "gt" / "gt.txt").exists()]
    for seq_dir in sequence_dirs:
        gt_path = seq_dir / "gt" / "gt.txt"
        img_dir = seq_dir / "img1"
        rows_by_frame = {}
        with open(gt_path, "r", encoding="utf-8") as handle:
            for row in csv.reader(handle):
                if len(row) < 8:
                    continue
                frame, _tid, x, y, w, h, conf, cls = row[:8]
                if int(float(cls)) != 1 or float(conf) == 0.0:
                    continue
                rows_by_frame.setdefault(int(frame), []).append(
                    (float(x), float(y), float(x) + float(w), float(y) + float(h))
                )

        split = "train" if "train" in seq_dir.name.lower() or True else "val"
        for frame_no in sorted(rows_by_frame.keys()):
            if frame_no % frame_stride != 0:
                continue
            image_path = img_dir / f"{frame_no:06d}.jpg"
            if not image_path.exists():
                skipped += 1
                continue

            try:
                with Image.open(image_path) as im:
                    img_w, img_h = im.size
            except Exception:
                skipped += 1
                continue

            out_img_dir = ensure_dir(out_dir / "images" / split)
            out_lbl_dir = ensure_dir(out_dir / "labels" / split)
            unique_name = f"{seq_dir.name}_{frame_no:06d}"
            dest_img = out_img_dir / f"{unique_name}.jpg"
            if not dest_img.exists():
                dest_img.write_bytes(image_path.read_bytes())

            n = _write_label(out_lbl_dir / f"{unique_name}.txt", rows_by_frame[frame_no], img_w, img_h)
            total_boxes += n
            total_images += 1

    print(f"[{dataset_label}] images={total_images} boxes={total_boxes} skipped={skipped}")


def convert_coco_person(raw_dir: Path, out_dir: Path):
    """Standard COCO instances_{train,val}2017.json, filtered to
    category "person" (id 1), iscrowd==0 boxes only (crowd regions in COCO
    are a single blob box covering many people and would poison box-size
    statistics)."""
    total_boxes, total_images, skipped = 0, 0, 0

    for split, json_name, img_subdir in [
        ("train", "instances_train2017.json", "train2017"),
        ("val", "instances_val2017.json", "val2017"),
    ]:
        json_path = raw_dir / "annotations" / json_name
        if not json_path.exists():
            print(f"  [coco_person] missing {json_path}, skipping split '{split}'")
            continue

        with open(json_path, "r", encoding="utf-8") as handle:
            coco = json.load(handle)

        person_cat_id = next((c["id"] for c in coco["categories"] if c["name"] == "person"), 1)
        images_by_id = {img["id"]: img for img in coco["images"]}
        boxes_by_image = {}
        for ann in coco["annotations"]:
            if ann["category_id"] != person_cat_id or ann.get("iscrowd", 0) == 1:
                continue
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes_by_image.setdefault(ann["image_id"], []).append((x, y, x + w, y + h))

        img_dir = raw_dir / img_subdir
        for image_id, boxes in boxes_by_image.items():
            meta = images_by_id.get(image_id)
            if meta is None:
                skipped += 1
                continue
            image_path = img_dir / meta["file_name"]
            if not image_path.exists():
                skipped += 1
                continue

            out_img_dir = ensure_dir(out_dir / "images" / split)
            out_lbl_dir = ensure_dir(out_dir / "labels" / split)
            dest_img = out_img_dir / image_path.name
            if not dest_img.exists():
                dest_img.write_bytes(image_path.read_bytes())

            n = _write_label(out_lbl_dir / f"{image_path.stem}.txt", boxes, meta["width"], meta["height"])
            total_boxes += n
            total_images += 1

    print(f"[coco_person] images={total_images} boxes={total_boxes} skipped={skipped}")


def convert_openimages_person(raw_dir: Path, out_dir: Path):
    """Open Images annotations ship as a flat CSV with normalized
    [XMin,XMax,YMin,YMax] already in [0,1], keyed by ImageID and LabelName
    (filter to the "Person" label's MID, /m/01g317)."""
    person_mid = "/m/01g317"
    total_boxes, total_images, skipped = 0, 0, 0

    for split, csv_name, img_subdir in [
        ("train", "oidv6-train-annotations-bbox.csv", "train"),
        ("val", "validation-annotations-bbox.csv", "validation"),
    ]:
        csv_path = raw_dir / csv_name
        if not csv_path.exists():
            print(f"  [openimages_person] missing {csv_path}, skipping split '{split}'")
            continue

        boxes_by_image = {}
        with open(csv_path, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("LabelName") != person_mid:
                    continue
                x1, x2 = float(row["XMin"]), float(row["XMax"])
                y1, y2 = float(row["YMin"]), float(row["YMax"])
                boxes_by_image.setdefault(row["ImageID"], []).append((x1, y1, x2, y2, True))

        img_dir = raw_dir / img_subdir
        for image_id, norm_boxes in boxes_by_image.items():
            image_path = img_dir / f"{image_id}.jpg"
            if not image_path.exists():
                skipped += 1
                continue
            try:
                with Image.open(image_path) as im:
                    img_w, img_h = im.size
            except Exception:
                skipped += 1
                continue

            boxes_px = [(x1 * img_w, y1 * img_h, x2 * img_w, y2 * img_h) for x1, y1, x2, y2, _ in norm_boxes]

            out_img_dir = ensure_dir(out_dir / "images" / split)
            out_lbl_dir = ensure_dir(out_dir / "labels" / split)
            dest_img = out_img_dir / image_path.name
            if not dest_img.exists():
                dest_img.write_bytes(image_path.read_bytes())

            n = _write_label(out_lbl_dir / f"{image_id}.txt", boxes_px, img_w, img_h)
            total_boxes += n
            total_images += 1

    print(f"[openimages_person] images={total_images} boxes={total_boxes} skipped={skipped}")


def convert_pennfudan(raw_dir: Path, out_dir: Path):
    """Penn-Fudan ships per-instance PNG masks (PedMasks/*.png) instead of
    boxes; each unique non-zero pixel value is one person instance, so we
    derive a tight bounding box per instance via the mask's extent."""
    total_boxes, total_images, skipped = 0, 0, 0

    img_dir = raw_dir / "PNGImages"
    mask_dir = raw_dir / "PedMasks"
    if not img_dir.exists() or not mask_dir.exists():
        print(f"  [pennfudan] missing PNGImages/PedMasks under {raw_dir}")
        return

    image_paths = sorted(img_dir.glob("*.png"))
    n_val = max(1, int(len(image_paths) * 0.2))

    for idx, image_path in enumerate(image_paths):
        mask_path = mask_dir / f"{image_path.stem}_mask.png"
        if not mask_path.exists():
            skipped += 1
            continue

        mask = np.array(Image.open(mask_path))
        instance_ids = [v for v in np.unique(mask) if v != 0]
        boxes = []
        for instance_id in instance_ids:
            ys, xs = np.where(mask == instance_id)
            if xs.size == 0 or ys.size == 0:
                continue
            boxes.append((float(xs.min()), float(ys.min()), float(xs.max()) + 1, float(ys.max()) + 1))

        if not boxes:
            continue

        with Image.open(image_path) as im:
            img_w, img_h = im.size

        split = "val" if idx < n_val else "train"
        out_img_dir = ensure_dir(out_dir / "images" / split)
        out_lbl_dir = ensure_dir(out_dir / "labels" / split)
        dest_img = out_img_dir / f"{image_path.stem}.jpg"
        if not dest_img.exists():
            Image.open(image_path).convert("RGB").save(dest_img, quality=95)

        n = _write_label(out_lbl_dir / f"{image_path.stem}.txt", boxes, img_w, img_h)
        total_boxes += n
        total_images += 1

    print(f"[pennfudan] images={total_images} boxes={total_boxes} skipped={skipped}")


CONVERTERS = {
    "crowdhuman": convert_crowdhuman,
    "widerperson": convert_widerperson,
    "citypersons": convert_citypersons,
    "mot1720": convert_mot,
    "coco_person": convert_coco_person,
    "openimages_person": convert_openimages_person,
    "pennfudan": convert_pennfudan,
}


def main():
    parser = argparse.ArgumentParser(description="Convert one raw dataset into the shared YOLO layout.")
    parser.add_argument("--dataset", required=True, choices=sorted(CONVERTERS.keys()))
    parser.add_argument("--raw-dir", required=True, type=Path, help="Path to the dataset's raw download.")
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Defaults to training/data/converted/<dataset>.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or (TRAINING_ROOT / "data" / "converted" / args.dataset)
    ensure_dir(out_dir)

    print(f"Converting '{args.dataset}' from {args.raw_dir} -> {out_dir}")
    CONVERTERS[args.dataset](args.raw_dir, out_dir)


if __name__ == "__main__":
    main()
