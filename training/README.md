# Custom Surveillance Human Detector — Training Pipeline

A production-quality pipeline to train a **human-only** YOLO11 detector that
outperforms stock `yolo11s.pt` specifically on CCTV/surveillance-style
footage (occlusion, crowds, low light, motion blur, off-axis camera angles),
while staying fast enough to run in front of BoT-SORT in real time.

> **Honesty check before you run anything:** this pipeline needs (a) several
> of the raw datasets below downloaded yourself — most require a free
> click-through registration the pipeline cannot automate — totaling
> 40-80GB, and (b) a GPU and roughly 1-3 days of training time for the full
> recipe. If you're on a hackathon clock, read "Fast path" at the bottom.

---

## 1. Pipeline overview

```
raw datasets (you download)
        │
        ▼
convert_annotations.py   →  training/data/converted/<dataset>/  (YOLO format, class 0 = person)
        │
        ▼
merge_datasets.py        →  training/data/merged/                (weighted pool, per config.yaml)
        │
        ▼
prepare_dataset.py       →  training/data/final/                 (deduped, blur-filtered, re-split, data.yaml)
        │
        ▼
augment_weather.py       →  bakes fog/rain/low-light/shadow/blur/noise into the train split
        │
        ▼
train.py                 →  training/runs/surveillance_human_detector/weights/{best,last}.pt
        │
        ▼
evaluate.py               →  precision/recall/mAP/PR/F1/confusion-matrix + latency, on val AND test
benchmark.py               →  best.pt vs yolo11s.pt, sliced by crowded/low-light/occluded/small-person
        │
        ▼
scripts/export_model.py  →  ONNX + TorchScript, and copies best.pt → ../custom_human_model.pt
        │
        ▼
main.py / tracker_engine.py auto-detect custom_human_model.pt — nothing else to change
```

Every script is standalone and re-runnable; nothing here needs a notebook.

---

## 2. Dataset strategy

All datasets are reduced to a **single class: `person`**. Per-dataset detail
and inclusion rationale (also documented inline in `config.yaml`):

| Dataset | ~Images | Why it's here |
|---|---|---|
| **CrowdHuman** | ~19k (15k train / 4.4k val) | The single highest-value dataset for this project: ~470k boxes, ~23 people/image average, annotated with both full-body and visible-body boxes — i.e. it directly teaches the model what an occluded person looks like. Primary driver of crowd/occlusion robustness. |
| **WiderPerson** | ~13k | Dense pedestrian scenes across street/mall/campus-like settings; five sub-classes (pedestrian, rider, partially-visible, crowd, ignore) collapsed to `person` (crowd/ignore regions dropped). Adds outdoor diversity CrowdHuman under-represents. |
| **CityPersons** | ~5k | Urban street-camera footage (Cityscapes-derived) with occlusion-ratio metadata; used to filter out near-fully-occluded boxes. Strong signal for small/distant pedestrians — the hardest case for a CCTV camera mounted high and wide. |
| **MOT17 + MOT20** | Video, subsampled every 8th frame | Native surveillance-camera framing: train-station and mall-concourse footage (MOT20 in particular is extreme crowd density), real motion blur, indoor/night lighting. Frame-subsampled and down-weighted (0.6x) because consecutive video frames are near-duplicates — full-rate inclusion would make the dataset mostly redundant. |
| **COCO (person only)** | ~64k images with people | Pose and context diversity stock surveillance sets lack: sitting, running, sports, carrying bags, varied backgrounds. Down-weighted (0.5x) since COCO people skew close-up/large relative to the small-and-distant CCTV case. Prevents the model from overfitting to a narrow "CCTV look." |
| **Open Images (Person)** | Large, subsampled via weight | Broad ethnicity/clothing/lighting long-tail diversity at scale. Down-weighted (0.4x) — box-only labels, no occlusion metadata, used as a top-up rather than a primary source. |
| **Penn-Fudan** | 170 | Tiny; kept for clean walking-pose eval spot-checks only, not training volume (weight 0.2x). |

**Target size:** with full weights applied, the merged pool lands in the
40,000-70,000 image range — inside the 30k-100k target — dominated by
CrowdHuman + WiderPerson + CityPersons (the three actually shot for
pedestrian/surveillance detection), topped up with MOT/COCO/OpenImages for
diversity. Adjust each `weight` in `config.yaml` to shift the mix.

**Coverage of the requested scenarios:**
indoor (MOT20, CrowdHuman indoor scenes) / outdoor (WiderPerson, CityPersons,
MOT17) / malls & transit (MOT20) / campuses (WiderPerson) / streets
(CityPersons) / day+night (COCO + MOT + synthetic low-light augmentation) /
weather (synthetic fog/rain augmentation, since no source dataset here
natively ships labelled weather) / heavy crowd density (CrowdHuman, MOT20) /
occlusion (CrowdHuman full/visible box pairs, CityPersons visibility ratio) /
poses & carried bags (COCO).

### Where to get each dataset
- CrowdHuman: `https://www.crowdhuman.org/` (registration + download form)
- WiderPerson: `http://www.cbsr.ia.ac.cn/users/sfzhang/WiderPerson/`
- CityPersons: `https://github.com/cvgroup-njust/CityPersons` (built on Cityscapes; Cityscapes account required)
- MOT17 / MOT20: `https://motchallenge.net/`
- COCO: `https://cocodataset.org/#download` (2017 train/val images + annotations)
- Open Images: `https://storage.googleapis.com/openimages/web/index.html` (use the Person class subset + bbox CSVs)
- Penn-Fudan: `https://www.cis.upenn.edu/~jshi/ped_html/`

---

## 3. Cleaning & preprocessing (`prepare_dataset.py`)

1. **Label validation** — every YOLO line is parsed and re-checked:
   coordinates must be in `[0,1]`, width/height must be non-degenerate, and
   boxes smaller than `min_box_area_px` (36px², i.e. ~6x6) or
   `min_box_side_px` (3px) are dropped as unlabelable noise rather than
   signal.
2. **Blur filtering** — Laplacian variance below `blur_variance_threshold`
   (25.0) drops the image. This targets frames blurred enough that even a
   human labeller couldn't confirm the box — distinct from the *controlled*
   motion blur we synthesize on purpose in step 4 below.
3. **De-duplication** — perceptual hash (`imagehash.phash`) with a Hamming
   distance threshold of 5 catches near-duplicates, which are common across
   MOT's consecutive frames and across datasets that happen to source the
   same public photos.
4. **Re-split** — the surviving, deduplicated pool is shuffled and split
   70/20/10 train/val/test, and a `data.yaml` is written for Ultralytics.

Run:
```bash
python training/scripts/merge_datasets.py
python training/scripts/prepare_dataset.py
python training/scripts/augment_weather.py   # optional but recommended
```

---

## 4. Augmentation

Two layers:

**A. Baked-in weather/lighting** (`augment_weather.py`, Albumentations,
train split only, pixel-only so labels are untouched): motion blur, Gaussian
blur, fog, rain, low-light + sensor noise, hard shadows, generic noise. Each
targets a specific real-world CCTV failure mode — see the docstring at the
top of `augment_weather.py` for the reasoning behind each one.

**B. Live, on-the-fly augmentation during training** (Ultralytics-native,
configured in `config.yaml` → `augmentation`, applied by `train.py`):

| Param | Value | Why |
|---|---|---|
| `hsv_h/s/v` | 0.015 / 0.7 / 0.5 | Clothing color, camera white-balance, and exposure variation |
| `degrees` | 8° | CCTV mounts are rarely perfectly level |
| `scale` | 0.5 | Simulates near/far camera distance |
| `perspective` | 0.0006 | Off-axis high-mounted camera angle |
| `flipud` | 0.0 | People are never upside-down in CCTV — don't teach that |
| `fliplr` | 0.5 | Free pose-diversity doubling; humans are roughly bilaterally symmetric |
| `mosaic` | 1.0 | The single biggest lever for small-object and crowded-scene mAP — stitches 4 images so the model sees far more object density and scale variety per step |
| `mixup` | 0.10 | Light regularization against the label noise inherent in crowd datasets |
| `copy_paste` | 0.30 | Pastes people from one image into another — **directly synthesizes more occlusion and crowd density**, the model's core weak spot |
| `erasing` | 0.4 | Random erasing (CutOut-style) simulates partial-body occlusion by objects |

`close_mosaic: 15` disables mosaic for the final 15 epochs (standard YOLO
practice) so the model converges on undistorted, "clean" images.

---

## 5. Architecture choice

| Variant | Params | Relative speed | When it wins |
|---|---|---|---|
| YOLO11n | ~2.6M | fastest | Edge devices with no GPU; accuracy ceiling too low for crowded/occluded surveillance scenes |
| **YOLO11s** | **~9.4M** | **fast, real-time on a modern GPU or Apple Silicon** | **Chosen**: best accuracy-per-FPS for this project — has enough capacity to learn occlusion/crowd cues from CrowdHuman that `n` cannot, while staying light enough to run comfortably alongside BoT-SORT's own ReID model in real time |
| YOLO11m | ~20M | moderate | Meaningfully higher mAP, but the FPS drop compounds with BoT-SORT+ReID already in the pipeline; use only if you have a dedicated GPU and don't need real-time |
| YOLO11l / YOLO11x | 25M+ | slow | Diminishing returns for a single-class (person) task; the extra capacity mostly helps multi-class problems this project doesn't have |

**No architecture surgery is applied** — for a single-class problem, YOLO11's
existing multi-scale detection head (P3/P4/P5) already covers the near/far
person-size range surveillance footage needs; modifying it would risk
destabilizing the pretrained backbone weights transfer for a task that
doesn't need extra classes or heads. The gains here come from *data* (the
occlusion/crowd-heavy dataset mix) and *training recipe*, not model surgery.

`single_cls=True` is set explicitly in `train.py` even though every source
dataset is already collapsed to one class — it disables classification-loss
bookkeeping paths in Ultralytics that only matter for multi-class problems.

---

## 6. Training configuration & hyperparameters (`config.yaml` → `model`)

| Param | Value | Why |
|---|---|---|
| `imgsz` | 960 | Higher than YOLO's default 640 — surveillance cameras are often wide-angle with small, distant people; more input resolution directly improves small-object recall at a manageable speed cost for `s`-size |
| `epochs` | 200, `patience` 30 | Long enough for the cosine schedule to fully anneal; early-stops if val mAP50-95 stalls for 30 epochs so you don't wait out a plateau |
| `batch` | -1 (auto) | Ultralytics fits the largest batch your VRAM allows — more stable batch statistics than guessing a fixed number |
| `optimizer` | AdamW | Converges faster and more stably than SGD for fine-tuning a pretrained backbone on a domain-shifted dataset (surveillance vs. COCO's photos) |
| `lr0` / `lrf` / `cos_lr` | 0.001 / 0.01 / true | Cosine decay from 0.001 down to 0.00001 — smooth annealing avoids the loss spikes step schedules can cause late in training |
| `warmup_epochs` | 5 | Prevents the freshly-attached detection head from destabilizing the pretrained backbone in the first few epochs |
| `momentum` / `weight_decay` | 0.937 / 0.0005 | Ultralytics' well-validated defaults for YOLO11 — no reason to deviate without evidence of over/under-fitting |
| `label_smoothing` | 0.0 | Label smoothing mainly helps multi-class *classification* by softening confusion between classes; with a single class there's nothing to separate, so it's a no-op here |
| `box` / `cls` / `dfl` | 7.5 / 0.5 / 1.5 | `box` (localization) weighted up relative to Ultralytics' multi-class defaults, since tight boxes on small/occluded people matter more here than the (trivial, single-class) classification decision |
| `amp` | true | Mixed precision — ~2x throughput, lets you fit a larger batch/imgsz on the same GPU |
| `close_mosaic` | 15 | See augmentation section above |
| `multi_scale` | true | Randomly resizes ±50% per batch — robustness to the wide variety of camera resolutions real CCTV rigs use |
| EMA | always on (Ultralytics default) | Exponential moving average of weights for the final checkpoint — measurably more stable validation mAP than the raw last-step weights |

**Inference-time hyperparameters** (`config.yaml` → `inference`):
`conf_threshold=0.25` (default Ultralytics choice, high-recall side which
matters for a security use case — a missed person is worse than an extra
low-confidence box the identity-tracking layer can filter downstream),
`iou_threshold=0.45` (NMS threshold — kept moderate since CrowdHuman-trained
models need slightly looser NMS to avoid suppressing correct boxes on
genuinely overlapping/crowded people), `max_det=300` (generous cap for
dense-crowd frames).

---

## 7. Evaluation (`evaluate.py`)

Runs Ultralytics' validator against **both** val and test splits and prints:
precision, recall, F1, mAP50, mAP50-95 — plus PR curve, F1 curve, and
confusion matrix plots (written by Ultralytics automatically to
`training/runs/eval/<split>/`). Also independently times single-image
inference (50-sample average + P95) to report mean latency and FPS on your
actual hardware, since Ultralytics' own reported speed can vary by
batch/warmup assumptions.

```bash
python training/evaluate.py --weights training/runs/surveillance_human_detector/weights/best.pt
```

---

## 8. Benchmark vs. stock YOLO11 (`benchmark.py`)

Runs **both** `yolo11s.pt` and your trained model over the same test split
and reports precision/recall/small-object-recall sliced into four
conditions, plus FPS:

- **all** — the whole test split
- **crowded** — images with ≥8 ground-truth people
- **low_light** — images with mean grayscale brightness < 60
- **occluded** — images containing at least one pair of ground-truth boxes with IoU > 0.3

```bash
python training/benchmark.py --custom-weights training/runs/surveillance_human_detector/weights/best.pt
```

---

## 9. Export & deployment (`scripts/export_model.py`)

Exports `best.pt` to ONNX and TorchScript, and — unless `--no-deploy` is
passed — copies `best.pt` to `custom_human_model.pt` at the **repo root**.

```bash
python training/scripts/export_model.py --weights training/runs/surveillance_human_detector/weights/best.pt
```

`main.py` auto-detects `custom_human_model.pt` at the repo root and uses it
by default; if it isn't present, it falls back to `yolo11s.pt`. **No code
change is required** — this is the "replace the model path" requirement,
satisfied by dropping the exported file in place. To force the stock model
even when a custom one exists: `python main.py --model yolo11s.pt`.

---

## 10. Performance notes

- `amp: true` in training and `half: true` in `config.yaml` → `inference`
  give FP16 throughout, roughly halving inference memory/latency versus
  FP32 with no measurable accuracy loss on modern GPUs.
- `resolve_device()` in `scripts/common.py` auto-picks CUDA → Apple
  Silicon MPS → CPU, so the same scripts run unmodified on an NVIDIA
  workstation or a Mac.
- `imgsz=960` is a deliberate speed/accuracy tradeoff point for this task;
  drop to 640 if you need more FPS headroom and can accept reduced
  small-person recall.

---

## 11. Fast path (hackathon time budget)

The full recipe above is the "do it right" version. If you're short on
time or GPU budget:

1. Skip MOT17/MOT20 and Open Images — CrowdHuman + WiderPerson + CityPersons
   + COCO-person already cover crowd, occlusion, street-level, and pose
   diversity, and are the fastest to download/convert.
2. Set `epochs: 60`, `patience: 15` in `config.yaml`.
3. Skip `augment_weather.py` — Ultralytics' native mosaic/hsv/copy-paste
   augmentation alone still meaningfully beats stock `yolo11s.pt` on
   surveillance footage; weather synthesis is the highest-effort, lowest-
   marginal-return step here.
4. Start from `yolo11n.pt` instead of `yolo11s.pt` in `config.yaml` if you
   need to train and iterate on a laptop CPU/weak GPU — swap back to `s`
   for the final submission run once you have GPU access.
