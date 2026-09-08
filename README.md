# Persistent Person Tracker

Detects every person in a video, assigns each one a persistent ID, and keeps
that ID stable across occlusion, tracker churn, and re-entry into frame.

The output video is the canvas: person boxes with an `ID N` label above each
head, plus a small one-line summary along the bottom edge (active count,
total IDs assigned, re-identification recoveries). No separate dashboard.

## Pipeline

```
video  ->  YOLO11 person detector  ->  BoT-SORT short-term tracker
        ->  identity engine (appearance re-ID, memory, locking)
        ->  track validator (occlusion / lifecycle / id-switch check)
        ->  draw + write output video
```

## What the identity layer adds on top of raw tracking

- **Persistent identity memory** — every ID keeps a rolling gallery of
  recent appearance samples (HSV color-histogram signature), not just one
  snapshot, so it tolerates lighting/pose changes.
- **Re-identification / re-entry recovery** — when BoT-SORT drops a track
  (occlusion, brief exit from frame) and a new detection appears, the new
  detection is compared against recently-lost identities before a fresh ID
  is minted. A good match restores the old ID instead of creating a new one.
- **Track lifecycle** — each person moves through `NEW -> ACTIVE ->
  OCCLUDED -> LOST` (and back to `ACTIVE` on recovery), so a missed frame
  doesn't immediately kill an identity.
- **Identity locking** — once an ID has enough consecutive high-confidence
  frames, it's marked locked (`✓` on screen) so short-lived detection noise
  can't easily bump it off its identity.
- **ID-switch flagging** — if a tracker ID's live appearance suddenly
  disagrees strongly with its own memory (a likely tracker mix-up), the box
  turns orange with a `?` marker instead of silently trusting the switch.
- **Confidence-aware IDs** — each track carries a live confidence blended
  from detector confidence, appearance stability, and lock state; a
  per-track quality score (continuity, stability, occlusion penalty) is
  tracked internally for anything that needs to reason about track
  reliability.
- **Trajectory smoothing** — boxes are exponentially smoothed frame to
  frame so labels/boxes don't jitter.

## Run

1. Install Python 3.10+.
2. Put a video into `input/`.
3. Run:

```bash
pip install -r requirements.txt
python main.py
```

Or choose a specific video/model:

```bash
python main.py --video input/my_video.mp4 --model yolo11s.pt
```

The first run downloads the YOLO11 model automatically.

Output: `output/tracked_<video-name>.mp4`

## Reading the output

- White box + `✓` = locked, stable identity.
- Yellow box + `~` = still establishing (new or recently re-identified).
- Orange box + `?` = possible ID switch — appearance suddenly disagreed
  with this ID's memory.

No vision system can guarantee identity recovery when visual information is
insufficient — extreme blur, tiny/far-away people, long-term full occlusion,
or genuinely indistinguishable clothing/appearance. The pipeline is designed
to be robust across a range of resolutions, frame rates, durations and
moderate visual degradation, not to guarantee perfect recovery in every case.

## Suggested test footage

- **Short/dense test:** MOT17-12 (MOTChallenge, 1080p/30fps, ~30s, busy
  shopping-mall scene with crossing pedestrians).
- **Long-duration stress test:** PETS 2009 sequences (e.g. S2.L1, ~12.5 min)
  — long multi-target pedestrian footage with people entering/leaving frame.

Watch specifically for people who leave the visible area and re-enter later
— that's the scenario `id_front`/`id_back` occlusion recovery and re-entry
detection are meant to handle.
