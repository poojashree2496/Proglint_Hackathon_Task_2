"""
Persistent Person Tracker engine.

Pipeline: YOLO11 (person-only detection) -> BoT-SORT (short-term
frame-to-frame association) -> a persistent identity layer on top that:

  * keeps a per-person appearance gallery and re-identifies anyone who
    was lost (occluded, walked off-screen, crossed another person) and
    reappears, instead of handing them a new id,
  * runs a Kalman filter per person for smooth boxes and motion
    prediction through occlusion,
  * detects and skips appearance updates while two people overlap
    (the crop would otherwise mix both people's colors and corrupt the
    gallery), and
  * self-heals the rare case where BoT-SORT's own internal id swaps
    between two crossing people, instead of silently propagating the
    swapped identity forever.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

import cv2
import numpy as np
from ultralytics import YOLO


NEW, ACTIVE, OCCLUDED, LOST = "NEW", "ACTIVE", "OCCLUDED", "LOST"

PERSON_CLASS_ID = 0  # COCO class 0 == "person". Nothing else is ever tracked.


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------

def xyxy_to_cxcywh(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1], dtype=np.float32)


def cxcywh_to_xyxy(state: np.ndarray) -> np.ndarray:
    cx, cy, w, h = state
    w = max(1.0, float(w))
    h = max(1.0, float(h))
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=np.float32)


def iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return float(inter_area / union)


def center_distance_norm(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Center distance normalized by the average box diagonal (scale-invariant)."""
    ca = xyxy_to_cxcywh(box_a)
    cb = xyxy_to_cxcywh(box_b)
    dist = float(np.hypot(ca[0] - cb[0], ca[1] - cb[1]))
    scale = 0.5 * (np.hypot(ca[2], ca[3]) + np.hypot(cb[2], cb[3]))
    return dist / max(1.0, scale)


# --------------------------------------------------------------------------
# Assignment: a compact, dependency-free Hungarian (Kuhn-Munkres) algorithm.
# scipy is not a guaranteed dependency here, so this is implemented directly.
# --------------------------------------------------------------------------

def linear_sum_assignment_min(cost: np.ndarray) -> Tuple[List[int], List[int]]:
    """Minimum-cost bipartite assignment. Returns (row_indices, col_indices).

    Pads non-square inputs with a large constant so every row gets an
    assignment; callers are responsible for discarding pairs whose cost
    lands on that padding value.
    """
    if cost.size == 0:
        return [], []

    n_rows, n_cols = cost.shape
    n = max(n_rows, n_cols)
    pad_value = float(np.max(cost)) + 1e6 if cost.size else 1e6
    padded = np.full((n, n), pad_value, dtype=np.float64)
    padded[:n_rows, :n_cols] = cost

    INF = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, n + 1):
                if not used[j]:
                    cur = padded[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    result_col_for_row = [-1] * n
    for j in range(1, n + 1):
        if p[j] != 0:
            result_col_for_row[p[j] - 1] = j - 1

    row_ind = list(range(n_rows))
    col_ind = [result_col_for_row[r] for r in range(n_rows)]
    return row_ind, col_ind


# --------------------------------------------------------------------------
# Motion model: constant-velocity Kalman filter over [cx, cy, w, h]
# --------------------------------------------------------------------------

class BoxKalmanFilter:
    def __init__(self, box: np.ndarray):
        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.transitionMatrix = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.kf.transitionMatrix[i, i + 4] = 1.0

        self.kf.measurementMatrix = np.zeros((4, 8), dtype=np.float32)
        for i in range(4):
            self.kf.measurementMatrix[i, i] = 1.0

        self.kf.processNoiseCov = np.eye(8, dtype=np.float32)
        self.kf.processNoiseCov[4:, 4:] *= 0.02   # velocities drift slowly
        self.kf.processNoiseCov[:4, :4] *= 0.5

        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 2.0
        self.kf.errorCovPost = np.eye(8, dtype=np.float32) * 10.0

        cx, cy, w, h = xyxy_to_cxcywh(box)
        self.kf.statePost = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32).reshape(8, 1)

    def predict(self) -> np.ndarray:
        state = self.kf.predict()
        return cxcywh_to_xyxy(state[:4].flatten())

    def update(self, box: np.ndarray) -> np.ndarray:
        cx, cy, w, h = xyxy_to_cxcywh(box)
        measurement = np.array([cx, cy, w, h], dtype=np.float32).reshape(4, 1)
        state = self.kf.correct(measurement)
        return cxcywh_to_xyxy(state[:4].flatten())

    def current_box(self) -> np.ndarray:
        return cxcywh_to_xyxy(self.kf.statePost[:4].flatten())


# --------------------------------------------------------------------------
# Appearance features
# --------------------------------------------------------------------------

def extract_feature(frame, box) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = map(int, box)
    height, width = frame.shape[:2]

    x1 = max(0, x1)
    x2 = min(width, x2)
    y1 = max(0, y1)
    y2 = min(height, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    ch, cw = crop.shape[:2]
    margin_x = max(1, int(cw * 0.10))
    margin_y = max(1, int(ch * 0.08))
    crop = crop[margin_y:max(margin_y + 1, ch - margin_y), margin_x:max(margin_x + 1, cw - margin_x)]
    crop = cv2.resize(crop, (64, 128), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    parts = []
    for channel, bins, limit in [(0, 16, 180), (1, 16, 256), (2, 16, 256)]:
        hist = cv2.calcHist([hsv], [channel], None, [bins], [0, limit])
        hist = cv2.normalize(hist, hist).flatten()
        parts.append(hist)

    # Coarse 2x2 spatial grid (head/torso/legs-ish split) so two people
    # wearing the same-colored shirt but different pants still separate.
    for row in range(2):
        for col in range(2):
            area = hsv[row * 64:(row + 1) * 64, col * 32:(col + 1) * 32]
            parts.extend([
                np.array([area[:, :, 0].mean() / 180.0], dtype=np.float32),
                np.array([area[:, :, 1].mean() / 255.0], dtype=np.float32),
                np.array([area[:, :, 2].mean() / 255.0], dtype=np.float32),
                np.array([area[:, :, 2].std() / 255.0], dtype=np.float32),
            ])

    vector = np.concatenate(parts).astype(np.float32)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / (norm + 1e-8)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8))


# --------------------------------------------------------------------------
# Track state
# --------------------------------------------------------------------------

@dataclass
class Track:
    pid: int
    feature: Optional[np.ndarray]
    kalman: BoxKalmanFilter
    samples: list = field(default_factory=list)
    box: Optional[np.ndarray] = None
    last_seen: float = 0.0
    first_seen: float = 0.0
    state: str = NEW
    hits: int = 0
    frames_alive: int = 0
    lost_events: int = 0
    reassoc_count: int = 0
    consecutive_hits: int = 0
    locked: bool = False
    confidence: float = 0.5
    switch_flag: bool = False
    low_sim_streak: int = 0

    def quality(self) -> float:
        continuity = self.hits / max(1, self.frames_alive)
        stability = 1.0 - min(1.0, self.reassoc_count / 5.0)
        penalty = 1.0 - min(1.0, self.lost_events / 10.0)
        score = 0.5 * continuity + 0.3 * stability + 0.2 * penalty
        return max(0.0, min(1.0, score))

    def best_gallery_score(self, feature: np.ndarray) -> float:
        candidates = ([self.feature] if self.feature is not None else []) + self.samples
        candidates = [c for c in candidates if c is not None and c.size == feature.size]
        if not candidates:
            return -1.0
        return max(cosine_similarity(feature, c) for c in candidates)


# --------------------------------------------------------------------------
# The tracker
# --------------------------------------------------------------------------

class PersistentPersonTracker:
    def __init__(
        self,
        model_name="yolo11s.pt",
        lost_memory_seconds=30.0,
        occlusion_draw_seconds=2.5,
        reid_threshold=0.86,
        reid_threshold_relaxed=0.78,
        switch_threshold=0.42,
        demote_after_frames=4,
        lock_after_hits=15,
        max_samples=8,
        smoothing=0.6,
        crowd_iou_thresh=0.20,
    ):
        self.model = YOLO(model_name)

        # How long a lost/occluded identity stays eligible for re-id
        # recovery — this is what lets someone who leaves the frame and
        # walks back in minutes-scale-short-of get their original id back.
        self.lost_memory_seconds = lost_memory_seconds
        # How long we keep drawing a motion-predicted "ghost" box during
        # occlusion before hiding it (identity memory itself lasts longer,
        # governed by lost_memory_seconds).
        self.occlusion_draw_seconds = occlusion_draw_seconds

        self.reid_threshold = reid_threshold
        self.reid_threshold_relaxed = reid_threshold_relaxed
        self.switch_threshold = switch_threshold
        self.demote_after_frames = demote_after_frames
        self.lock_after_hits = lock_after_hits
        self.max_samples = max_samples
        self.smoothing = smoothing
        self.crowd_iou_thresh = crowd_iou_thresh

        self.next_pid = 1
        self.tracker_to_pid: Dict[int, int] = {}
        self.tracks: Dict[int, Track] = {}
        self.total_reassociations = 0

    # ---------------------------------------------------------------- utils

    def make_new_track(self, feature, box, now) -> int:
        person_id = self.next_pid
        self.next_pid += 1

        self.tracks[person_id] = Track(
            pid=person_id,
            feature=feature.copy() if feature is not None else None,
            kalman=BoxKalmanFilter(box),
            samples=[feature.copy()] if feature is not None else [],
            box=box.copy(),
            last_seen=now,
            first_seen=now,
            state=NEW,
        )
        return person_id

    def update_track_appearance(self, track: Track, feature: Optional[np.ndarray]):
        if feature is None:
            return

        if track.feature is None:
            track.feature = feature.copy()
        else:
            track.feature = (0.8 * track.feature + 0.2 * feature).astype(np.float32)
            track.feature /= (np.linalg.norm(track.feature) + 1e-8)

        track.samples.append(feature.copy())
        if len(track.samples) > self.max_samples:
            track.samples.pop(0)

    # ----------------------------------------------------------- detections

    @staticmethod
    def _compute_crowding(boxes: List[np.ndarray], crowd_iou_thresh: float) -> List[bool]:
        n = len(boxes)
        crowded = [False] * n
        for i in range(n):
            for j in range(i + 1, n):
                if iou(boxes[i], boxes[j]) >= crowd_iou_thresh:
                    crowded[i] = True
                    crowded[j] = True
        return crowded

    def _finalize_match(self, track: Track, box, feature, confidence, now, crowded):
        """Apply a confirmed detection (continuing or re-identified) to a track."""
        track.box = track.kalman.update(box)
        track.last_seen = now
        track.state = ACTIVE
        track.hits += 1
        track.consecutive_hits += 1

        if not track.locked and track.consecutive_hits >= self.lock_after_hits:
            track.locked = True

        if not crowded:
            self.update_track_appearance(track, feature)

        stability = 1.0 if not track.switch_flag else 0.4
        lock_bonus = 0.05 if track.locked else 0.0
        track.confidence = max(0.0, min(1.0, 0.6 * float(confidence) + 0.4 * stability + lock_bonus))

    def _process_continuing(self, tracker_id, feature, box, confidence, now, crowded, used_ids) -> bool:
        """Try the fast path: tracker_id already maps to a live persistent id.

        Returns True if handled (matched or demoted-and-requeued is signalled
        by returning False so the caller treats it as an orphan detection).
        """
        person_id = self.tracker_to_pid.get(tracker_id)
        if person_id is None or person_id not in self.tracks:
            return False

        track = self.tracks[person_id]
        track.switch_flag = False

        if not crowded and feature is not None and track.feature is not None and track.hits > 3:
            match_score = cosine_similarity(feature, track.feature)
            if match_score < self.switch_threshold:
                track.low_sim_streak += 1
            else:
                track.low_sim_streak = 0

            if track.low_sim_streak >= self.demote_after_frames:
                # Sustained appearance mismatch on a "known" tracker_id
                # almost always means BoT-SORT's internal id got swapped
                # onto a different person during a crossing. Rather than
                # keep forcing the wrong identity, release this mapping so
                # the detection is re-evaluated as an orphan (it will
                # either re-id to its true owner or spawn a fresh id), and
                # push the old identity into the lost pool so its real
                # owner can still recover it later.
                track.low_sim_streak = 0
                track.state = OCCLUDED
                track.lost_events += 1
                del self.tracker_to_pid[tracker_id]
                return False

            track.switch_flag = track.low_sim_streak > 0

        self._finalize_match(track, box, feature, confidence, now, crowded)
        used_ids.add(person_id)
        return True

    def _reid_orphans(self, orphans, now, used_ids):
        """Batch re-identification: match freshly-seen detections (new
        tracker_ids, or ones just demoted for a suspected swap) against the
        pool of recently lost/occluded identities using the Hungarian
        algorithm, instead of a greedy first-come-first-served scan. This
        matters when several people re-enter/uncross in the same frame:
        greedy matching can grab a locally-best-but-globally-wrong pairing.
        """
        pool_ids = [
            pid for pid, track in self.tracks.items()
            if pid not in used_ids
            and track.state in (OCCLUDED, LOST)
            and now - track.last_seen <= self.lost_memory_seconds
        ]

        assigned = {}
        if orphans and pool_ids:
            cost = np.zeros((len(orphans), len(pool_ids)), dtype=np.float64)
            valid = np.zeros((len(orphans), len(pool_ids)), dtype=bool)

            for oi, orphan in enumerate(orphans):
                feature = orphan["feature"]
                box = orphan["box"]
                for pj, pid in enumerate(pool_ids):
                    track = self.tracks[pid]
                    if feature is None:
                        cost[oi, pj] = 1e5
                        continue

                    sim = track.best_gallery_score(feature)
                    age = now - track.last_seen
                    age_penalty = 0.035 * min(age / self.lost_memory_seconds, 1.0)
                    predicted_box = track.kalman.current_box()
                    spatial_gate_ok = age <= self.occlusion_draw_seconds and iou(predicted_box, box) > 0.05

                    effective_threshold = (
                        self.reid_threshold_relaxed if spatial_gate_ok else self.reid_threshold
                    )
                    score = sim - age_penalty
                    if score >= effective_threshold:
                        valid[oi, pj] = True
                        cost[oi, pj] = 1.0 - score
                    else:
                        cost[oi, pj] = 1e5 + (1.0 - max(sim, 0.0))

            row_ind, col_ind = linear_sum_assignment_min(cost)
            for r, c in zip(row_ind, col_ind):
                if c < 0 or c >= len(pool_ids):
                    continue
                if not valid[r, c]:
                    continue
                assigned[r] = pool_ids[c]

        for oi, orphan in enumerate(orphans):
            tracker_id = orphan["tracker_id"]
            box = orphan["box"]
            feature = orphan["feature"]
            confidence = orphan["confidence"]
            crowded = orphan["crowded"]

            if oi in assigned:
                person_id = assigned[oi]
                track = self.tracks[person_id]
                track.reassoc_count += 1
                track.consecutive_hits = 0
                track.switch_flag = False
                track.low_sim_streak = 0
                self.total_reassociations += 1
            else:
                person_id = self.make_new_track(feature, box, now)
                track = self.tracks[person_id]

            self.tracker_to_pid[tracker_id] = person_id
            self._finalize_match(track, box, feature, confidence, now, crowded)
            used_ids.add(person_id)

    # ------------------------------------------------------------- lifecycle

    def mark_missing_tracks(self, seen_ids, now):
        for person_id, track in self.tracks.items():
            if person_id in seen_ids:
                continue

            gap = now - track.last_seen
            if track.state == ACTIVE:
                track.state = OCCLUDED
                track.lost_events += 1
            elif track.state == OCCLUDED and gap > self.lost_memory_seconds:
                track.state = LOST

    def tick_frame_count(self):
        for track in self.tracks.values():
            track.frames_alive += 1
            if track.state in (OCCLUDED, LOST):
                # Keep the motion model rolling forward through occlusion
                # so a reappearing detection can be spatially gated
                # against a plausible predicted position, not a stale one.
                track.kalman.predict()

    # ------------------------------------------------------------- drawing

    @staticmethod
    def status_symbol(track: Track) -> str:
        if track.switch_flag:
            return "?"
        if track.locked:
            return "✓"
        return "~"

    def draw_label(self, frame, track: Track, box: Optional[np.ndarray] = None, ghost: bool = False):
        if box is None:
            box = track.box
        x1, y1, x2, y2 = map(int, box)
        height, width = frame.shape[:2]

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.48, min(0.78, width / 2200.0))
        thickness = max(1, int(round(scale * 2)))

        label = f"ID {track.pid} {self.status_symbol(track)}"
        (text_width, text_height), baseline = cv2.getTextSize(label, font, scale, thickness)
        center_x = (x1 + x2) // 2
        label_x = max(0, min(width - text_width - 10, center_x - text_width // 2))
        label_y = max(text_height + baseline + 6, y1 - 8)

        box_color = (255, 255, 255)
        if track.switch_flag:
            box_color = (0, 165, 255)
        elif not track.locked:
            box_color = (0, 220, 255)
        if ghost:
            box_color = (0, 0, 255)

        cv2.rectangle(
            frame,
            (label_x - 5, label_y - text_height - baseline - 4),
            (label_x + text_width + 5, label_y + 2),
            (20, 20, 20),
            -1,
        )
        cv2.putText(
            frame,
            label,
            (label_x, label_y - 2),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

        if ghost:
            self._draw_dashed_rect(frame, (x1, y1), (x2, y2), box_color, max(2, thickness))
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, max(2, thickness))

    @staticmethod
    def _draw_dashed_rect(frame, pt1, pt2, color, thickness, dash_len=10):
        x1, y1 = pt1
        x2, y2 = pt2
        for (sx, sy, ex, ey) in [(x1, y1, x2, y1), (x1, y2, x2, y2), (x1, y1, x1, y2), (x2, y1, x2, y2)]:
            length = max(abs(ex - sx), abs(ey - sy))
            if length == 0:
                continue
            steps = max(1, length // dash_len)
            for i in range(0, steps, 2):
                t0 = i / steps
                t1 = min(1.0, (i + 1) / steps)
                p0 = (int(sx + (ex - sx) * t0), int(sy + (ey - sy) * t0))
                p1 = (int(sx + (ex - sx) * t1), int(sy + (ey - sy) * t1))
                cv2.line(frame, p0, p1, color, thickness)

    def draw_summary(self, frame, active_count):
        height, width = frame.shape[:2]
        scale = max(0.42, min(0.6, width / 2400.0))
        text = (
            f"Active People: {active_count}  |  "
            f"Total Unique People: {self.next_pid - 1}  |  "
            f"Re-identification Recoveries: {self.total_reassociations}"
        )

        cv2.putText(frame, text, (10, height - 12), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (10, height - 12), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)

    # ------------------------------------------------------------- main loop

    def process_video(self, video_path: Path, output_path: Path):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video info: {width}x{height} @ {fps:.2f} FPS, {frame_count} frames")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError("Could not create output video.")

        frame_no = 0
        start = time.time()

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                now = frame_no / fps
                self.tick_frame_count()

                result = self.model.track(
                    frame,
                    persist=True,
                    tracker="botsort.yaml",
                    classes=[PERSON_CLASS_ID],  # person-only, always
                    conf=0.25,
                    verbose=False,
                )[0]

                used_ids = set()

                if result.boxes is not None and len(result.boxes):
                    boxes = result.boxes.xyxy.cpu().numpy()
                    scores = (
                        result.boxes.conf.cpu().numpy()
                        if result.boxes.conf is not None
                        else np.ones(len(boxes))
                    )

                    if result.boxes.id is not None:
                        tracker_ids = result.boxes.id.cpu().numpy().astype(int)
                    else:
                        tracker_ids = np.arange(len(boxes)) + 10_000_000

                    keep_mask = scores >= 0.25
                    boxes = boxes[keep_mask]
                    scores = scores[keep_mask]
                    tracker_ids = tracker_ids[keep_mask]

                    crowded_flags = self._compute_crowding(list(boxes), self.crowd_iou_thresh)
                    features = [extract_feature(frame, box) for box in boxes]

                    orphans = []
                    for box, score, tracker_id, feature, crowded in zip(
                        boxes, scores, tracker_ids, features, crowded_flags
                    ):
                        handled = self._process_continuing(
                            int(tracker_id), feature, box, float(score), now, crowded, used_ids
                        )
                        if not handled:
                            orphans.append({
                                "tracker_id": int(tracker_id),
                                "box": box,
                                "feature": feature,
                                "confidence": float(score),
                                "crowded": crowded,
                            })

                    if orphans:
                        self._reid_orphans(orphans, now, used_ids)

                self.mark_missing_tracks(used_ids, now)

                # Draw confirmed detections for this frame.
                for person_id in used_ids:
                    self.draw_label(frame, self.tracks[person_id])

                # Draw a short motion-predicted "ghost" box for anyone who
                # just became occluded, so the label never simply vanishes
                # and reappears with a different id moments later.
                for person_id, track in self.tracks.items():
                    if person_id in used_ids:
                        continue
                    if track.state != OCCLUDED:
                        continue
                    if now - track.last_seen > self.occlusion_draw_seconds:
                        continue
                    predicted_box = track.kalman.current_box()
                    self.draw_label(frame, track, box=predicted_box, ghost=True)

                self.draw_summary(frame, len(used_ids))

                writer.write(frame)
                frame_no += 1

                if frame_no % 100 == 0:
                    elapsed = max(time.time() - start, 1e-6)
                    speed = frame_no / elapsed
                    print(f"\rProcessed {frame_no} frames | {speed:.1f} FPS", end="", flush=True)
        finally:
            cap.release()
            writer.release()

        print()
        print("Done.")
        print(f"Input : {video_path}")
        print(f"Output: {output_path}")
        print(f"Frames: {frame_no}")
        print(f"Total unique IDs assigned: {self.next_pid - 1}")
        print(f"Re-identification recoveries: {self.total_reassociations}")
