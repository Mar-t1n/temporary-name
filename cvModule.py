"""
Facial analysis module — exposes a FaceAnalyzer class for use by app.py
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import urllib.request
import os
import time

# ============ CONFIG ============
UPDATE_INTERVAL = 2.0
EMOTION_THRESHOLD = 0.55
EMOTION_MARGIN = 0.15
MAX_FACES = 5
CAPTURE_PADDING = 0.4
CAPTURE_DIR = "captures"

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = "face_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading face landmarker model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

os.makedirs(CAPTURE_DIR, exist_ok=True)

# ============ LANDMARKS ============
L_EYE_INNER, L_EYE_OUTER = 133, 33
R_EYE_INNER, R_EYE_OUTER = 362, 263
FOREHEAD_TOP, BROW_LINE = 10, 9
NOSE_BASE, CHIN_BOTTOM = 2, 152
FACE_LEFT, FACE_RIGHT = 234, 454
LIP_TOP, LIP_BOTTOM = 13, 14
LIP_LEFT, LIP_RIGHT = 61, 291
NOSE_LEFT, NOSE_RIGHT = 129, 358
NOSE_TIP = 1
JAW_LEFT, JAW_RIGHT = 172, 397

FACE_OVAL = [(10,338),(338,297),(297,332),(332,284),(284,251),(251,389),(389,356),
             (356,454),(454,323),(323,361),(361,288),(288,397),(397,365),(365,379),
             (379,378),(378,400),(400,377),(377,152),(152,148),(148,176),(176,149),
             (149,150),(150,136),(136,172),(172,58),(58,132),(132,93),(93,234),
             (234,127),(127,162),(162,21),(21,54),(54,103),(103,67),(67,109),(109,10)]
LEFT_EYE = [(33,7),(7,163),(163,144),(144,145),(145,153),(153,154),(154,155),(155,133),
            (33,246),(246,161),(161,160),(160,159),(159,158),(158,157),(157,173),(173,133)]
RIGHT_EYE = [(263,249),(249,390),(390,373),(373,374),(374,380),(380,381),(381,382),(382,362),
             (263,466),(466,388),(388,387),(387,386),(386,385),(385,384),(384,398),(398,362)]
LEFT_BROW = [(70,63),(63,105),(105,66),(66,107),(55,65),(65,52),(52,53),(53,46)]
RIGHT_BROW = [(300,293),(293,334),(334,296),(296,336),(285,295),(295,282),(282,283),(283,276)]
LIPS_OUTER = [(61,146),(146,91),(91,181),(181,84),(84,17),(17,314),(314,405),(405,321),
              (321,375),(375,291),(61,185),(185,40),(40,39),(39,37),(37,0),(0,267),
              (267,269),(269,270),(270,409),(409,291)]
LIPS_INNER = [(78,95),(95,88),(88,178),(178,87),(87,14),(14,317),(317,402),(402,318),
              (318,324),(324,308),(78,191),(191,80),(80,81),(81,82),(82,13),(13,312),
              (312,311),(311,310),(310,415),(415,308)]
NOSE_BRIDGE = [(168,6),(6,197),(197,195),(195,5),(5,4),(4,1),(1,19),(19,94),(94,2)]

ALL_CONNECTIONS = (FACE_OVAL + LEFT_EYE + RIGHT_EYE + LEFT_BROW + RIGHT_BROW
                   + LIPS_OUTER + LIPS_INNER + NOSE_BRIDGE)


# ============ HELPERS ============
def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def to_px(lm, w, h):
    return (int(lm.x * w), int(lm.y * h))


def face_bbox(landmarks, w, h, pad=0.0):
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    if pad > 0:
        bw, bh = x2 - x1, y2 - y1
        x1 -= bw * pad; x2 += bw * pad
        y1 -= bh * pad; y2 += bh * pad
    return (max(0, int(x1)), max(0, int(y1)),
            min(w, int(x2)), min(h, int(y2)))


def order_faces(face_landmarks_list, w, h):
    centers = []
    for i, lm in enumerate(face_landmarks_list):
        x1, _, x2, _ = face_bbox(lm, w, h)
        centers.append((i, (x1 + x2) / 2))
    centers.sort(key=lambda t: t[1])
    return [i for i, _ in centers]


# ============ METRICS ============
def compute_metrics(landmarks, w, h):
    pts = [to_px(lm, w, h) for lm in landmarks]
    m = {}

    upper = dist(pts[FOREHEAD_TOP], pts[BROW_LINE])
    middle = dist(pts[BROW_LINE], pts[NOSE_BASE])
    lower = dist(pts[NOSE_BASE], pts[CHIN_BOTTOM])
    total = upper + middle + lower
    if total > 0:
        m["facial_thirds"] = (round(upper/total*100, 1),
                              round(middle/total*100, 1),
                              round(lower/total*100, 1))

    face_w = dist(pts[FACE_LEFT], pts[FACE_RIGHT])
    face_h = dist(pts[FOREHEAD_TOP], pts[CHIN_BOTTOM])
    m["fwhr"] = round(face_w / face_h, 2) if face_h else 0

    lip_h = dist(pts[LIP_TOP], pts[LIP_BOTTOM])
    lip_w = dist(pts[LIP_LEFT], pts[LIP_RIGHT])
    m["lip_ratio"] = round(lip_h / lip_w, 2) if lip_w else 0

    eye_gap = dist(pts[L_EYE_INNER], pts[R_EYE_INNER])
    m["eye_spacing"] = round(eye_gap / face_w, 2) if face_w else 0

    nose_w = dist(pts[NOSE_LEFT], pts[NOSE_RIGHT])
    m["nose_width_ratio"] = round(nose_w / face_w, 2) if face_w else 0

    jaw_w = dist(pts[JAW_LEFT], pts[JAW_RIGHT])
    m["jaw_ratio"] = round(jaw_w / face_w, 2) if face_w else 0

    nose_x = pts[NOSE_TIP][0]
    pairs = [(L_EYE_OUTER, R_EYE_OUTER), (LIP_LEFT, LIP_RIGHT), (JAW_LEFT, JAW_RIGHT)]
    asym = np.mean([abs(abs(pts[l][0] - nose_x) - abs(pts[r][0] - nose_x)) for l, r in pairs])
    m["symmetry"] = round(max(0, 100 - asym), 1)

    return m, pts


def emotion_from_blendshapes(blendshapes):
    if not blendshapes:
        return "neutral", 0.0
    bs = {b.category_name: b.score for b in blendshapes}

    smile = min(bs.get("mouthSmileLeft", 0), bs.get("mouthSmileRight", 0))
    cheek = min(bs.get("cheekSquintLeft", 0), bs.get("cheekSquintRight", 0))
    frown = min(bs.get("mouthFrownLeft", 0), bs.get("mouthFrownRight", 0))
    brow_down = min(bs.get("browDownLeft", 0), bs.get("browDownRight", 0))
    sneer = min(bs.get("noseSneerLeft", 0), bs.get("noseSneerRight", 0))
    upper_up = min(bs.get("mouthUpperUpLeft", 0), bs.get("mouthUpperUpRight", 0))
    eye_wide = min(bs.get("eyeWideLeft", 0), bs.get("eyeWideRight", 0))
    jaw_open = bs.get("jawOpen", 0)
    brow_inner = bs.get("browInnerUp", 0)

    scores = {
        "happy":     smile + cheek * 1.5 if smile > 0.3 else 0,
        "sad":       frown + brow_inner * 0.4 if frown > 0.3 else 0,
        "surprised": (eye_wide + brow_inner) if (eye_wide > 0.4 and jaw_open > 0.3) else 0,
        "angry":     brow_down + sneer * 0.5 if brow_down > 0.4 else 0,
        "disgusted": sneer + upper_up * 0.5 if sneer > 0.4 else 0,
    }

    sorted_scores = sorted(scores.values(), reverse=True)
    top = max(scores, key=lambda k: scores[k])
    if sorted_scores[0] >= EMOTION_THRESHOLD and (sorted_scores[0] - sorted_scores[1]) >= EMOTION_MARGIN:
        return top, round(sorted_scores[0], 2)
    return "neutral", 0.0


# ============ DRAWING ============
def draw_mesh(frame, pts, mode="neutral"):
    """Sick-looking mesh that recolors based on mode."""
    overlay = frame.copy()

    palettes = {
        "neutral": [(0, 255, 220), (0, 200, 255)],   # cyan
        "glaze":   [(255, 200, 100), (255, 150, 220)], # pink/gold
        "hate":    [(80, 80, 255), (50, 50, 200)],     # red
    }
    c1, c2 = palettes.get(mode, palettes["neutral"])

    # Gradient effect: alternate colors along connections
    for i, (a, b) in enumerate(ALL_CONNECTIONS):
        color = c1 if i % 2 == 0 else c2
        cv2.line(overlay, pts[a], pts[b], color, 1, cv2.LINE_AA)

    # Glow on key landmarks
    key = [L_EYE_INNER, L_EYE_OUTER, R_EYE_INNER, R_EYE_OUTER, NOSE_TIP, CHIN_BOTTOM]
    for idx in key:
        cv2.circle(overlay, pts[idx], 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(overlay, pts[idx], 6, c1, 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    return frame


def draw_face_box(frame, landmarks, locked, label):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = face_bbox(landmarks, w, h)
    color = (0, 255, 100) if locked else (140, 140, 140)
    thick = 2 if locked else 1
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick, cv2.LINE_AA)
    cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


# ============ SMOOTHING ============
class MetricSmoother:
    def __init__(self):
        self.numeric_buf = {}
        self.thirds_buf = []
        self.emotion_buf = []
        self.committed_metrics = {}
        self.committed_emotion = ("neutral", 0.0)
        self.last_update = time.time()

    def reset(self):
        self.numeric_buf.clear()
        self.thirds_buf.clear()
        self.emotion_buf.clear()
        self.committed_metrics = {}
        self.committed_emotion = ("neutral", 0.0)

    def add(self, metrics, emotion):
        for k, v in metrics.items():
            if k == "facial_thirds":
                self.thirds_buf.append(v)
            elif isinstance(v, (int, float)):
                self.numeric_buf.setdefault(k, []).append(v)
        self.emotion_buf.append(emotion)

    def commit(self):
        out = {}
        for k, vs in self.numeric_buf.items():
            if vs:
                out[k] = round(float(np.mean(vs)), 2)
        if self.thirds_buf:
            arr = np.array(self.thirds_buf)
            out["facial_thirds"] = (round(float(arr[:, 0].mean()), 1),
                                    round(float(arr[:, 1].mean()), 1),
                                    round(float(arr[:, 2].mean()), 1))
        if self.emotion_buf:
            non_neutral = [e for e in self.emotion_buf if e[0] != "neutral"]
            if len(non_neutral) > len(self.emotion_buf) * 0.5:
                names = [e[0] for e in non_neutral]
                winner = max(set(names), key=names.count)
                avg_conf = np.mean([e[1] for e in non_neutral if e[0] == winner])
                self.committed_emotion = (winner, round(float(avg_conf), 2))
            else:
                self.committed_emotion = ("neutral", 0.0)

        self.committed_metrics = out
        self.numeric_buf.clear()
        self.thirds_buf.clear()
        self.emotion_buf.clear()
        self.last_update = time.time()


# ============ ANALYZER ============
class FaceAnalyzer:
    """High-level interface for the Flask app."""

    def __init__(self):
        base = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=MAX_FACES,
            output_face_blendshapes=True,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self.smoother = MetricSmoother()
        self.locked_idx = 0
        self.start = time.time()
        self.last_clean_frame = None
        self.last_locked_landmarks = None
        self.mode = "neutral"   # neutral | glaze | hate

    def set_mode(self, mode):
        if mode in ("neutral", "glaze", "hate"):
            self.mode = mode

    def cycle_face(self):
        """Move to next detected face; reset smoothing."""
        self.locked_idx += 1
        self.smoother.reset()

    def process_frame(self, frame):
        """Run detection on one frame, return annotated frame."""
        h, w = frame.shape[:2]
        self.last_clean_frame = frame.copy()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int((time.time() - self.start) * 1000)
        result = self.landmarker.detect_for_video(mp_image, ts_ms)

        faces = result.face_landmarks or []
        blendshapes = result.face_blendshapes or []
        ordered = order_faces(faces, w, h) if faces else []

        if ordered:
            self.locked_idx %= len(ordered)
        else:
            self.locked_idx = 0

        for rank, real_idx in enumerate(ordered):
            lm = faces[real_idx]
            is_locked = (rank == self.locked_idx)
            draw_face_box(frame, lm, is_locked, f"#{rank + 1}")

        if ordered:
            real_locked = ordered[self.locked_idx]
            locked_lm = faces[real_locked]
            self.last_locked_landmarks = locked_lm
            metrics, pts = compute_metrics(locked_lm, w, h)
            frame = draw_mesh(frame, pts, mode=self.mode)
            emotion = ("neutral", 0.0)
            if real_locked < len(blendshapes):
                emotion = emotion_from_blendshapes(blendshapes[real_locked])
            self.smoother.add(metrics, emotion)

        if time.time() - self.smoother.last_update >= UPDATE_INTERVAL:
            self.smoother.commit()

        return frame, len(ordered)

    def get_state(self):
        emo, conf = self.smoother.committed_emotion
        return {
            "metrics": self.smoother.committed_metrics,
            "emotion": emo,
            "confidence": conf,
            "mode": self.mode,
        }

    def capture(self):
        """Save a clean cropped face. Returns (path, jpeg_bytes) or (None, None)."""
        if self.last_clean_frame is None or self.last_locked_landmarks is None:
            return None, None
        h, w = self.last_clean_frame.shape[:2]
        x1, y1, x2, y2 = face_bbox(self.last_locked_landmarks, w, h, pad=CAPTURE_PADDING)
        crop = self.last_clean_frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None, None
        path = os.path.join(CAPTURE_DIR, f"face_{int(time.time())}.jpg")
        cv2.imwrite(path, crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return path, (buf.tobytes() if ok else None)

    def close(self):
        self.landmarker.close()