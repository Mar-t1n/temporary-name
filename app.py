"""
FaceJudge - desktop app (PyQt6 + OpenCV)
Run: pip install PyQt6 opencv-python mediapipe numpy
     python app.py
"""

import sys
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import time
import math
import urllib.request

from PyQt6.QtCore import Qt, QTimer, QPointF, QRectF, pyqtSignal
from PyQt6.QtGui import (QImage, QPixmap, QPainter, QColor, QFont, QPen,
                         QBrush, QLinearGradient, QRadialGradient, QPainterPath,
                         QFontDatabase)
from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QWidget,
                             QVBoxLayout, QHBoxLayout, QPushButton,
                             QGraphicsDropShadowEffect)

from cvModule import FaceAnalyzer, ALL_CONNECTIONS, to_px

# ============ CONFIG ============
GESTURE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
GESTURE_MODEL_PATH = "gesture_recognizer.task"

GESTURE_HOLD_FRAMES = 12
GESTURE_COOLDOWN = 2.0
MISS_TOLERANCE = 4

MODE_COLORS = {
    "neutral": QColor(0, 255, 220),
    "glaze":   QColor(255, 112, 196),
    "hate":    QColor(255, 80, 80),
}

MODE_CAPTIONS = {
    "neutral": "Hold thumb sideways, up to glaze, down to hate",
    "glaze":   "Glazing in progress...",
    "hate":    "Roasting in progress...",
}

if not os.path.exists(GESTURE_MODEL_PATH):
    print("Downloading gesture recognizer model...")
    urllib.request.urlretrieve(GESTURE_MODEL_URL, GESTURE_MODEL_PATH)


# ============ HAND POSE CLASSIFIER ============
# 21 hand landmarks. Fingers: index(5-8), middle(9-12), ring(13-16), pinky(17-20)
# Thumb: 1-4.  PIP joints (proximal): 6,10,14,18.  Tip: 8,12,16,20.
def classify_thumb_gesture(landmarks):
    """Return 'up', 'down', 'side', or None.

    Requires:
      1) Thumb extended (tip far from MCP)
      2) Other four fingers CURLED (tip below or near PIP joint in y, or close to palm)
      3) Then classifies thumb direction by tip-vs-MCP vector.
    """
    if not landmarks:
        return None

    # Helpers
    def y(i): return landmarks[i].y
    def x(i): return landmarks[i].x
    def dist(a, b):
        return math.hypot(landmarks[a].x - landmarks[b].x,
                          landmarks[a].y - landmarks[b].y)

    wrist = 0
    thumb_mcp, thumb_tip = 2, 4

    # 1) Thumb must be extended away from palm
    palm_size = dist(wrist, 9)  # wrist to middle MCP
    if palm_size < 0.05:
        return None
    if dist(thumb_mcp, thumb_tip) < palm_size * 0.5:
        return None

    # 2) Other four fingers must be curled.
    # A curled finger: the tip is closer to the wrist than its PIP joint is.
    # (For an extended finger, tip is farther from wrist than PIP.)
    finger_pip_tip = [(6, 8), (10, 12), (14, 16), (18, 20)]
    curled_count = 0
    for pip, tip in finger_pip_tip:
        d_pip = dist(wrist, pip)
        d_tip = dist(wrist, tip)
        if d_tip < d_pip * 1.05:  # tip is at or closer than PIP -> finger is curled
            curled_count += 1

    if curled_count < 3:  # allow one finger to be sloppy
        return None

    # 3) Classify thumb direction
    dx = x(thumb_tip) - x(thumb_mcp)
    dy = y(thumb_tip) - y(thumb_mcp)

    if abs(dy) > abs(dx) * 1.4:
        return "down" if dy > 0 else "up"
    if abs(dx) > abs(dy) * 1.2:
        return "side"
    return None


# ============ GESTURE DETECTOR ============
class GestureDetector:
    def __init__(self):
        base = python.BaseOptions(model_asset_path=GESTURE_MODEL_PATH)
        opts = vision.GestureRecognizerOptions(
            base_options=base,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(opts)
        self.start = time.time()
        self.streak_mode = None
        self.streak_count = 0
        self.miss_count = 0
        self.last_switch = 0.0

    def detect(self, rgb_frame):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        ts_ms = int((time.time() - self.start) * 1000)
        result = self.recognizer.recognize_for_video(mp_image, ts_ms)

        if not result.hand_landmarks:
            return None

        # ALWAYS use our custom classifier - it enforces curled fingers AND
        # supports sideways. We don't trust MediaPipe's classifier alone because
        # an open hand with extended thumb can register as Thumb_Up.
        orient = classify_thumb_gesture(result.hand_landmarks[0])
        mapping = {"up": "glaze", "down": "hate", "side": "neutral"}
        return mapping.get(orient) if orient is not None else None

    def update_mode(self, current_mode, target):
        now = time.time()

        if target is None:
            self.miss_count += 1
            if self.miss_count > MISS_TOLERANCE:
                self.streak_mode = None
                self.streak_count = 0
            return current_mode

        self.miss_count = 0

        if target == current_mode and now - self.last_switch < GESTURE_COOLDOWN:
            self.streak_mode = None
            self.streak_count = 0
            return current_mode

        if target == self.streak_mode:
            self.streak_count += 1
        else:
            self.streak_mode = target
            self.streak_count = 1

        if (self.streak_count >= GESTURE_HOLD_FRAMES
                and target != current_mode
                and now - self.last_switch >= GESTURE_COOLDOWN):
            self.last_switch = now
            self.streak_mode = None
            self.streak_count = 0
            return target

        return current_mode

    def progress(self):
        if self.streak_mode is None:
            return 0.0, None
        return min(1.0, self.streak_count / GESTURE_HOLD_FRAMES), self.streak_mode

    def close(self):
        self.recognizer.close()


# ============ HOLOGRAPHIC FACE OVERLAY ============
class MeshRenderer:
    """Sci-fi targeting overlay: angular wireframe, real telemetry, glitch effects."""

    # Indices used as anchor points for telemetry labels
    DATA_ANCHORS = {
        33:  ("L_EYE", "eye_w"),       # left eye outer
        263: ("R_EYE", "eye_w"),       # right eye outer
        1:   ("NOSE", "nose_w"),       # nose tip
        152: ("CHIN", "face_h"),       # chin
        234: ("L_JAW", "jaw_w"),       # left face edge
        454: ("R_JAW", "jaw_w"),       # right face edge
        13:  ("LIP", "lip_r"),         # upper lip
    }

    # Diagonal cross-bracing connections — the "tech" look comes from
    # bracing the wireframe with non-anatomical lines, not from the mesh itself.
    BRACE_LINES = [
        (33, 263),    # eye-to-eye
        (33, 152),    # left eye to chin
        (263, 152),   # right eye to chin
        (234, 454),   # cheek-to-cheek
        (10, 152),    # forehead to chin
        (61, 291),    # mouth corners
    ]

    def __init__(self):
        self.t = 0
        self.particles = []
        self.glitch_timer = 0
        self.last_pts = None

    def _real_telemetry(self, metrics, landmarks):
        """Build dict of {anchor_idx: 'LABEL value'} from real measurements."""
        out = {}
        face_w = metrics.get("fwhr", 0)
        # Eye width in normalized coords (left eye)
        if landmarks:
            l_inner = landmarks[133]; l_outer = landmarks[33]
            r_inner = landmarks[362]; r_outer = landmarks[263]
            l_eye_w = math.hypot(l_inner.x - l_outer.x, l_inner.y - l_outer.y)
            r_eye_w = math.hypot(r_inner.x - r_outer.x, r_inner.y - r_outer.y)
            nose_w_lm = math.hypot(landmarks[129].x - landmarks[358].x,
                                   landmarks[129].y - landmarks[358].y)
        else:
            l_eye_w = r_eye_w = nose_w_lm = 0

        out[33] = f"L.EYE  {l_eye_w*1000:5.1f}"
        out[263] = f"R.EYE  {r_eye_w*1000:5.1f}"
        out[1] = f"NOSE.W {nose_w_lm*1000:5.1f}"
        out[152] = f"FWHR   {metrics.get('fwhr', 0):.2f}"
        out[234] = f"JAW    {metrics.get('jaw_ratio', 0):.2f}"
        out[454] = f"SYM    {metrics.get('symmetry', 0):.1f}%"
        out[13] = f"LIP.R  {metrics.get('lip_ratio', 0):.2f}"
        return out

    def draw(self, painter: QPainter, pts, mode, w, h,
             landmarks=None, metrics=None):
        self.t += 1
        color = MODE_COLORS[mode]
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Face bounding box (for brackets, reticle, scan)
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        fx1, fx2 = min(xs), max(xs)
        fy1, fy2 = min(ys), max(ys)
        face_w = fx2 - fx1; face_h = fy2 - fy1
        cx, cy = (fx1 + fx2) / 2, (fy1 + fy2) / 2

        # Pre-compute z-depth for shading
        if landmarks:
            zs = [lm.z for lm in landmarks]
            z_min, z_max = min(zs), max(zs)
            z_range = max(0.001, z_max - z_min)
        else:
            zs = None

        # ========== LAYER 1: ghost wireframe (motion trail) ==========
        if self.last_pts is not None and len(self.last_pts) == len(pts):
            ghost = QColor(color); ghost.setAlpha(25)
            painter.setPen(QPen(ghost, 0.6))
            for a, b in ALL_CONNECTIONS:
                painter.drawLine(self.last_pts[a][0], self.last_pts[a][1],
                                 self.last_pts[b][0], self.last_pts[b][1])

        # ========== LAYER 2: depth-shaded wireframe (no glow halo) ==========
        # Single-pass thin lines — no fat outer glow, no soft halos.
        # Depth controls alpha so closer features are crisp, far features fade.
        for a, b in ALL_CONNECTIONS:
            if zs is not None:
                z_avg = (zs[a] + zs[b]) / 2
                t = 1.0 - (z_avg - z_min) / z_range  # 0=far, 1=close
                alpha = int(50 + 130 * t)
            else:
                alpha = 150
            c = QColor(color); c.setAlpha(alpha)
            painter.setPen(QPen(c, 0.7))
            painter.drawLine(pts[a][0], pts[a][1], pts[b][0], pts[b][1])

        # ========== LAYER 3: cross-bracing diagonals ==========
        # These are the lines that don't follow anatomy — give it the
        # "engineered scaffold" look instead of "skin texture" look.
        brace_color = QColor(color); brace_color.setAlpha(110)
        brace_pen = QPen(brace_color, 0.9)
        brace_pen.setStyle(Qt.PenStyle.DashLine)
        brace_pen.setDashPattern([6, 4])
        painter.setPen(brace_pen)
        for a, b in self.BRACE_LINES:
            painter.drawLine(pts[a][0], pts[a][1], pts[b][0], pts[b][1])

        # ========== LAYER 4: scan line with afterglow ==========
        sweep_period = 110
        phase = (self.t % sweep_period) / sweep_period
        scan_y = fy1 + face_h * phase

        for i in range(4):
            offset = (i + 1) * 6
            tr_y = scan_y - offset
            tr_alpha = int(90 * (1 - i / 4))
            painter.setPen(QPen(QColor(255, 255, 255, tr_alpha), 0.8))
            for a, b in ALL_CONNECTIONS:
                ya, yb = pts[a][1], pts[b][1]
                if min(ya, yb) <= tr_y <= max(ya, yb):
                    painter.drawLine(pts[a][0], pts[a][1], pts[b][0], pts[b][1])

        painter.setPen(QPen(QColor(255, 255, 255, 255), 1.5))
        for a, b in ALL_CONNECTIONS:
            ya, yb = pts[a][1], pts[b][1]
            if min(ya, yb) <= scan_y <= max(ya, yb):
                painter.drawLine(pts[a][0], pts[a][1], pts[b][0], pts[b][1])

        # ========== LAYER 5: angular targeting brackets ==========
        bracket_color = QColor(color); bracket_color.setAlpha(230)
        painter.setPen(QPen(bracket_color, 2))
        breath = math.sin(self.t * 0.05) * 3
        pad = 22 + breath
        bl = max(10, face_w * 0.13)

        corners = [
            (fx1 - pad, fy1 - pad, 1, 1),    # TL: legs go right and down
            (fx2 + pad, fy1 - pad, -1, 1),   # TR
            (fx1 - pad, fy2 + pad, 1, -1),   # BL
            (fx2 + pad, fy2 + pad, -1, -1),  # BR
        ]
        for x, y, dx, dy in corners:
            painter.drawLine(int(x), int(y), int(x + bl * dx), int(y))
            painter.drawLine(int(x), int(y), int(x), int(y + bl * dy))

        # ========== LAYER 6: rotating reticle + center crosshair ==========
        reticle_r = max(face_w, face_h) / 2 + 32 + breath
        rot = (self.t * 1.4) % 360
        ret_color = QColor(color); ret_color.setAlpha(140)
        painter.setPen(QPen(ret_color, 1.2))
        for i in range(4):
            start = rot + i * 90
            painter.drawArc(QRectF(cx - reticle_r, cy - reticle_r,
                                    reticle_r * 2, reticle_r * 2),
                            int(start * 16), int(22 * 16))

        # Tiny tick marks at cardinal points around face
        tick_r_inner = reticle_r - 6
        tick_r_outer = reticle_r + 6
        for deg in (0, 90, 180, 270):
            rad = math.radians(deg)
            painter.drawLine(int(cx + math.cos(rad) * tick_r_inner),
                             int(cy + math.sin(rad) * tick_r_inner),
                             int(cx + math.cos(rad) * tick_r_outer),
                             int(cy + math.sin(rad) * tick_r_outer))

        # Center crosshair (small, sharp)
        ch_color = QColor(color); ch_color.setAlpha(200)
        painter.setPen(QPen(ch_color, 1))
        painter.drawLine(int(cx - 10), int(cy), int(cx - 3), int(cy))
        painter.drawLine(int(cx + 3), int(cy), int(cx + 10), int(cy))
        painter.drawLine(int(cx), int(cy - 10), int(cx), int(cy - 3))
        painter.drawLine(int(cx), int(cy + 3), int(cx), int(cy + 10))

        # ========== LAYER 7: angular landmark markers (no fat dots) ==========
        marker_color = QColor(color); marker_color.setAlpha(220)
        painter.setPen(QPen(marker_color, 1.2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for idx in self.DATA_ANCHORS:
            x, y = pts[idx]
            # Small diamond outline (4px radius) — sharper than a circle
            diamond = QPainterPath()
            diamond.moveTo(x, y - 4)
            diamond.lineTo(x + 4, y)
            diamond.lineTo(x, y + 4)
            diamond.lineTo(x - 4, y)
            diamond.closeSubpath()
            painter.drawPath(diamond)
            # Small center pixel
            painter.fillRect(int(x), int(y), 1, 1, marker_color)

        # ========== LAYER 8: real telemetry labels ==========
        telemetry = self._real_telemetry(metrics or {}, landmarks)

        font = QFont("Consolas", 9)
        font.setWeight(QFont.Weight.Bold)
        painter.setFont(font)

        label_color = QColor(color); label_color.setAlpha(240)
        line_color = QColor(color); line_color.setAlpha(140)

        for idx, value in telemetry.items():
            if idx >= len(pts):
                continue
            x, y = pts[idx]
            # Decide which side to put label on (away from face center)
            side = 1 if x > cx else -1
            elbow_x = x + side * 35
            elbow_y = y - 16
            label_x = elbow_x + side * 8
            text_w = painter.fontMetrics().horizontalAdvance(value)

            # Leader line: short out, then horizontal
            painter.setPen(QPen(line_color, 1))
            painter.drawLine(int(x), int(y), int(elbow_x), int(elbow_y))
            end_x = label_x + (text_w if side > 0 else -text_w)
            painter.drawLine(int(elbow_x), int(elbow_y), int(end_x), int(elbow_y))

            # Tiny tick at label end
            painter.drawLine(int(end_x), int(elbow_y - 3),
                             int(end_x), int(elbow_y + 3))

            # Text
            painter.setPen(QPen(label_color))
            text_x = label_x if side > 0 else label_x - text_w
            painter.drawText(QPointF(text_x, elbow_y + 3), value)

        # ========== LAYER 9: glitch chromatic offset ==========
        if self.glitch_timer > 0:
            self.glitch_timer -= 1
            offset = np.random.randint(-4, 5)
            painter.setPen(QPen(QColor(255, 50, 50, 140), 0.8))
            for a, b in ALL_CONNECTIONS[::3]:
                painter.drawLine(pts[a][0] + offset, pts[a][1],
                                 pts[b][0] + offset, pts[b][1])
            painter.setPen(QPen(QColor(50, 200, 255, 140), 0.8))
            for a, b in ALL_CONNECTIONS[1::3]:
                painter.drawLine(pts[a][0] - offset, pts[a][1],
                                 pts[b][0] - offset, pts[b][1])
        elif np.random.random() < 0.006:
            self.glitch_timer = 5

        # ========== LAYER 10: spark particles ==========
        if self.t % 7 == 0 and len(self.particles) < 20:
            idx = np.random.choice(list(self.DATA_ANCHORS.keys()))
            angle = np.random.uniform(0, math.tau)
            speed = np.random.uniform(2, 4)
            self.particles.append({
                "x": pts[idx][0], "y": pts[idx][1],
                "vx": math.cos(angle) * speed, "vy": math.sin(angle) * speed,
                "life": 18, "max": 18,
            })

        new_p = []
        for p in self.particles:
            p["x"] += p["vx"]; p["y"] += p["vy"]
            p["vx"] *= 0.94; p["vy"] *= 0.94
            p["life"] -= 1
            if p["life"] > 0:
                t = p["life"] / p["max"]
                pc = QColor(color); pc.setAlpha(int(220 * t))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(pc))
                # Square sparks instead of round — more "pixel" / "tech" feel
                s = max(1, int(2 * t))
                painter.fillRect(int(p["x"]), int(p["y"]), s, s, pc)
                new_p.append(p)
        self.particles = new_p

        self.last_pts = list(pts)
        

# ============ VIDEO WIDGET ============
class VideoWidget(QLabel):
    clicked = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setMinimumSize(960, 720)
        self.setStyleSheet("background-color: #000;")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_pixmap = None
        self.landmarks_pts = None
        self.mode = "neutral"
        self.mesh = MeshRenderer()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()

    def update_frame(self, bgr_frame, landmarks_pts, landmarks_raw, metrics, mode):
        self.landmarks_pts = landmarks_pts
        self.mode = mode

        h, w, _ = bgr_frame.shape
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()

        if landmarks_pts is not None:
            painter = QPainter(qimg)
            self.mesh.draw(painter, landmarks_pts, mode, w, h,
                        landmarks_raw, metrics)
            painter.end()

        pix = QPixmap.fromImage(qimg)
        self.setPixmap(pix.scaled(self.size(),
                                Qt.AspectRatioMode.KeepAspectRatio,
                                Qt.TransformationMode.SmoothTransformation))


# ============ OVERLAY PANELS (CSS-styled) ============
class GlassPill(QWidget):
    def __init__(self, text="", color=QColor(0, 255, 220)):
        super().__init__()
        self._color = color
        self.label = QLabel(text)
        self.label.setStyleSheet("color: white; background: transparent;")
        font = QFont("Segoe UI", 14)
        font.setWeight(QFont.Weight.DemiBold)
        self.label.setFont(font)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 10, 20, 10)
        layout.addWidget(self.label)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._update_style()

        # Soft drop shadow
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(30)
        shadow.setColor(QColor(0, 0, 0, 180))
        shadow.setOffset(0, 4)
        self.setGraphicsEffect(shadow)

    def _update_style(self):
        c = self._color
        self.setStyleSheet(f"""
            GlassPill {{
                background-color: rgba(15, 15, 22, 180);
                border: 1px solid rgba({c.red()}, {c.green()}, {c.blue()}, 200);
                border-radius: 18px;
            }}
        """)

    def set_color(self, color: QColor):
        self._color = color
        self._update_style()

    def set_text(self, text):
        self.label.setText(text)


class ModeBadge(QWidget):
    """Top-left mode label."""
    def __init__(self):
        super().__init__()
        self.label = QLabel("NEUTRAL")
        font = QFont("Segoe UI", 13)
        font.setWeight(QFont.Weight.Black)
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 2)
        self.label.setFont(font)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.addWidget(self.label)

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 200))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)

        self._color = QColor(0, 255, 220)
        self._refresh()

    def _refresh(self):
        c = self._color
        self.label.setStyleSheet(f"color: rgb({c.red()},{c.green()},{c.blue()}); background: transparent;")
        self.setStyleSheet(f"""
            ModeBadge {{
                background-color: rgba(15, 15, 22, 200);
                border: 1px solid rgba({c.red()},{c.green()},{c.blue()}, 220);
                border-radius: 14px;
            }}
        """)

    def set_mode(self, mode, color: QColor):
        self.label.setText(mode.upper())
        self._color = color
        self._refresh()


class ProgressRing(QWidget):
    """Bottom-left ring showing gesture hold progress."""
    def __init__(self):
        super().__init__()
        self.setFixedSize(110, 56)
        self.progress = 0.0
        self.color = QColor(0, 255, 220)
        self.label_text = ""
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

    def set_state(self, progress, target_mode):
        self.progress = progress
        if target_mode:
            self.color = MODE_COLORS[target_mode]
            self.label_text = target_mode.upper()
        else:
            self.label_text = ""
        self.update()

    def paintEvent(self, e):
        if self.progress <= 0:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        cx, cy, r = 24, 28, 18

        # Background ring
        bg_pen = QPen(QColor(60, 60, 70, 220), 3)
        p.setPen(bg_pen)
        p.drawEllipse(QPointF(cx, cy), r, r)

        # Progress arc
        arc_pen = QPen(self.color, 3)
        arc_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(arc_pen)
        span = int(360 * 16 * self.progress)
        p.drawArc(QRectF(cx - r, cy - r, r * 2, r * 2), 90 * 16, -span)

        # Inner dot
        p.setBrush(QBrush(self.color))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(cx, cy), 5, 5)

        # Label
        font = QFont("Segoe UI", 10)
        font.setWeight(QFont.Weight.Bold)
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1)
        p.setFont(font)
        p.setPen(QPen(self.color))
        p.drawText(QRectF(cx + r + 8, 0, 80, self.height()),
                   Qt.AlignmentFlag.AlignVCenter, self.label_text)


# ============ MAIN WINDOW ============
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FaceJudge")
        self.setStyleSheet("background-color: #0a0a0f;")

        self.analyzer = FaceAnalyzer()
        self.gestures = GestureDetector()
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.mode = "neutral"

        # Central video widget
        self.video = VideoWidget()
        self.video.clicked.connect(self.cycle_face)
        self.setCentralWidget(self.video)

        # Floating overlays (children of video widget so they overlay it)
        self.mode_badge = ModeBadge()
        self.mode_badge.setParent(self.video)
        self.mode_badge.move(24, 24)

        self.emotion_pill = GlassPill("", QColor(180, 180, 180))
        self.emotion_pill.setParent(self.video)
        self.emotion_pill.hide()

        self.caption_pill = GlassPill(MODE_CAPTIONS["neutral"], MODE_COLORS["neutral"])
        self.caption_pill.setParent(self.video)

        self.progress_ring = ProgressRing()
        self.progress_ring.setParent(self.video)

        self.hint = QLabel("click to cycle face   ·   space to capture   ·   esc to quit")
        self.hint.setParent(self.video)
        self.hint.setStyleSheet("color: rgba(160,160,170,180); background: transparent;")
        font = QFont("Segoe UI", 9)
        self.hint.setFont(font)
        self.hint.adjustSize()

        # Render loop
        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(16)  # ~60fps target

        self.showFullScreen()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Escape or e.key() == Qt.Key.Key_Q:
            self.close()
        elif e.key() == Qt.Key.Key_Space:
            self.analyzer.capture()
            # Brief flash animation could go here
        super().keyPressEvent(e)

    def cycle_face(self):
        self.analyzer.cycle_face()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._reposition_overlays()

    def _reposition_overlays(self):
        w, h = self.video.width(), self.video.height()
        self.mode_badge.move(24, 24)

        self.emotion_pill.adjustSize()
        ew = self.emotion_pill.width()
        self.emotion_pill.move(w - ew - 24, 24)

        self.caption_pill.adjustSize()
        cw = self.caption_pill.width()
        self.caption_pill.move((w - cw) // 2, h - self.caption_pill.height() - 32)

        self.progress_ring.move(24, h - self.progress_ring.height() - 28)

        self.hint.adjustSize()
        self.hint.move(w - self.hint.width() - 24, h - self.hint.height() - 16)

    def tick(self):
        ok, frame = self.cap.read()
        if not ok:
            return
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        target = self.gestures.detect(rgb)
        new_mode = self.gestures.update_mode(self.mode, target)
        if new_mode != self.mode:
            self.mode = new_mode
            self.analyzer.set_mode(self.mode)
            self.mode_badge.set_mode(self.mode, MODE_COLORS[self.mode])
            self.caption_pill.set_text(MODE_CAPTIONS[self.mode])
            self.caption_pill.set_color(MODE_COLORS[self.mode])

        h, w = frame.shape[:2]
        landmarks_pts, landmarks_raw, metrics = self._analyze_face(frame, w, h)
        self.video.update_frame(frame, landmarks_pts, landmarks_raw, metrics, self.mode)

        state = self.analyzer.get_state()
        emo = state["emotion"]
        if emo and emo != "neutral":
            self.emotion_pill.set_text(emo.upper())
            self.emotion_pill.show()
        else:
            self.emotion_pill.hide()

        progress, streak_target = self.gestures.progress()
        self.progress_ring.set_state(progress, streak_target)

        self._reposition_overlays()
        
    def _analyze_face(self, frame, w, h):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int((time.time() - self.analyzer.start) * 1000)
        result = self.analyzer.landmarker.detect_for_video(mp_image, ts_ms)

        faces = result.face_landmarks or []
        if not faces:
            self.analyzer.last_clean_frame = frame.copy()
            return None, None, None

        from cvModule import order_faces, compute_metrics, emotion_from_blendshapes
        ordered = order_faces(faces, w, h)
        self.analyzer.locked_idx %= len(ordered)
        real_locked = ordered[self.analyzer.locked_idx]
        locked_lm = faces[real_locked]

        self.analyzer.last_clean_frame = frame.copy()
        self.analyzer.last_locked_landmarks = locked_lm

        metrics, pts = compute_metrics(locked_lm, w, h)
        emotion = ("neutral", 0.0)
        if result.face_blendshapes and real_locked < len(result.face_blendshapes):
            emotion = emotion_from_blendshapes(result.face_blendshapes[real_locked])
        self.analyzer.smoother.add(metrics, emotion)
        if time.time() - self.analyzer.smoother.last_update >= 2.0:
            self.analyzer.smoother.commit()

        return pts, locked_lm, metrics

    def closeEvent(self, e):
        self.timer.stop()
        self.cap.release()
        self.analyzer.close()
        self.gestures.close()
        super().closeEvent(e)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    sys.exit(app.exec())