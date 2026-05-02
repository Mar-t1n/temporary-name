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


# ============ HOLOGRAPHIC FACE MESH (drawn with QPainter) ============
# ============ HOLOGRAPHIC FACE OVERLAY ============
class MeshRenderer:
    """Sci-fi targeting overlay. Brackets, depth shading, glitch lines, data labels."""

    # Connections to render as the wireframe (subset of ALL_CONNECTIONS for speed)
    # We use ALL_CONNECTIONS but render the FACE_OVAL separately for the bracket effect
    KEY_NODES = [33, 133, 263, 362, 1, 152, 10, 234, 454, 13, 14, 61, 291, 168, 6]
    DATA_NODES = [33, 263, 1, 152, 234, 454]  # nodes that get readout labels

    # Face oval indices (perimeter) — used for bracket corners
    FACE_OVAL_PTS = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                     397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                     172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

    def __init__(self):
        self.t = 0
        self.particles = []
        self.glitch_timer = 0
        self.last_pts = None  # for motion trails
        self.data_values = {}  # cached random-looking readouts per node
        self.data_refresh = 0
        # Pre-generate "data" strings (realistic-looking gibberish)
        self._refresh_data()

    def _refresh_data(self):
        """Generate fake telemetry strings for data labels."""
        self.data_values = {
            n: f"{np.random.randint(0, 999):03d}.{np.random.randint(0, 99):02d}"
            for n in self.DATA_NODES
        }

    def draw(self, painter: QPainter, pts, mode, w, h, landmarks=None):
        """pts: 478 (x,y) tuples. landmarks: original mp landmarks (for z-depth)."""
        self.t += 1
        if self.t % 30 == 0:
            self._refresh_data()

        color = MODE_COLORS[mode]
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # Bounding box of face for brackets/HUD
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        fx1, fx2 = min(xs), max(xs)
        fy1, fy2 = min(ys), max(ys)
        face_w = fx2 - fx1
        face_h = fy2 - fy1
        cx, cy = (fx1 + fx2) / 2, (fy1 + fy2) / 2

        # ======== LAYER 1: motion-trail wireframe (previous frame ghosted) ========
        if self.last_pts is not None and len(self.last_pts) == len(pts):
            ghost = QColor(color)
            ghost.setAlpha(35)
            ghost_pen = QPen(ghost, 0.8)
            painter.setPen(ghost_pen)
            for a, b in ALL_CONNECTIONS:
                painter.drawLine(self.last_pts[a][0], self.last_pts[a][1],
                                 self.last_pts[b][0], self.last_pts[b][1])

        # ======== LAYER 2: depth-shaded wireframe ========
        # Color intensity varies with z (closer = brighter)
        if landmarks is not None:
            zs = [lm.z for lm in landmarks]
            z_min, z_max = min(zs), max(zs)
            z_range = max(0.001, z_max - z_min)
        else:
            zs = None

        # Outer glow pass (wide, low alpha)
        glow = QColor(color)
        glow.setAlpha(28)
        glow_pen = QPen(glow, 5)
        glow_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(glow_pen)
        for a, b in ALL_CONNECTIONS:
            painter.drawLine(pts[a][0], pts[a][1], pts[b][0], pts[b][1])

        # Core wireframe with depth-based alpha
        for a, b in ALL_CONNECTIONS:
            if zs is not None:
                z_avg = (zs[a] + zs[b]) / 2
                # Closer points are more negative z → brighter
                t = 1.0 - (z_avg - z_min) / z_range
                alpha = int(80 + 140 * t)
                width = 0.6 + 1.0 * t
            else:
                alpha = 180
                width = 1.0
            c = QColor(color)
            c.setAlpha(alpha)
            pen = QPen(c, width)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawLine(pts[a][0], pts[a][1], pts[b][0], pts[b][1])

        # ======== LAYER 3: scan line with afterglow trail ========
        sweep_period = 90
        sweep_phase = (self.t % sweep_period) / sweep_period
        scan_y = fy1 + face_h * sweep_phase

        # Trailing afterglow (3 fading bands behind the scan line)
        for i in range(3):
            offset = (i + 1) * 8
            trail_y = scan_y - offset
            trail_alpha = int(70 * (1 - i / 3))
            tc = QColor(255, 255, 255, trail_alpha)
            tpen = QPen(tc, 1.2)
            painter.setPen(tpen)
            for a, b in ALL_CONNECTIONS:
                ya, yb = pts[a][1], pts[b][1]
                if min(ya, yb) <= trail_y <= max(ya, yb):
                    painter.drawLine(pts[a][0], pts[a][1], pts[b][0], pts[b][1])

        # Bright scan line itself
        scan_color = QColor(255, 255, 255, 255)
        scan_pen = QPen(scan_color, 2.2)
        painter.setPen(scan_pen)
        for a, b in ALL_CONNECTIONS:
            ya, yb = pts[a][1], pts[b][1]
            if min(ya, yb) <= scan_y <= max(ya, yb):
                painter.drawLine(pts[a][0], pts[a][1], pts[b][0], pts[b][1])

        # ======== LAYER 4: targeting brackets at face corners ========
        bracket_color = QColor(color)
        bracket_color.setAlpha(230)
        bracket_pen = QPen(bracket_color, 2.5)
        bracket_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(bracket_pen)

        # Subtle breathing animation
        breath = math.sin(self.t * 0.05) * 4
        pad = 18 + breath
        bl = max(8, face_w * 0.12)  # bracket leg length

        # Top-left
        painter.drawLine(int(fx1 - pad), int(fy1 - pad + bl),
                         int(fx1 - pad), int(fy1 - pad))
        painter.drawLine(int(fx1 - pad), int(fy1 - pad),
                         int(fx1 - pad + bl), int(fy1 - pad))
        # Top-right
        painter.drawLine(int(fx2 + pad - bl), int(fy1 - pad),
                         int(fx2 + pad), int(fy1 - pad))
        painter.drawLine(int(fx2 + pad), int(fy1 - pad),
                         int(fx2 + pad), int(fy1 - pad + bl))
        # Bottom-left
        painter.drawLine(int(fx1 - pad), int(fy2 + pad - bl),
                         int(fx1 - pad), int(fy2 + pad))
        painter.drawLine(int(fx1 - pad), int(fy2 + pad),
                         int(fx1 - pad + bl), int(fy2 + pad))
        # Bottom-right
        painter.drawLine(int(fx2 + pad - bl), int(fy2 + pad),
                         int(fx2 + pad), int(fy2 + pad))
        painter.drawLine(int(fx2 + pad), int(fy2 + pad - bl),
                         int(fx2 + pad), int(fy2 + pad))

        # ======== LAYER 5: rotating reticle around face center ========
        reticle_r = max(face_w, face_h) / 2 + 30 + breath
        rot_angle = (self.t * 1.2) % 360
        reticle_color = QColor(color)
        reticle_color.setAlpha(120)
        rpen = QPen(reticle_color, 1.4)
        painter.setPen(rpen)
        # Draw 4 arc segments rotating around the face
        for i in range(4):
            start_deg = rot_angle + i * 90
            painter.drawArc(QRectF(cx - reticle_r, cy - reticle_r,
                                    reticle_r * 2, reticle_r * 2),
                            int(start_deg * 16), int(20 * 16))

        # Crosshair at face center
        ch_color = QColor(color)
        ch_color.setAlpha(180)
        chpen = QPen(ch_color, 1.2)
        painter.setPen(chpen)
        painter.drawLine(int(cx - 12), int(cy), int(cx - 4), int(cy))
        painter.drawLine(int(cx + 4), int(cy), int(cx + 12), int(cy))
        painter.drawLine(int(cx), int(cy - 12), int(cx), int(cy - 4))
        painter.drawLine(int(cx), int(cy + 4), int(cx), int(cy + 12))

        # ======== LAYER 6: glowing key landmark nodes ========
        pulse = 0.5 + 0.5 * math.sin(self.t * 0.18)
        for idx in self.KEY_NODES:
            cx_n, cy_n = pts[idx]
            # Wide outer halo
            grad = QRadialGradient(QPointF(cx_n, cy_n), 18)
            c1 = QColor(color); c1.setAlpha(int(120 * pulse + 40))
            c2 = QColor(color); c2.setAlpha(0)
            grad.setColorAt(0, c1)
            grad.setColorAt(1, c2)
            painter.setBrush(QBrush(grad))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(cx_n, cy_n), 18, 18)
            # Bright core
            painter.setBrush(QBrush(QColor(255, 255, 255, 255)))
            painter.drawEllipse(QPointF(cx_n, cy_n), 2.8, 2.8)
            # Ring outline
            ring_pen = QPen(color, 1.2)
            painter.setPen(ring_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(cx_n, cy_n), 5, 5)

        # ======== LAYER 7: data readouts on select nodes ========
        font = QFont("Consolas", 8)
        font.setWeight(QFont.Weight.Bold)
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0.5)
        painter.setFont(font)
        for idx in self.DATA_NODES:
            cx_n, cy_n = pts[idx]
            value = self.data_values.get(idx, "000.00")
            # Draw a short connector line
            offset_x = 28 if cx_n > cx else -28
            offset_y = -14
            tx, ty = cx_n + offset_x, cy_n + offset_y
            line_color = QColor(color)
            line_color.setAlpha(160)
            painter.setPen(QPen(line_color, 1))
            painter.drawLine(int(cx_n), int(cy_n), int(tx), int(ty))
            painter.drawLine(int(tx), int(ty),
                             int(tx + (15 if offset_x > 0 else -15)), int(ty))
            # Background pill for text
            text_color = QColor(color)
            text_color.setAlpha(255)
            painter.setPen(QPen(text_color))
            text_x = tx + (18 if offset_x > 0 else -18)
            painter.drawText(QPointF(text_x if offset_x > 0 else text_x - 40,
                                     ty + 3), value)

        # ======== LAYER 8: glitch/chromatic offset (occasional) ========
        if self.glitch_timer > 0:
            self.glitch_timer -= 1
            offset = np.random.randint(-3, 4)
            glitch_color = QColor(255, 50, 50, 120)
            painter.setPen(QPen(glitch_color, 1))
            for a, b in ALL_CONNECTIONS[::4]:  # every 4th line
                painter.drawLine(pts[a][0] + offset, pts[a][1],
                                 pts[b][0] + offset, pts[b][1])
        elif np.random.random() < 0.005:  # rare random glitch
            self.glitch_timer = 4

        # ======== LAYER 9: spark particles ========
        if self.t % 6 == 0 and len(self.particles) < 30:
            idx = np.random.choice(self.KEY_NODES)
            angle = np.random.uniform(0, math.tau)
            speed = np.random.uniform(2.5, 5)
            self.particles.append({
                "x": pts[idx][0], "y": pts[idx][1],
                "vx": math.cos(angle) * speed, "vy": math.sin(angle) * speed,
                "life": 22, "max_life": 22,
            })

        new_particles = []
        for p in self.particles:
            p["x"] += p["vx"]; p["y"] += p["vy"]
            p["vx"] *= 0.96; p["vy"] *= 0.96  # drag
            p["life"] -= 1
            if p["life"] > 0:
                t = p["life"] / p["max_life"]
                pc = QColor(color); pc.setAlpha(int(255 * t))
                painter.setBrush(QBrush(pc))
                painter.setPen(Qt.PenStyle.NoPen)
                size = 1.5 + t * 1.5
                painter.drawEllipse(QPointF(p["x"], p["y"]), size, size)
                new_particles.append(p)
        self.particles = new_particles

        # Save current pts for next frame's motion trail
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

    def update_frame(self, bgr_frame, landmarks_pts, landmarks_raw, mode):
        self.landmarks_pts = landmarks_pts
        self.mode = mode

        h, w, _ = bgr_frame.shape
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()

        if landmarks_pts is not None:
            painter = QPainter(qimg)
            self.mesh.draw(painter, landmarks_pts, mode, w, h, landmarks_raw)
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
        landmarks_pts, landmarks_raw = self._analyze_face(frame, w, h)

        self.video.update_frame(frame, landmarks_pts, landmarks_raw, self.mode)

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
            return None, None

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

        return pts, locked_lm  # return raw landmarks too

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