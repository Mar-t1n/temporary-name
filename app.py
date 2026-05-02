"""
FaceJudge - desktop app (PyQt6 + OpenCV)
Run: pip install PyQt6 opencv-python mediapipe numpy
     python app.py
"""

import sys
import os
import time
import math
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from PyQt6.QtCore import Qt, QTimer, QPointF, QRectF, pyqtSignal
from PyQt6.QtGui import (QImage, QPixmap, QPainter, QColor, QFont, QPen,
                         QBrush, QRadialGradient, QPainterPath)
from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QWidget,
                             QVBoxLayout, QHBoxLayout, QPushButton,
                             QGraphicsDropShadowEffect, QLineEdit, QFileDialog,
                             QFrame)

from cvModule import FaceAnalyzer, ALL_CONNECTIONS

# ============ CONFIG ============
GESTURE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
GESTURE_MODEL_PATH = "gesture_recognizer.task"

GESTURE_HOLD_FRAMES = 8       # how many recent frames must agree
GESTURE_WINDOW = 12           # size of the rolling vote window
GESTURE_COOLDOWN = 1.2
MISS_TOLERANCE = 10           # forgive up to ~third of a second of detection gaps

MODE_COLORS = {
    "neutral": QColor(160, 160, 170),
    "glaze":   QColor(60, 220, 130),
    "hate":    QColor(255, 70, 70),
}
ACCENT = QColor(120, 200, 255)  # UI accent independent of mode

MODE_CAPTIONS = {
    "neutral": "Hold thumb sideways, up to glaze, down to hate",
    "glaze":   "Glazing in progress...",
    "hate":    "Roasting in progress...",
}

if not os.path.exists(GESTURE_MODEL_PATH):
    print("Downloading gesture recognizer model...")
    urllib.request.urlretrieve(GESTURE_MODEL_URL, GESTURE_MODEL_PATH)


# ============ GESTURE CLASSIFIER ============
def classify_thumb_gesture(landmarks):
    """Return 'up' / 'down' / 'side' / None.

    Looser, more forgiving than before:
      - Curl check counts a finger as curled if its tip is anywhere near
        or past the PIP joint (1.15x slack instead of 1.05x).
      - Only 2 of 4 fingers must be curled (the other two can be lazy).
      - Thumb only needs to be moderately extended.
      - Direction zones are wide and overlap-free.
    """
    if not landmarks:
        return None

    def d(a, b):
        return math.hypot(landmarks[a].x - landmarks[b].x,
                          landmarks[a].y - landmarks[b].y)

    palm = d(0, 9)  # wrist to middle-finger MCP
    if palm < 0.04:
        return None

    # Thumb must be at least somewhat extended away from its base
    if d(2, 4) < palm * 0.35:
        return None

    # Count curled fingers — tip is curled if it's not significantly
    # farther from wrist than the PIP joint.
    curled = sum(1 for pip, tip in [(6, 8), (10, 12), (14, 16), (18, 20)]
                 if d(0, tip) < d(0, pip) * 1.15)

    # Need at least 2 curled to count as a thumb gesture (was 3).
    if curled < 2:
        return None

    # Direction: vector from thumb base (CMC, idx 1) to tip (idx 4)
    # Using CMC instead of MCP gives a longer baseline = more stable angle.
    dx = landmarks[4].x - landmarks[1].x
    dy = landmarks[4].y - landmarks[1].y

    # Use angle in degrees so the zones are intuitive.
    # 0° = pointing right, -90° = up, 90° = down, ±180° = left
    angle = math.degrees(math.atan2(dy, dx))

    # Sideways: anything pointing roughly horizontal (mirrored ok)
    # — within ±35° of horizontal on either side.
    if abs(angle) < 35 or abs(angle) > 145:
        return "side"
    # Up: between -145° and -35° (anywhere in upper half, excluding sides)
    if -145 < angle < -35:
        return "up"
    # Down: between 35° and 145°
    if 35 < angle < 145:
        return "down"
    return None


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

        # Rolling window of recent classifications (per-frame raw guesses)
        self.window = []
        self.miss_count = 0
        self.last_switch = 0.0

        # The mode currently being "voted into" via the window
        self.candidate = None

    def detect(self, rgb_frame):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        ts_ms = int((time.time() - self.start) * 1000)
        result = self.recognizer.recognize_for_video(mp_image, ts_ms)
        if not result.hand_landmarks:
            return None
        orient = classify_thumb_gesture(result.hand_landmarks[0])
        if orient is None:
            return None
        return {"up": "glaze", "down": "hate", "side": "neutral"}.get(orient)

    def update_mode(self, current_mode, target):
        """Vote-based mode switching with miss tolerance."""
        now = time.time()

        if target is None:
            # No detection this frame - tolerate up to MISS_TOLERANCE
            self.miss_count += 1
            if self.miss_count > MISS_TOLERANCE:
                self.window.clear()
                self.candidate = None
            return current_mode

        self.miss_count = 0
        self.window.append(target)
        if len(self.window) > GESTURE_WINDOW:
            self.window.pop(0)

        # Already on this mode and inside cooldown? freeze the vote.
        if target == current_mode and now - self.last_switch < GESTURE_COOLDOWN:
            return current_mode

        # Need at least HOLD_FRAMES of the same target in the recent window
        # for a switch to commit.
        counts = {"glaze": 0, "hate": 0, "neutral": 0}
        for v in self.window:
            counts[v] = counts.get(v, 0) + 1

        winner = max(counts, key=lambda k: counts[k])
        if (counts[winner] >= GESTURE_HOLD_FRAMES
                and winner != current_mode
                and now - self.last_switch >= GESTURE_COOLDOWN):
            self.last_switch = now
            self.window.clear()
            self.candidate = None
            return winner

        # Track candidate for progress ring
        self.candidate = winner if counts[winner] > 0 else None
    
        # Track candidate for progress ring (but not the one we're already on)
        if winner != current_mode and counts[winner] > 0:
            self.candidate = winner
        else:
            self.candidate = None
        return current_mode


    def progress(self):
        if not self.candidate:
            return 0.0, None
        counts = {"glaze": 0, "hate": 0, "neutral": 0}
        for v in self.window:
            counts[v] = counts.get(v, 0) + 1
        return min(1.0, counts[self.candidate] / GESTURE_HOLD_FRAMES), self.candidate

    def close(self):
        self.recognizer.close()


# ============ HOLOGRAPHIC FACE OVERLAY ============
class MeshRenderer:
    DATA_ANCHORS = [33, 263, 1, 152, 234, 454, 13]
    BRACE_LINES = [(33, 263), (33, 152), (263, 152), (234, 454), (10, 152), (61, 291)]

    def __init__(self):
        self.t = 0
        self.particles = []
        self.glitch_timer = 0
        self.last_pts = None

    def _telemetry(self, metrics, landmarks):
        out = {}
        if landmarks:
            out[33] = f"L.EYE  {math.hypot(landmarks[133].x - landmarks[33].x, landmarks[133].y - landmarks[33].y) * 1000:5.1f}"
            out[263] = f"R.EYE  {math.hypot(landmarks[362].x - landmarks[263].x, landmarks[362].y - landmarks[263].y) * 1000:5.1f}"
            out[1] = f"NOSE.W {math.hypot(landmarks[129].x - landmarks[358].x, landmarks[129].y - landmarks[358].y) * 1000:5.1f}"
        out[152] = f"FWHR   {metrics.get('fwhr', 0):.2f}"
        out[234] = f"JAW    {metrics.get('jaw_ratio', 0):.2f}"
        out[454] = f"SYM    {metrics.get('symmetry', 0):.1f}%"
        out[13] = f"LIP.R  {metrics.get('lip_ratio', 0):.2f}"
        return out

    def draw(self, painter, pts, mode, w, h, landmarks=None, metrics=None):
        self.t += 1
        color = MODE_COLORS[mode]
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        fx1, fx2 = min(xs), max(xs); fy1, fy2 = min(ys), max(ys)
        face_w = fx2 - fx1; face_h = fy2 - fy1
        cx, cy = (fx1 + fx2) / 2, (fy1 + fy2) / 2

        zs = [lm.z for lm in landmarks] if landmarks else None
        if zs:
            z_min, z_max = min(zs), max(zs)
            z_range = max(0.001, z_max - z_min)

        # Ghost trail
        if self.last_pts and len(self.last_pts) == len(pts):
            ghost = QColor(color); ghost.setAlpha(25)
            painter.setPen(QPen(ghost, 0.6))
            for a, b in ALL_CONNECTIONS:
                painter.drawLine(self.last_pts[a][0], self.last_pts[a][1],
                                 self.last_pts[b][0], self.last_pts[b][1])

        # Depth-shaded wireframe
        for a, b in ALL_CONNECTIONS:
            if zs:
                t = 1.0 - ((zs[a] + zs[b]) / 2 - z_min) / z_range
                alpha = int(50 + 130 * t)
            else:
                alpha = 150
            c = QColor(color); c.setAlpha(alpha)
            painter.setPen(QPen(c, 0.7))
            painter.drawLine(pts[a][0], pts[a][1], pts[b][0], pts[b][1])

        # Cross bracing
        brace = QColor(color); brace.setAlpha(110)
        bp = QPen(brace, 0.9); bp.setStyle(Qt.PenStyle.DashLine); bp.setDashPattern([6, 4])
        painter.setPen(bp)
        for a, b in self.BRACE_LINES:
            painter.drawLine(pts[a][0], pts[a][1], pts[b][0], pts[b][1])

        # Scan line + afterglow
        phase = (self.t % 110) / 110
        scan_y = fy1 + face_h * phase
        for i in range(4):
            tr_y = scan_y - (i + 1) * 6
            painter.setPen(QPen(QColor(255, 255, 255, int(90 * (1 - i / 4))), 0.8))
            for a, b in ALL_CONNECTIONS:
                if min(pts[a][1], pts[b][1]) <= tr_y <= max(pts[a][1], pts[b][1]):
                    painter.drawLine(pts[a][0], pts[a][1], pts[b][0], pts[b][1])
        painter.setPen(QPen(QColor(255, 255, 255, 255), 1.5))
        for a, b in ALL_CONNECTIONS:
            if min(pts[a][1], pts[b][1]) <= scan_y <= max(pts[a][1], pts[b][1]):
                painter.drawLine(pts[a][0], pts[a][1], pts[b][0], pts[b][1])

        # Targeting brackets
        breath = math.sin(self.t * 0.05) * 3
        pad = 22 + breath
        bl = max(10, face_w * 0.13)
        bc = QColor(color); bc.setAlpha(230)
        painter.setPen(QPen(bc, 2))
        for x, y, dx, dy in [(fx1 - pad, fy1 - pad, 1, 1),
                             (fx2 + pad, fy1 - pad, -1, 1),
                             (fx1 - pad, fy2 + pad, 1, -1),
                             (fx2 + pad, fy2 + pad, -1, -1)]:
            painter.drawLine(int(x), int(y), int(x + bl * dx), int(y))
            painter.drawLine(int(x), int(y), int(x), int(y + bl * dy))

        # Rotating reticle + crosshair
        r = max(face_w, face_h) / 2 + 32 + breath
        rot = (self.t * 1.4) % 360
        rc = QColor(color); rc.setAlpha(140)
        painter.setPen(QPen(rc, 1.2))
        for i in range(4):
            painter.drawArc(QRectF(cx - r, cy - r, r * 2, r * 2),
                            int((rot + i * 90) * 16), int(22 * 16))
        for deg in (0, 90, 180, 270):
            rad = math.radians(deg)
            painter.drawLine(int(cx + math.cos(rad) * (r - 6)),
                             int(cy + math.sin(rad) * (r - 6)),
                             int(cx + math.cos(rad) * (r + 6)),
                             int(cy + math.sin(rad) * (r + 6)))
        ch = QColor(color); ch.setAlpha(200)
        painter.setPen(QPen(ch, 1))
        for dx, dy in [(-10, 0), (3, 0), (0, -10), (0, 3)]:
            painter.drawLine(int(cx + dx), int(cy + dy),
                             int(cx + dx + (7 if dx != 0 else 0)),
                             int(cy + dy + (7 if dy != 0 else 0)))

        # Diamond markers
        mc = QColor(color); mc.setAlpha(220)
        painter.setPen(QPen(mc, 1.2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for idx in self.DATA_ANCHORS:
            x, y = pts[idx]
            d = QPainterPath()
            d.moveTo(x, y - 4); d.lineTo(x + 4, y)
            d.lineTo(x, y + 4); d.lineTo(x - 4, y); d.closeSubpath()
            painter.drawPath(d)

        # Telemetry labels
        font = QFont("Consolas", 9); font.setWeight(QFont.Weight.Bold)
        painter.setFont(font)
        lc = QColor(color); lc.setAlpha(140)
        tc = QColor(color); tc.setAlpha(240)
        for idx, value in self._telemetry(metrics or {}, landmarks).items():
            if idx >= len(pts):
                continue
            x, y = pts[idx]
            side = 1 if x > cx else -1
            ex, ey = x + side * 35, y - 16
            tw = painter.fontMetrics().horizontalAdvance(value)
            painter.setPen(QPen(lc, 1))
            painter.drawLine(int(x), int(y), int(ex), int(ey))
            end_x = ex + side * (8 + tw)
            painter.drawLine(int(ex), int(ey), int(end_x), int(ey))
            painter.drawLine(int(end_x), int(ey - 3), int(end_x), int(ey + 3))
            painter.setPen(QPen(tc))
            text_x = ex + side * 8 if side > 0 else ex - 8 - tw
            painter.drawText(QPointF(text_x, ey + 3), value)

        # Glitch
        if self.glitch_timer > 0:
            self.glitch_timer -= 1
            off = np.random.randint(-4, 5)
            painter.setPen(QPen(QColor(255, 50, 50, 140), 0.8))
            for a, b in ALL_CONNECTIONS[::3]:
                painter.drawLine(pts[a][0] + off, pts[a][1], pts[b][0] + off, pts[b][1])
            painter.setPen(QPen(QColor(50, 200, 255, 140), 0.8))
            for a, b in ALL_CONNECTIONS[1::3]:
                painter.drawLine(pts[a][0] - off, pts[a][1], pts[b][0] - off, pts[b][1])
        elif np.random.random() < 0.006:
            self.glitch_timer = 5

        # Particles
        if self.t % 7 == 0 and len(self.particles) < 20:
            idx = np.random.choice(self.DATA_ANCHORS)
            angle = np.random.uniform(0, math.tau)
            speed = np.random.uniform(2, 4)
            self.particles.append({"x": pts[idx][0], "y": pts[idx][1],
                                   "vx": math.cos(angle) * speed,
                                   "vy": math.sin(angle) * speed,
                                   "life": 18, "max": 18})
        new_p = []
        for p in self.particles:
            p["x"] += p["vx"]; p["y"] += p["vy"]
            p["vx"] *= 0.94; p["vy"] *= 0.94
            p["life"] -= 1
            if p["life"] > 0:
                t = p["life"] / p["max"]
                pc = QColor(color); pc.setAlpha(int(220 * t))
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
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.mesh = MeshRenderer()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.setFocus()
            self.clicked.emit()

    def update_frame(self, bgr_frame, pts, raw, metrics, mode):
        h, w, _ = bgr_frame.shape
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
        if pts is not None:
            painter = QPainter(qimg)
            self.mesh.draw(painter, pts, mode, w, h, raw, metrics)
            painter.end()
        pix = QPixmap.fromImage(qimg)
        self.setPixmap(pix.scaled(self.size(),
                                  Qt.AspectRatioMode.KeepAspectRatio,
                                  Qt.TransformationMode.SmoothTransformation))


# ============ OVERLAY WIDGETS ============
class ModeBadge(QWidget):
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
        shadow.setBlurRadius(20); shadow.setColor(QColor(0, 0, 0, 200))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)
        self.set_mode("neutral", MODE_COLORS["neutral"])

    def set_mode(self, mode, color):
        self.label.setText(mode.upper())
        c = color
        self.label.setStyleSheet(f"color: rgb({c.red()},{c.green()},{c.blue()}); background: transparent;")
        self.setStyleSheet(f"""
            ModeBadge {{
                background-color: rgba(15, 15, 22, 200);
                border: 1px solid rgba({c.red()},{c.green()},{c.blue()}, 220);
                border-radius: 14px;
            }}
        """)


class GlassPill(QWidget):
    def __init__(self, text="", color=ACCENT):
        super().__init__()
        self._color = color
        self.label = QLabel(text)
        font = QFont("Segoe UI", 14); font.setWeight(QFont.Weight.DemiBold)
        self.label.setFont(font)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 10, 20, 10)
        layout.addWidget(self.label)
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(30); shadow.setColor(QColor(0, 0, 0, 180))
        shadow.setOffset(0, 4)
        self.setGraphicsEffect(shadow)
        self._refresh()

    def _refresh(self):
        c = self._color
        self.label.setStyleSheet("color: white; background: transparent;")
        self.setStyleSheet(f"""
            GlassPill {{
                background-color: rgba(15, 15, 22, 180);
                border: 1px solid rgba({c.red()},{c.green()},{c.blue()}, 200);
                border-radius: 18px;
            }}
        """)

    def set_text(self, text): self.label.setText(text)
    def set_color(self, color): self._color = color; self._refresh()


class ProgressRing(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(110, 56)
        self.progress = 0.0
        self.color = MODE_COLORS["neutral"]
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
        p.setPen(QPen(QColor(60, 60, 70, 220), 3))
        p.drawEllipse(QPointF(cx, cy), r, r)
        arc = QPen(self.color, 3); arc.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(arc)
        p.drawArc(QRectF(cx - r, cy - r, r * 2, r * 2),
                  90 * 16, -int(360 * 16 * self.progress))
        p.setBrush(QBrush(self.color)); p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(cx, cy), 5, 5)
        font = QFont("Segoe UI", 10); font.setWeight(QFont.Weight.Bold)
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1)
        p.setFont(font)
        p.setPen(QPen(self.color))
        p.drawText(QRectF(cx + r + 8, 0, 80, self.height()),
                   Qt.AlignmentFlag.AlignVCenter, self.label_text)


class TransitionFlash(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.alpha = 0.0
        self.text = ""
        self.color = MODE_COLORS["neutral"]
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._step)

    def trigger(self, mode_name, color):
        self.text = mode_name.upper()
        self.color = color
        self.alpha = 1.0
        parent = self.parent()
        # parent() may return a generic QObject; ensure it has a QWidget size()
        if isinstance(parent, QWidget):
            self.resize(parent.size())
        self.raise_(); self.show()
        self._timer.start(16)

    def _step(self):
        self.alpha -= 0.04
        if self.alpha <= 0:
            self.alpha = 0; self.hide(); self._timer.stop()
        self.update()

    def paintEvent(self, e):
        if self.alpha <= 0:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        grad = QRadialGradient(w / 2, h / 2, max(w, h) * 0.7)
        c1 = QColor(self.color); c1.setAlpha(int(60 * self.alpha))
        c2 = QColor(self.color); c2.setAlpha(0)
        grad.setColorAt(0, c2); grad.setColorAt(0.7, c2); grad.setColorAt(1, c1)
        p.fillRect(self.rect(), QBrush(grad))
        edge = QColor(self.color); edge.setAlpha(int(220 * self.alpha))
        thickness = int(8 * self.alpha) + 2
        p.fillRect(0, 0, w, thickness, edge)
        p.fillRect(0, h - thickness, w, thickness, edge)
        p.fillRect(0, 0, thickness, h, edge)
        p.fillRect(w - thickness, 0, thickness, h, edge)
        scale = 1.0 + (1.0 - self.alpha) * 0.4
        font = QFont("Segoe UI", int(72 * scale)); font.setWeight(QFont.Weight.Black)
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 8)
        p.setFont(font)
        text_color = QColor(self.color); text_color.setAlpha(int(255 * self.alpha))
        p.setPen(QPen(text_color))
        p.drawText(QRectF(0, 0, w, h), Qt.AlignmentFlag.AlignCenter, self.text)


# ============ PROFILE PANEL (always visible, bottom-right) ============
class ProfilePanel(QFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.setObjectName("profilePanel")
        self.setFixedSize(320, 290)
        self.data = {"github": "", "linkedin": "", "resume_path": ""}

        accent = ACCENT
        ar, ag, ab = accent.red(), accent.green(), accent.blue()
        self.setStyleSheet(f"""
            #profilePanel {{
                background-color: rgba(15, 15, 22, 210);
                border: 1px solid rgba(255, 255, 255, 35);
                border-radius: 14px;
            }}
            QLabel#sectionLabel {{
                color: rgba(160, 160, 180, 255);
                font-size: 9px;
                font-weight: 700;
                letter-spacing: 1.5px;
                background: transparent;
            }}
            QLabel#title {{
                color: white;
                font-size: 11px;
                font-weight: 800;
                letter-spacing: 2px;
                background: transparent;
            }}
            QLabel#prefix {{
                color: rgba(140, 140, 160, 255);
                font-size: 12px;
                background: transparent;
                padding: 0 2px 0 8px;
            }}
            QLineEdit {{
                background: transparent;
                border: none;
                color: white;
                padding: 6px 4px;
                font-size: 12px;
                selection-background-color: rgba({ar},{ag},{ab}, 100);
            }}
            QFrame#prefixGroup {{
                background-color: rgba(0, 0, 0, 110);
                border: 1px solid rgba(255, 255, 255, 30);
                border-radius: 7px;
            }}
            QFrame#prefixGroup:focus-within {{
                border: 1px solid rgba({ar},{ag},{ab}, 200);
            }}
            QPushButton#fileBtn {{
                background-color: rgba(0, 0, 0, 110);
                border: 1px dashed rgba(255, 255, 255, 50);
                border-radius: 7px;
                color: rgba(200, 200, 220, 255);
                padding: 9px 12px;
                text-align: left;
                font-size: 11px;
            }}
            QPushButton#fileBtn:hover {{
                border: 1px dashed rgba({ar},{ag},{ab}, 200);
                color: white;
            }}
            QPushButton#saveBtn {{
                background-color: rgba({ar},{ag},{ab}, 230);
                border: none;
                border-radius: 7px;
                color: black;
                padding: 9px;
                font-size: 10px;
                font-weight: 800;
                letter-spacing: 1.5px;
            }}
            QPushButton#saveBtn:hover {{
                background-color: rgba({ar},{ag},{ab}, 255);
            }}
        """)

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(35); shadow.setColor(QColor(0, 0, 0, 200))
        shadow.setOffset(0, 6)
        self.setGraphicsEffect(shadow)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        title = QLabel("PROFILE"); title.setObjectName("title")
        layout.addWidget(title)
        layout.addSpacing(4)

        layout.addWidget(self._section_label("GITHUB"))
        self.gh_input = QLineEdit(); self.gh_input.setPlaceholderText("username")
        layout.addWidget(self._with_prefix("github.com/", self.gh_input))

        layout.addWidget(self._section_label("LINKEDIN"))
        self.li_input = QLineEdit(); self.li_input.setPlaceholderText("username")
        layout.addWidget(self._with_prefix("linkedin.com/in/", self.li_input))

        layout.addWidget(self._section_label("RESUME"))
        self.file_btn = QPushButton("Upload PDF / DOCX")
        self.file_btn.setObjectName("fileBtn")
        self.file_btn.clicked.connect(self._pick_resume)
        layout.addWidget(self.file_btn)

        layout.addSpacing(2)
        self.save_btn = QPushButton("SAVE")
        self.save_btn.setObjectName("saveBtn")
        self.save_btn.clicked.connect(self._save)
        layout.addWidget(self.save_btn)

    def _section_label(self, text):
        lbl = QLabel(text); lbl.setObjectName("sectionLabel")
        return lbl

    def _with_prefix(self, prefix, line_edit):
        frame = QFrame(); frame.setObjectName("prefixGroup")
        h = QHBoxLayout(frame); h.setContentsMargins(0, 0, 4, 0); h.setSpacing(0)
        p = QLabel(prefix); p.setObjectName("prefix")
        h.addWidget(p); h.addWidget(line_edit, 1)
        return frame

    def _pick_resume(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select your resume", "",
            "Documents (*.pdf *.doc *.docx);;All files (*)")
        if path:
            self.data["resume_path"] = path
            name = os.path.basename(path)
            if len(name) > 26:
                name = name[:23] + "..."
            self.file_btn.setText(f"✓  {name}")

    def _save(self):
        self.data["github"] = self.gh_input.text().strip()
        self.data["linkedin"] = self.li_input.text().strip()
        original = self.save_btn.text()
        self.save_btn.setText("SAVED")
        QTimer.singleShot(1200, lambda: self.save_btn.setText(original))

    def get_data(self):
        return dict(self.data)


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
        self.mesh_enabled = True

        self.video = VideoWidget()
        self.video.clicked.connect(self.analyzer.cycle_face)
        self.setCentralWidget(self.video)

        # Overlays parented to video
        self.emotion_pill = GlassPill("", QColor(180, 180, 180))
        self.emotion_pill.setParent(self.video); self.emotion_pill.hide()

        self.mode_badge = ModeBadge()
        self.mode_badge.setParent(self.video)

        self.caption_pill = GlassPill(MODE_CAPTIONS["neutral"], MODE_COLORS["neutral"])
        self.caption_pill.setParent(self.video)

        self.progress_ring = ProgressRing()
        self.progress_ring.setParent(self.video)

        self.profile_panel = ProfilePanel(self.video)

        self.hint = QLabel("click: cycle face   ·   space: capture   ·   m: toggle mesh   ·   esc: quit")
        self.hint.setParent(self.video)
        self.hint.setStyleSheet("color: rgba(160,160,170,180); background: transparent;")
        self.hint.setFont(QFont("Segoe UI", 9))
        self.hint.adjustSize()

        self.flash = TransitionFlash(self.video)

        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(16)

        self.showFullScreen()

    def keyPressEvent(self, e):
        # Esc and Q always quit, regardless of focus
        if e.key() == Qt.Key.Key_Escape:
            self.close()
            return

        # Other shortcuts skip when typing
        focused = QApplication.focusWidget()
        if isinstance(focused, QLineEdit):
            super().keyPressEvent(e)
            return

        if e.key() == Qt.Key.Key_Space:
            self.analyzer.capture()
        elif e.key() == Qt.Key.Key_M:
            self.mesh_enabled = not self.mesh_enabled
        super().keyPressEvent(e)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._reposition()

    def _reposition(self):
        w, h = self.video.width(), self.video.height()
        margin = 24

        # Top-left: emotion pill
        self.emotion_pill.adjustSize()
        self.emotion_pill.move(margin, margin)

        # Top-right: mode badge
        self.mode_badge.adjustSize()
        self.mode_badge.move(w - self.mode_badge.width() - margin, margin)

        # Bottom-center: caption
        self.caption_pill.adjustSize()
        self.caption_pill.move((w - self.caption_pill.width()) // 2,
                               h - self.caption_pill.height() - margin - 8)

        # Bottom-right: profile panel (always visible)
        self.profile_panel.move(w - self.profile_panel.width() - margin,
                                h - self.profile_panel.height() - margin)

        # Above profile panel: progress ring
        self.progress_ring.move(
            w - self.progress_ring.width() - margin,
            h - self.profile_panel.height() - self.progress_ring.height() - margin - 8)

        # Bottom-left: hint
        self.hint.adjustSize()
        self.hint.move(margin, h - self.hint.height() - 16)

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
            self.flash.trigger(self.mode, MODE_COLORS[self.mode])

        h, w = frame.shape[:2]
        pts, raw, metrics = self._analyze(frame, w, h)

        send_pts = pts if self.mesh_enabled else None
        send_raw = raw if self.mesh_enabled else None
        send_metrics = metrics if self.mesh_enabled else None
        self.video.update_frame(frame, send_pts, send_raw, send_metrics, self.mode)

        emo = self.analyzer.get_state()["emotion"]
        if emo and emo != "neutral":
            self.emotion_pill.set_text(emo.upper())
            self.emotion_pill.show()
        else:
            self.emotion_pill.hide()

        progress, streak_target = self.gestures.progress()
        self.progress_ring.set_state(progress, streak_target)

        self._reposition()

    def _analyze(self, frame, w, h):
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
        real = ordered[self.analyzer.locked_idx]
        lm = faces[real]

        self.analyzer.last_clean_frame = frame.copy()
        self.analyzer.last_locked_landmarks = lm

        metrics, pts = compute_metrics(lm, w, h)
        emotion = ("neutral", 0.0)
        if result.face_blendshapes and real < len(result.face_blendshapes):
            emotion = emotion_from_blendshapes(result.face_blendshapes[real])
        self.analyzer.smoother.add(metrics, emotion)
        if time.time() - self.analyzer.smoother.last_update >= 2.0:
            self.analyzer.smoother.commit()

        return pts, lm, metrics

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