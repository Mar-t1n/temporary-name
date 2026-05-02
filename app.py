"""
FaceJudge - desktop app
Minimal overlay UI. Mode controlled by hand gestures:
  Thumb Up      -> glaze
  Thumb Sideways -> neutral
  Thumb Down    -> hate
Run: pip install opencv-python mediapipe numpy pillow
     python app.py
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageDraw, ImageFont
import os
import time
import urllib.request
import platform
from cvModule import FaceAnalyzer

# ============ CONFIG ============
GESTURE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
GESTURE_MODEL_PATH = "gesture_recognizer.task"

GESTURE_HOLD_FRAMES = 10
GESTURE_COOLDOWN = 2.0
MISS_TOLERANCE = 3   # how many missed frames before streak resets

MODE_COLORS = {
    "neutral": (0, 255, 220),
    "glaze":   (255, 112, 196),
    "hate":    (80, 80, 255),
}

MODE_CAPTIONS = {
    "neutral": "Thumb up to glaze, thumb down to hate, sideways for neutral",
    "glaze":   "Glazing in progress...",
    "hate":    "Roasting in progress...",
}


# ============ DOWNLOAD GESTURE MODEL ============
if not os.path.exists(GESTURE_MODEL_PATH):
    print("Downloading gesture recognizer model...")
    urllib.request.urlretrieve(GESTURE_MODEL_URL, GESTURE_MODEL_PATH)


# ============ FONT LOADING ============
def find_system_font(weight="regular"):
    """Locate a clean sans-serif font on Win/Mac/Linux."""
    sysname = platform.system()
    candidates = []
    if sysname == "Windows":
        base = "C:/Windows/Fonts"
        if weight == "bold":
            candidates = ["segoeuib.ttf", "arialbd.ttf"]
        else:
            candidates = ["segoeui.ttf", "arial.ttf"]
        candidates = [os.path.join(base, c) for c in candidates]
    elif sysname == "Darwin":
        if weight == "bold":
            candidates = ["/System/Library/Fonts/HelveticaNeue.ttc",
                          "/Library/Fonts/Arial Bold.ttf"]
        else:
            candidates = ["/System/Library/Fonts/HelveticaNeue.ttc",
                          "/Library/Fonts/Arial.ttf"]
    else:  # Linux
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if weight == "bold"
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
        ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


FONT_REGULAR = find_system_font("regular")
FONT_BOLD = find_system_font("bold") or FONT_REGULAR

_font_cache = {}
def get_font(size, bold=False):
    key = (size, bold)
    if key not in _font_cache:
        path = FONT_BOLD if bold else FONT_REGULAR
        try:
            _font_cache[key] = ImageFont.truetype(path, size) if path else ImageFont.load_default()
        except Exception:
            _font_cache[key] = ImageFont.load_default()
    return _font_cache[key]


# ============ TEXT DRAWING (PIL -> OpenCV) ============
def draw_text(frame, text, pos, size=20, color=(255, 255, 255), bold=False, anchor="lt"):
    """Render TrueType text onto a BGR OpenCV frame.

    anchor: PIL anchor string. 'lt'=left-top, 'mm'=middle-middle, 'rt'=right-top, etc.
    Returns (width, height) of the rendered text.
    """
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    font = get_font(size, bold)
    # PIL color is RGB; we received BGR
    rgb = (color[2], color[1], color[0])
    draw.text(pos, text, font=font, fill=rgb, anchor=anchor)
    frame[:] = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    bbox = draw.textbbox(pos, text, font=font, anchor=anchor)
    return (bbox[2] - bbox[0], bbox[3] - bbox[1])


def text_size(text, size=20, bold=False):
    font = get_font(size, bold)
    bbox = font.getbbox(text)
    return (bbox[2] - bbox[0], bbox[3] - bbox[1])


# ============ THUMB SIDEWAYS DETECTION ============
# MediaPipe hand landmarks: 0=wrist, 1=thumb_cmc, 2=thumb_mcp, 3=thumb_ip, 4=thumb_tip
def classify_thumb_orientation(hand_landmarks):
    """Return 'up', 'down', 'side', or None based on raw landmarks.

    Used as a fallback / supplement when MediaPipe's gesture classifier returns
    something other than Thumb_Up / Thumb_Down (it has no sideways category).
    """
    if not hand_landmarks:
        return None
    wrist = hand_landmarks[0]
    thumb_tip = hand_landmarks[4]
    thumb_mcp = hand_landmarks[2]

    # Vector from MCP to TIP (the thumb's pointing direction)
    dx = thumb_tip.x - thumb_mcp.x
    dy = thumb_tip.y - thumb_mcp.y

    # Reference: thumb must be extended away from wrist for any of these to count
    if abs(thumb_tip.y - wrist.y) < 0.05 and abs(thumb_tip.x - wrist.x) < 0.05:
        return None

    # Image y increases downward
    if abs(dy) > abs(dx) * 1.3:
        return "down" if dy > 0 else "up"
    elif abs(dx) > abs(dy) * 1.3:
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
        self.streak_mode = None     # the candidate mode being held
        self.streak_count = 0
        self.miss_count = 0
        self.last_switch = 0.0

    def detect(self, rgb_frame):
        """Return one of: 'glaze', 'hate', 'neutral', or None."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        ts_ms = int((time.time() - self.start) * 1000)
        result = self.recognizer.recognize_for_video(mp_image, ts_ms)

        if not result.hand_landmarks:
            return None

        landmarks = result.hand_landmarks[0]

        # Use MediaPipe's classifier first for up/down (high confidence)
        if result.gestures and result.gestures[0]:
            top = result.gestures[0][0]
            if top.score >= 0.6:
                if top.category_name == "Thumb_Up":
                    return "glaze"
                if top.category_name == "Thumb_Down":
                    return "hate"

        # Fallback to landmark-based classification (catches sideways)
        orient = classify_thumb_orientation(landmarks)
        if orient == "up":
            return "glaze"
        if orient == "down":
            return "hate"
        if orient == "side":
            return "neutral"
        return None

    def update_mode(self, current_mode, target):
        """Track gesture streak with miss-tolerance to prevent re-triggers."""
        now = time.time()

        if target is None:
            # Hand momentarily lost — tolerate a few frames before resetting
            self.miss_count += 1
            if self.miss_count > MISS_TOLERANCE:
                self.streak_mode = None
                self.streak_count = 0
            return current_mode

        self.miss_count = 0

        # If we just switched, lock out same-target re-triggers until cooldown ends
        if (target == current_mode
                and now - self.last_switch < GESTURE_COOLDOWN):
            self.streak_mode = None
            self.streak_count = 0
            return current_mode

        # Build / extend streak
        if target == self.streak_mode:
            self.streak_count += 1
        else:
            self.streak_mode = target
            self.streak_count = 1

        # Commit switch
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


# ============ OVERLAY UI ============
def draw_rounded_rect(img, x1, y1, x2, y2, color, radius=12, alpha=0.6, border=None):
    """Translucent rounded rectangle, optionally with a colored border."""
    overlay = img.copy()
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, -1)
    cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, -1)
    cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, -1)
    cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    if border is not None:
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), border, 1, cv2.LINE_AA)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), border, 1, cv2.LINE_AA)
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), border, 1, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), border, 1, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, border, 1, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, border, 1, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, border, 1, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, border, 1, cv2.LINE_AA)


def draw_overlay(frame, mode, emotion, progress, streak_target):
    h, w = frame.shape[:2]
    color = MODE_COLORS[mode]

    # Top accent bar
    cv2.rectangle(frame, (0, 0), (w, 4), color, -1)

    # Mode pill (top-left)
    label = mode.upper()
    tw, th = text_size(label, size=18, bold=True)
    pad_x, pad_y = 14, 8
    x1, y1 = 18, 18
    x2, y2 = x1 + tw + pad_x * 2, y1 + th + pad_y * 2
    draw_rounded_rect(frame, x1, y1, x2, y2, (15, 15, 20), radius=10, alpha=0.75, border=color)
    draw_text(frame, label, (x1 + pad_x, y1 + pad_y - 2),
              size=18, color=color, bold=True)

    # Emotion (top-right) - only when non-neutral
    if emotion and emotion != "neutral":
        emo = emotion.upper()
        ew, eh = text_size(emo, size=14, bold=True)
        ex2 = w - 20
        ex1 = ex2 - ew - 24
        ey1, ey2 = 18, 18 + eh + 14
        draw_rounded_rect(frame, ex1, ey1, ex2, ey2, (15, 15, 20), radius=8, alpha=0.7, border=(180, 180, 180))
        draw_text(frame, emo, (ex1 + 12, ey1 + 6),
                  size=14, color=(220, 220, 220), bold=True)

    # Bottom caption pill
    caption = MODE_CAPTIONS[mode]
    cw, ch = text_size(caption, size=18)
    cap_pad_x, cap_pad_y = 22, 14
    box_w = cw + cap_pad_x * 2
    box_h = ch + cap_pad_y * 2
    bx1 = (w - box_w) // 2
    by1 = h - box_h - 30
    bx2, by2 = bx1 + box_w, by1 + box_h
    draw_rounded_rect(frame, bx1, by1, bx2, by2, (15, 15, 20), radius=14, alpha=0.7, border=color)
    draw_text(frame, caption, (bx1 + cap_pad_x, by1 + cap_pad_y - 2),
              size=18, color=(240, 240, 240))

    # Gesture progress ring (bottom-left)
    if progress > 0 and streak_target:
        target_color = MODE_COLORS.get(streak_target, color)
        center = (52, h - 52)
        radius = 22
        # Background ring
        cv2.circle(frame, center, radius, (45, 45, 50), 3, cv2.LINE_AA)
        # Progress arc
        end_angle = int(360 * progress)
        if end_angle > 0:
            cv2.ellipse(frame, center, (radius, radius), -90, 0, end_angle,
                        target_color, 3, cv2.LINE_AA)
        # Inner dot
        cv2.circle(frame, center, 6, target_color, -1, cv2.LINE_AA)
        # Label next to ring
        draw_text(frame, streak_target.upper(),
                  (center[0] + radius + 12, center[1] - 8),
                  size=13, color=target_color, bold=True)

    # Hint (bottom-right)
    hint = "click: cycle face   space: capture   q: quit"
    hw_, hh_ = text_size(hint, size=12)
    draw_text(frame, hint, (w - hw_ - 18, h - hh_ - 18),
              size=12, color=(140, 140, 140))

    return frame


# ============ MAIN APP ============
class App:
    def __init__(self):
        self.analyzer = FaceAnalyzer()
        self.gestures = GestureDetector()
        self.cap = cv2.VideoCapture(0)
        # Bump capture resolution for crisper text rendering
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.mode = "neutral"
        self.cycle_request = False
        self.capture_request = False

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.cycle_request = True

    def run(self):
        win = "FaceJudge"
        cv2.namedWindow(win, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setMouseCallback(win, self.on_mouse)

        while self.cap.isOpened():
            ok, frame = self.cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)

            if self.cycle_request:
                self.analyzer.cycle_face()
                self.cycle_request = False
            if self.capture_request:
                self.analyzer.capture()
                self.capture_request = False

            # Gesture detection on clean frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            target = self.gestures.detect(rgb)
            new_mode = self.gestures.update_mode(self.mode, target)
            if new_mode != self.mode:
                self.mode = new_mode
                self.analyzer.set_mode(self.mode)

            # Face mesh + analysis
            frame, _ = self.analyzer.process_frame(frame)

            state = self.analyzer.get_state()
            progress, streak_target = self.gestures.progress()
            frame = draw_overlay(frame, self.mode, state["emotion"],
                                 progress, streak_target)

            cv2.imshow(win, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                self.capture_request = True

        self.cap.release()
        cv2.destroyAllWindows()
        self.analyzer.close()
        self.gestures.close()


if __name__ == "__main__":
    App().run()