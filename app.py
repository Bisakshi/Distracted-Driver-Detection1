from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import base64
import time
import warnings
warnings.filterwarnings('ignore')

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import (
    FaceLandmarker, FaceLandmarkerOptions,
    HandLandmarker, HandLandmarkerOptions,
    RunningMode,
)
import urllib.request
import os

app = Flask(__name__)
CORS(app)

# ── Model download ──────────────────────────────────────────────────────────
FACE_MODEL_PATH = "face_landmarker.task"
FACE_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

HAND_MODEL_PATH = "hand_landmarker.task"
HAND_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

def download_model(path, url, name):
    if not os.path.exists(path):
        print(f"⏳ {name} download ho raha hai...")
        urllib.request.urlretrieve(url, path)
        print(f"✅ {name} downloaded!")

download_model(FACE_MODEL_PATH, FACE_MODEL_URL, "Face Landmarker")
download_model(HAND_MODEL_PATH, HAND_MODEL_URL, "Hand Landmarker")

# ── Face Landmarker setup ───────────────────────────────────────────────────
print("⏳ Face Landmarker load ho raha hai...")
face_options = FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=True,
)
face_landmarker = FaceLandmarker.create_from_options(face_options)
print("✅ Face Landmarker ready!")

# ── Hand Landmarker setup ───────────────────────────────────────────────────
print("⏳ Hand Landmarker load ho raha hai...")
hand_options = HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
hand_landmarker = HandLandmarker.create_from_options(hand_options)
print("✅ Hand Landmarker ready!")

# ── Face Landmark indices ───────────────────────────────────────────────────
LEFT_EYE     = [362, 385, 387, 263, 373, 380]
RIGHT_EYE    = [33,  160, 158, 133, 153, 144]
MOUTH_TOP    = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT   = 61
MOUTH_RIGHT  = 291
NOSE_TIP     = 1
LEFT_TEMPLE  = 234
RIGHT_TEMPLE = 454
CHIN         = 152
FOREHEAD     = 10

# ── Hand Landmark indices ───────────────────────────────────────────────────
WRIST      = 0
THUMB_TIP  = 4
THUMB_MCP  = 2
INDEX_TIP  = 8
INDEX_MCP  = 5
MIDDLE_TIP = 12
MIDDLE_MCP = 9
RING_TIP   = 16
RING_MCP   = 13
PINKY_TIP  = 20
PINKY_MCP  = 17

# ══════════════════════════════════════════════════════════════════════════════
# ── DYNAMIC SETTINGS — yahan sab kuch control hota hai ───────────────────────
# ══════════════════════════════════════════════════════════════════════════════
SETTINGS = {
    # --- Thresholds ---
    "ear_close_threshold":    0.15,   # EAR < is → eyes closed (Sleeping)
    "ear_drowsy_threshold":   0.22,   # EAR < is → half open (Drowsy)
    "ear_blink_threshold":    0.20,   # EAR < is → blink counted
    "mar_yawn_threshold":     0.55,   # MAR > is → yawning
    "perclos_sleep_thresh":   0.60,   # PERCLOS > is → Sleeping
    "perclos_drowsy_thresh":  0.25,   # PERCLOS > is → Drowsy
    "blink_score_sleep":      0.85,   # avg_blink_score > is (+ ear<0.15) → Sleeping
    "blink_score_drowsy":     0.60,   # avg_blink_score > is → Drowsy
    "yawn_count_fatigue":     3,      # yawn_count >= is → Fatigued
    "head_yaw_threshold":     20,     # |yaw| > is → Looking Left/Right
    "head_pitch_threshold":   20,     # |pitch| > is → Looking Up/Down
    "fatigue_warn":           30,     # UI fatigue warn level
    "fatigue_danger":         65,     # UI fatigue danger level
    "blink_rate_high":        25,     # blink/min > is → high
    "blink_rate_low":         8,      # blink/min < is → low
    "distraction_secs":       3,      # distraction timer alert after N secs

    # --- Alert toggles ---
    "alert_phone":            True,
    "alert_texting":          True,
    "alert_eating":           True,
    "alert_hands_off":        True,
    "alert_head_turn":        True,
    "alert_yawning":          True,
    "alert_drowsy":           True,
    "alert_sleeping":         True,
    "alert_finger_move":      False,
    "alert_perclos":          True,
    "alert_blink_high":       True,
    "alert_blink_low":        False,
}

# ── Helper: EAR ─────────────────────────────────────────────────────────────
def eye_aspect_ratio(landmarks, eye_indices, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices]
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (A + B) / (2.0 * C) if C > 0 else 0

# ── Helper: MAR ─────────────────────────────────────────────────────────────
def mouth_aspect_ratio(landmarks, w, h):
    top    = np.array([landmarks[MOUTH_TOP].x * w,    landmarks[MOUTH_TOP].y * h])
    bottom = np.array([landmarks[MOUTH_BOTTOM].x * w, landmarks[MOUTH_BOTTOM].y * h])
    left   = np.array([landmarks[MOUTH_LEFT].x * w,   landmarks[MOUTH_LEFT].y * h])
    right  = np.array([landmarks[MOUTH_RIGHT].x * w,  landmarks[MOUTH_RIGHT].y * h])
    vertical   = np.linalg.norm(top - bottom)
    horizontal = np.linalg.norm(left - right)
    return vertical / horizontal if horizontal > 0 else 0

# ── Helper: Head Pose ───────────────────────────────────────────────────────
def get_head_pose(landmarks, w, h):
    nose     = np.array([landmarks[NOSE_TIP].x * w,     landmarks[NOSE_TIP].y * h])
    l_temp   = np.array([landmarks[LEFT_TEMPLE].x * w,  landmarks[LEFT_TEMPLE].y * h])
    r_temp   = np.array([landmarks[RIGHT_TEMPLE].x * w, landmarks[RIGHT_TEMPLE].y * h])
    chin     = np.array([landmarks[CHIN].x * w,         landmarks[CHIN].y * h])
    forehead = np.array([landmarks[FOREHEAD].x * w,     landmarks[FOREHEAD].y * h])
    face_center_x = (l_temp[0] + r_temp[0]) / 2
    face_width    = abs(r_temp[0] - l_temp[0])
    face_center_y = (forehead[1] + chin[1]) / 2
    face_height   = abs(chin[1] - forehead[1])
    yaw   = (nose[0] - face_center_x) / face_width  * 100 if face_width  > 0 else 0
    pitch = (nose[1] - face_center_y) / face_height * 100 if face_height > 0 else 0
    return float(yaw), float(pitch)

# ── Helper: Finger states ────────────────────────────────────────────────────
def get_finger_states(hand_lms, w, h):
    def pt(idx):
        return np.array([hand_lms[idx].x * w, hand_lms[idx].y * h])
    wrist = pt(WRIST)
    fingers = {
        "thumb":  (THUMB_TIP,  THUMB_MCP),
        "index":  (INDEX_TIP,  INDEX_MCP),
        "middle": (MIDDLE_TIP, MIDDLE_MCP),
        "ring":   (RING_TIP,   RING_MCP),
        "pinky":  (PINKY_TIP,  PINKY_MCP),
    }
    finger_extended = {}
    for name, (tip_idx, mcp_idx) in fingers.items():
        tip = pt(tip_idx)
        mcp = pt(mcp_idx)
        finger_extended[name] = bool(
            np.linalg.norm(tip - wrist) > np.linalg.norm(mcp - wrist) * 1.1
        )
    return finger_extended

# ── Helper: Analyze hands ────────────────────────────────────────────────────
def analyze_hands(hand_result, w, h, face_lms=None):
    result = {
        "hand_count":       0,
        "hands_on_wheel":   False,
        "hands_off_wheel":  True,
        "phone_hand":       False,
        "texting_hand":     False,
        "eating_drinking":  False,
        "finger_movement":  False,
        "fingers_extended": {},
        "hand_positions":   [],
        "gesture":          "none",
    }

    if not hand_result.hand_landmarks:
        return result

    result["hand_count"] = len(hand_result.hand_landmarks)

    mouth_xy = ear_left_xy = ear_right_xy = None
    if face_lms:
        mouth_xy     = (face_lms[MOUTH_TOP].x * w,    face_lms[MOUTH_TOP].y * h)
        ear_left_xy  = (face_lms[LEFT_TEMPLE].x * w,  face_lms[LEFT_TEMPLE].y * h)
        ear_right_xy = (face_lms[RIGHT_TEMPLE].x * w, face_lms[RIGHT_TEMPLE].y * h)

    wheel_hands = 0
    all_fingers = {}

    for idx, hand_lms in enumerate(hand_result.hand_landmarks):
        handedness = "Right"
        if hand_result.handedness and idx < len(hand_result.handedness):
            handedness = hand_result.handedness[idx][0].display_name

        finger_states = get_finger_states(hand_lms, w, h)
        all_fingers[handedness] = finger_states

        wrist_x = hand_lms[WRIST].x * w
        wrist_y = hand_lms[WRIST].y * h
        result["hand_positions"].append({"hand": handedness, "x": round(wrist_x, 1), "y": round(wrist_y, 1)})

        if wrist_y > h * 0.55:
            wheel_hands += 1

        if wrist_y < h * 0.45:
            if ear_left_xy and ear_right_xy:
                wrist_pt = np.array([wrist_x, wrist_y])
                if (np.linalg.norm(wrist_pt - np.array(ear_left_xy)) < w * 0.25 or
                        np.linalg.norm(wrist_pt - np.array(ear_right_xy)) < w * 0.25):
                    result["phone_hand"] = True
            elif wrist_x < w * 0.25 or wrist_x > w * 0.75:
                result["phone_hand"] = True

        if mouth_xy:
            if np.linalg.norm(np.array([wrist_x, wrist_y]) - np.array(mouth_xy)) < w * 0.25:
                result["eating_drinking"] = True
        elif h * 0.2 < wrist_y < h * 0.55 and w * 0.3 < wrist_x < w * 0.7:
            if hand_lms[INDEX_TIP].y * h < wrist_y:
                result["eating_drinking"] = True

        if wrist_y > h * 0.5 and finger_states.get("index") and finger_states.get("thumb"):
            result["texting_hand"] = True

        if sum(1 for v in finger_states.values() if v) >= 3:
            result["finger_movement"] = True

    result["fingers_extended"] = all_fingers
    result["hands_on_wheel"]   = wheel_hands >= 1
    result["hands_off_wheel"]  = wheel_hands == 0

    if   result["phone_hand"]:      result["gesture"] = "phone"
    elif result["eating_drinking"]: result["gesture"] = "eating"
    elif result["texting_hand"]:    result["gesture"] = "texting"
    elif result["hands_on_wheel"]:  result["gesture"] = "driving"
    elif result["hand_count"] > 0:  result["gesture"] = "hands_visible"

    return result

# ── Trackers ──────────────────────────────────────────────────────────────────
class BlinkTracker:
    def __init__(self):
        self.blink_count = 0; self.was_closed = False
        self.blinks_per_minute = 0; self.minute_history = []
    def update(self, ear, threshold=None):
        if threshold is None:
            threshold = SETTINGS["ear_blink_threshold"]
        is_closed = ear < threshold
        if self.was_closed and not is_closed:
            self.blink_count += 1; self.minute_history.append(time.time())
        self.was_closed = is_closed
        now = time.time()
        self.minute_history    = [t for t in self.minute_history if now - t < 60]
        self.blinks_per_minute = len(self.minute_history)

class YawnTracker:
    def __init__(self):
        self.yawn_count = 0; self.was_yawning = False
    def update(self, mar, threshold=None):
        if threshold is None:
            threshold = SETTINGS["mar_yawn_threshold"]
        is_yawning = mar > threshold
        if self.was_yawning and not is_yawning:
            self.yawn_count += 1
        self.was_yawning = is_yawning

class PerclosTracker:
    def __init__(self, window=60):
        self.history = []; self.timestamps = []
        self.window = window
    def update(self, ear):
        threshold = SETTINGS["ear_blink_threshold"]
        now = time.time()
        self.history.append(1 if ear < threshold else 0)
        self.timestamps.append(now)
        cutoff = now - self.window
        while self.timestamps and self.timestamps[0] < cutoff:
            self.timestamps.pop(0); self.history.pop(0)
        return sum(self.history) / len(self.history) if self.history else 0

class FingerMovementTracker:
    def __init__(self, history_size=8):
        self.history = []; self.history_size = history_size
    def update(self, fingers_dict):
        flat = []
        for hand in sorted(fingers_dict.keys()):
            for f in ["thumb", "index", "middle", "ring", "pinky"]:
                flat.append(int(fingers_dict[hand].get(f, False)))
        self.history.append(flat)
        if len(self.history) > self.history_size:
            self.history.pop(0)
        if len(self.history) < 3:
            return False
        changes = sum(
            1 for i in range(1, len(self.history))
            for a, b in zip(self.history[i-1], self.history[i]) if a != b
        )
        return changes >= 3

blink_tracker   = BlinkTracker()
yawn_tracker    = YawnTracker()
perclos_tracker = PerclosTracker()
finger_tracker  = FingerMovementTracker()

# ── Main detection ────────────────────────────────────────────────────────────
def detect_driver(frame):
    S = SETTINGS  # shortcut
    h, w  = frame.shape[:2]
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    face_result = face_landmarker.detect(mp_img)
    hand_result = hand_landmarker.detect(mp_img)

    base = {
        "status": "No Face", "ear": None, "mar": None,
        "blink_rate": blink_tracker.blinks_per_minute,
        "total_blinks": blink_tracker.blink_count,
        "yawn_count": yawn_tracker.yawn_count, "is_yawning": False,
        "perclos": round(perclos_tracker.update(0.30) * 100, 1),
        "head_yaw": 0, "head_pitch": 0, "head_direction": "Center",
        "eye_state": "Open",
        "hand_count": 0, "hands_on_wheel": False, "hands_off_wheel": True,
        "phone_hand": False, "phone_call": False,
        "texting_hand": False, "texting": False,
        "eating_drinking": False, "finger_movement": False,
        "fingers_extended": {}, "hand_positions": [], "gesture": "none",
        "fatigue_score": 0, "alerts": [],
        "settings": S,  # frontend ko current settings bhi milti hain
    }

    face_lms_ref = face_result.face_landmarks[0] if face_result.face_landmarks else None
    hand_data    = analyze_hands(hand_result, w, h, face_lms=face_lms_ref)

    if hand_data["fingers_extended"]:
        hand_data["finger_movement"] = finger_tracker.update(hand_data["fingers_extended"])

    base.update({
        "hand_count": hand_data["hand_count"],
        "hands_on_wheel": hand_data["hands_on_wheel"],
        "hands_off_wheel": hand_data["hands_off_wheel"],
        "phone_hand": hand_data["phone_hand"] and S["alert_phone"],
        "phone_call": hand_data["phone_hand"] and S["alert_phone"],
        "texting_hand": hand_data["texting_hand"] and S["alert_texting"],
        "texting": hand_data["texting_hand"] and S["alert_texting"],
        "eating_drinking": hand_data["eating_drinking"] and S["alert_eating"],
        "finger_movement": hand_data["finger_movement"] and S["alert_finger_move"],
        "fingers_extended": hand_data["fingers_extended"],
        "hand_positions": hand_data["hand_positions"],
        "gesture": hand_data["gesture"],
    })

    if not face_result.face_landmarks:
        alerts = []
        if hand_data["phone_hand"]      and S["alert_phone"]:   alerts.append("📱 Phone use detected!")
        if hand_data["eating_drinking"] and S["alert_eating"]:  alerts.append("🍔 Eating/Drinking while driving!")
        if hand_data["texting_hand"]    and S["alert_texting"]: alerts.append("💬 Texting detected!")
        base["alerts"] = alerts
        return base

    lms = face_result.face_landmarks[0]

    left_ear  = eye_aspect_ratio(lms, LEFT_EYE,  w, h)
    right_ear = eye_aspect_ratio(lms, RIGHT_EYE, w, h)
    ear       = (left_ear + right_ear) / 2.0
    mar       = mouth_aspect_ratio(lms, w, h)
    perclos   = perclos_tracker.update(ear)

    blink_tracker.update(ear)
    yawn_tracker.update(mar)
    currently_yawning = yawn_tracker.was_yawning

    yaw_thresh   = S["head_yaw_threshold"]
    pitch_thresh = S["head_pitch_threshold"]
    yaw, pitch = get_head_pose(lms, w, h)
    if   yaw < -yaw_thresh:   head_dir = "Looking Left"
    elif yaw >  yaw_thresh:   head_dir = "Looking Right"
    elif pitch < -pitch_thresh: head_dir = "Looking Up"
    elif pitch >  pitch_thresh: head_dir = "Looking Down"
    else:                       head_dir = "Center"

    eye_blink_left = eye_blink_right = 0.0
    if face_result.face_blendshapes:
        for bs in face_result.face_blendshapes[0]:
            if bs.category_name == "eyeBlinkLeft":  eye_blink_left  = bs.score
            if bs.category_name == "eyeBlinkRight": eye_blink_right = bs.score
    avg_blink_score = (eye_blink_left + eye_blink_right) / 2.0

    if   ear < S["ear_close_threshold"]: eye_state = "Closed"
    elif ear < S["ear_drowsy_threshold"]: eye_state = "Half Open"
    else:                                 eye_state = "Open"

    # ── Fatigue score calculation ──────────────────────────────────────────
    fatigue  = min(perclos * 150, 40)
    fatigue += min(blink_tracker.blinks_per_minute * 0.5, 15)
    fatigue += min(yawn_tracker.yawn_count * 5, 25)
    if currently_yawning: fatigue += 10
    fatigue = min(int(fatigue), 100)

    # ── Status + Alerts (settings-controlled) ─────────────────────────────
    alerts = []
    status = "Active"

    is_sleeping = (
        S["alert_sleeping"] and
        (perclos > S["perclos_sleep_thresh"] or
         (avg_blink_score > S["blink_score_sleep"] and ear < S["ear_close_threshold"]))
    )
    is_drowsy = (
        S["alert_drowsy"] and
        (perclos > S["perclos_drowsy_thresh"] or
         avg_blink_score > S["blink_score_drowsy"] or
         ear < S["ear_drowsy_threshold"])
    )
    is_fatigued = (
        S["alert_yawning"] and
        currently_yawning and
        yawn_tracker.yawn_count >= S["yawn_count_fatigue"]
    )
    is_phone     = hand_data["phone_hand"]      and S["alert_phone"]
    is_eating    = hand_data["eating_drinking"] and S["alert_eating"]
    is_texting   = hand_data["texting_hand"]    and S["alert_texting"]
    is_hands_off = hand_data["hands_off_wheel"] and hand_data["hand_count"] == 0 and S["alert_hands_off"]
    is_head_turn = head_dir != "Center"          and S["alert_head_turn"]

    if is_sleeping:
        status = "Sleeping"
        alerts.append("⚠️ Eyes closed! Pull over safely!")
    elif is_drowsy:
        status = "Drowsy"
        alerts.append("😴 Drowsiness detected — Take a break!")
    elif is_fatigued:
        status = "Fatigued"
        alerts.append("🥱 Multiple yawns — Fatigue warning!")
    elif is_phone:
        status = "Distracted"
        alerts.append("📱 Phone use detected while driving!")
    elif is_eating:
        status = "Distracted"
        alerts.append("🍔 Eating/Drinking while driving!")
    elif is_texting:
        status = "Distracted"
        alerts.append("💬 Texting detected — Eyes on road!")
    elif is_hands_off:
        status = "Distracted"
        alerts.append("🚗 No hands detected on wheel!")
    elif is_head_turn:
        status = "Distracted"
        alerts.append(f"👀 Driver attention: {head_dir}")

    # Extra info alerts
    if S["alert_perclos"] and perclos > S["perclos_drowsy_thresh"] and status == "Active":
        alerts.append(f"📊 PERCLOS high: {perclos*100:.1f}%")
    if S["alert_blink_high"] and blink_tracker.blinks_per_minute > S["blink_rate_high"]:
        alerts.append(f"👁 Blink rate high: {blink_tracker.blinks_per_minute}/min")
    if S["alert_blink_low"] and 0 < blink_tracker.blinks_per_minute < S["blink_rate_low"]:
        alerts.append(f"👁 Blink rate low: {blink_tracker.blinks_per_minute}/min")
    if S["alert_finger_move"] and hand_data["finger_movement"]:
        alerts.append("✋ Finger activity detected")

    return {
        "status": status,
        "ear": round(ear, 3),
        "mar": round(mar, 3),
        "blink_rate": blink_tracker.blinks_per_minute,
        "total_blinks": blink_tracker.blink_count,
        "yawn_count": yawn_tracker.yawn_count,
        "is_yawning": bool(currently_yawning),
        "perclos": round(perclos * 100, 1),
        "head_yaw": round(yaw, 1),
        "head_pitch": round(pitch, 1),
        "head_direction": head_dir,
        "eye_state": eye_state,
        "hand_count": hand_data["hand_count"],
        "hands_on_wheel": hand_data["hands_on_wheel"],
        "hands_off_wheel": hand_data["hands_off_wheel"],
        "phone_hand": is_phone,
        "phone_call": is_phone,
        "texting_hand": is_texting,
        "texting": is_texting,
        "eating_drinking": is_eating,
        "finger_movement": hand_data["finger_movement"] and S["alert_finger_move"],
        "fingers_extended": hand_data["fingers_extended"],
        "hand_positions": hand_data["hand_positions"],
        "gesture": hand_data["gesture"],
        "fatigue_score": fatigue,
        "alerts": alerts,
        "settings": S,
    }


# ── Flask routes ──────────────────────────────────────────────────────────────
@app.route('/detect', methods=['POST'])
def detect():
    try:
        if not request.is_json:
            file = request.files.get('frame')
            if file is None:
                return jsonify({"error": "No frame received"}), 400
            img_bytes = file.read()
        else:
            data = request.json.get('image', '')
            if ',' in data:
                data = data.split(',')[1]
            img_bytes = base64.b64decode(data)

        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Frame decode failed"}), 400

        result = detect_driver(frame)
        return jsonify(result)

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback; traceback.print_exc()
        return jsonify({"status": "Active", "error": str(e)}), 500


@app.route('/reset', methods=['POST'])
def reset_counters():
    global blink_tracker, yawn_tracker, perclos_tracker, finger_tracker
    blink_tracker   = BlinkTracker()
    yawn_tracker    = YawnTracker()
    perclos_tracker = PerclosTracker()
    finger_tracker  = FingerMovementTracker()
    return jsonify({"message": "Counters reset"})


# ══════════════════════════════════════════════════════════════════════════════
# ── NEW ROUTE: /update_settings — frontend se settings receive karo ──────────
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/update_settings', methods=['POST'])
def update_settings():
    """
    Frontend se JSON aata hai jisme thresholds + alert toggles hote hain.
    Example payload:
    {
      "ear_close_threshold": 0.15,
      "mar_yawn_threshold": 0.60,
      "alert_phone": true,
      "alert_texting": false,
      ...
    }
    """
    global SETTINGS
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data received"}), 400

        updated = []
        for key, value in data.items():
            if key in SETTINGS:
                # Type safety: bool stays bool, numbers stay numbers
                expected_type = type(SETTINGS[key])
                if expected_type == bool:
                    SETTINGS[key] = bool(value)
                elif expected_type == float:
                    SETTINGS[key] = float(value)
                elif expected_type == int:
                    SETTINGS[key] = int(value)
                else:
                    SETTINGS[key] = value
                updated.append(key)
            else:
                print(f"[WARN] Unknown setting key: {key}")

        print(f"✅ Settings updated: {updated}")
        return jsonify({"message": "Settings updated", "updated": updated, "current": SETTINGS})

    except Exception as e:
        print(f"[ERROR] Settings update failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/get_settings', methods=['GET'])
def get_settings():
    """Current settings return karo"""
    return jsonify(SETTINGS)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})


if __name__ == '__main__':
    app.run(debug=False, port=5000)