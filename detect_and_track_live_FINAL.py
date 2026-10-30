import cv2
import numpy as np
import time
import psutil
import json
import yaml
from ultralytics import YOLO

from ml_adapter import MLAdapter
from rule_engine.smoothing import ViolationManager
from rule_engine.zones import load_zones
from replay import process_observation


# ============================================================
# CONFIGURATION
# ============================================================

print("Loading configuration files...")

try:
    with open("configs/thresholds.yaml", "r") as f:
        thresholds = yaml.safe_load(f) or {}

    K = thresholds.get("voting_k", 3)
    N = thresholds.get("voting_n", 5)
    TAU = thresholds.get("face_touch_threshold", 0.4)
    CONF_THRESH = thresholds.get("detection_confidence", 0.5)

    print(f"    Thresholds loaded: k={K}, n={N}, tau={TAU}")

except Exception as e:
    print(f"    Warning: Could not load thresholds.yaml ({e}). Using defaults.")
    K, N, TAU, CONF_THRESH = 3, 5, 0.4, 0.5


try:
    with open("configs/sop.yaml", "r") as f:
        sop = yaml.safe_load(f) or {}

    GOWNING_ORDER = sop.get(
        "gowning_order",
        ["hairnet", "mask", "gloves"]
    )

    print(f"    Gowning order loaded: {GOWNING_ORDER}")

except Exception as e:
    print(f"    Warning: Could not load sop.yaml ({e}). Using defaults.")
    GOWNING_ORDER = ["hairnet", "mask", "gloves"]


# Real zone file.
try:
    zones = load_zones("configs/zones.yaml")
    print("    Zones loaded successfully from configs/zones.yaml.")
    USE_ZONES = True

except Exception as e:
    print(f"    Warning: Could not load zones ({e}).")
    print("    Falling back to fixed production zone.")

    zones = {
        "production": {
            "required_ppe": ["hairnet", "mask", "gloves"]
        }
    }

    USE_ZONES = False


MODEL_PATH = "best.pt"

# 0 = laptop webcam
VIDEO_PATH = 0

FIXED_ZONE = "production"

# Webcam frames are resized to match zones.yaml coordinates.
FRAME_WIDTH = 640
FRAME_HEIGHT = 360

# Ignore tiny background person detections.
MIN_PERSON_AREA = 12000

# Keep confirmed alert visible for a while.
ALERT_HOLD_FRAMES = 45

MAX_READ_FAILURES = 10
READ_RETRY_DELAY = 0.2

# YOLO Pose face-touch settings.
FACE_TOUCH_THRESHOLD = 0.60
POSE_KEYPOINT_CONF = 0.25

# Cumulative active handwash & oscillation tracking per person
handwash_accumulated_time = {}  # track_id -> float (seconds)
handwash_prev_wrists = {}       # track_id -> list of wrist coordinates
handwash_completed = {}         # track_id -> bool
previous_zone_for_handwash = {} # track_id -> str
handwash_wrist_history = {}     # track_id -> list of recent wrist midpoints

MOVEMENT_SENSITIVITY_THRESHOLD = 3.5  # Minimum pixel displacement per frame
HANDWASH_REQUIRED_SECONDS = 5.0
OSCILLATION_WINDOW_FRAMES = 12      # Rolling window size for motion history
MIN_REVERSALS_REQUIRED = 2          # Minimum directional changes required for scrubbing


# ============================================================
# MODEL + RULE ENGINE SETUP
# ============================================================

print("Loading YOLO models...")

ppe_model = YOLO(MODEL_PATH)
person_model = YOLO("yolov8n.pt")
pose_model = YOLO("yolov8n-pose.pt")
print("Face-touch detection enabled with YOLO Pose.")

print("Initializing Rule Engine...")

adapter = MLAdapter(hold_frames=5)

manager = ViolationManager(
    k=K,
    n=N,
    cooldown_frames=30
)

person_states = {}

# track_id -> {"until": frame_number, "text": label}
active_alerts = {}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def is_contained(item_box, person_box, threshold=0.6):
    """
    Check whether enough of the PPE bounding box lies inside
    the detected person's bounding box.
    """
    ix1 = max(item_box[0], person_box[0])
    iy1 = max(item_box[1], person_box[1])
    ix2 = min(item_box[2], person_box[2])
    iy2 = min(item_box[3], person_box[3])

    item_area = (
        (item_box[2] - item_box[0])
        * (item_box[3] - item_box[1])
    )

    inter_area = (
        max(0, ix2 - ix1)
        * max(0, iy2 - iy1)
    )

    if item_area <= 0:
        return False

    return (inter_area / item_area) >= threshold


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def extract_pose_people(pose_result):
    people = []
    if pose_result.boxes is None or pose_result.keypoints is None:
        return people
    if len(pose_result.boxes) == 0:
        return people
    boxes = pose_result.boxes.xyxy.cpu().numpy()
    xy = pose_result.keypoints.xy.cpu().numpy()
    conf = None
    if pose_result.keypoints.conf is not None:
        conf = pose_result.keypoints.conf.cpu().numpy()
    for i in range(len(boxes)):
        people.append({"box": boxes[i], "xy": xy[i], "conf": conf[i] if conf is not None else None})
    return people


def keypoint_valid(xy, conf, idx):
    p = xy[idx]
    if p[0] <= 0 or p[1] <= 0:
        return False
    if conf is not None and conf[idx] < POSE_KEYPOINT_CONF:
        return False
    return True


def check_face_touch(person_box, pose_people):
    best_pose = None
    best_iou = 0.0
    for pose_person in pose_people:
        iou = box_iou(person_box, pose_person["box"])
        if iou > best_iou:
            best_iou = iou
            best_pose = pose_person
    if best_pose is None or best_iou < 0.10:
        return False, None, None
    xy = best_pose["xy"]
    conf = best_pose["conf"]
    NOSE, LSHOULDER, RSHOULDER, LWRIST, RWRIST = 0, 5, 6, 9, 10
    if not all(keypoint_valid(xy, conf, i) for i in [NOSE, LSHOULDER, RSHOULDER]):
        return False, None, best_pose
    shoulder_width = np.linalg.norm(xy[LSHOULDER] - xy[RSHOULDER])
    if shoulder_width < 1:
        return False, None, best_pose
    distances = []
    if keypoint_valid(xy, conf, LWRIST):
        distances.append(np.linalg.norm(xy[NOSE] - xy[LWRIST]) / shoulder_width)
    if keypoint_valid(xy, conf, RWRIST):
        distances.append(np.linalg.norm(xy[NOSE] - xy[RWRIST]) / shoulder_width)
    if not distances:
        return False, None, best_pose
    distance = float(min(distances))
    return distance < FACE_TOUCH_THRESHOLD, distance, best_pose


def draw_face_touch_points(frame, pose_person, touching):
    if pose_person is None:
        return
    xy, conf = pose_person["xy"], pose_person["conf"]
    color = (0, 0, 255) if touching else (0, 255, 255)
    for idx in [0, 5, 6, 9, 10]:
        if keypoint_valid(xy, conf, idx):
            x, y = map(int, xy[idx])
            cv2.circle(frame, (x, y), 3, color, -1)


def event_track_id(event):
    """
    Extract track id from a rule-engine event.
    """
    if not isinstance(event, dict):
        return None

    track_id = event.get("track_id")

    if (
        track_id is None
        and isinstance(event.get("violation"), dict)
    ):
        track_id = event["violation"].get("track_id")

    try:
        return int(track_id) if track_id is not None else None
    except (TypeError, ValueError):
        return None


def event_short_text(event):
    """
    Convert a rule-engine event into short text suitable for display.
    """
    if not isinstance(event, dict):
        return "PPE violation"

    event_type = str(
        event.get("type")
        or event.get("violation_type")
        or event.get("rule_name")
        or "PPE violation"
    ).lower()

    if "face" in event_type:
        return "Face touch"

    item = event.get("item")

    if item:
        return f"Missing {str(item).lower()}"

    if "unauthor" in event_type:
        return "Unauthorised zone"

    if "gowning" in event_type:
        return "Gowning sequence"

    if "zone" in event_type:
        return "Zone violation"

    return "PPE violation"


def draw_zone_overlay(frame, zones_dict):
    """
    Draw configured zone polygons and labels.
    """
    if not USE_ZONES:
        return

    for zone_name, zone_data in zones_dict.items():
        polygon = zone_data.get("polygon", [])

        if not polygon:
            continue

        pts = np.array(
            polygon,
            dtype=np.int32
        ).reshape((-1, 1, 2))

        cv2.polylines(
            frame,
            [pts],
            isClosed=True,
            color=(0, 255, 255),
            thickness=2
        )

        label_x = int(polygon[0][0]) + 8
        label_y = int(polygon[0][1]) + 22

        cv2.putText(
            frame,
            f"ZONE: {zone_name.upper()}",
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2
        )


def draw_person_panel(
    frame,
    track_id,
    box,
    ppe_list,
    current_zone=None,
    face_touch=False,
    face_distance=None,
    handwash_time=None,
    handwash_done=False,
    alert=None
):
    """Draw a clean person box and compact status panel."""
    px1, py1, px2, py2 = box

    if alert:
        color = (0, 0, 255)
        status_text = f"ALERT: {alert['text']}"
    elif face_touch:
        color = (0, 165, 255)
        status_text = "Face touch pending"
    elif handwash_time is not None and not handwash_done:
        color = (0, 165, 255)
        status_text = "Handwash in progress"
    elif handwash_done:
        color = (0, 255, 0)
        status_text = "Handwash complete"
    else:
        color = (0, 255, 0)
        status_text = "No confirmed violation"

    cv2.rectangle(frame, (px1, py1), (px2, py2), color, 3)

    unique_ppe = []
    for ppe in ppe_list:
        if ppe not in unique_ppe:
            unique_ppe.append(ppe)

    ppe_text = ",".join(unique_ppe[:2]) if unique_ppe else "none"

    line1 = f"ID {track_id}"
    line2 = f"ZONE: {(current_zone or 'outside').upper()}"
    line3 = status_text

    if handwash_time is not None:
        if handwash_done:
            line4 = "Handwash: COMPLETE"
        else:
            line4 = f"Handwash: {handwash_time:.1f}/{HANDWASH_REQUIRED_SECONDS:.0f}s"
    elif face_distance is not None:
        line4 = f"PPE: {ppe_text} | Face d={face_distance:.2f}"
    else:
        line4 = f"PPE: {ppe_text}"

    panel_height = 78
    panel_width = min(300, frame.shape[1] - px1 - 5)
    panel_x1 = max(0, px1)

    if py1 >= panel_height + 5:
        panel_y1 = py1 - panel_height
        panel_y2 = py1
    else:
        panel_y1 = py1
        panel_y2 = min(frame.shape[0], py1 + panel_height)

    panel_x2 = min(frame.shape[1] - 1, panel_x1 + panel_width)

    cv2.rectangle(
        frame,
        (panel_x1, panel_y1),
        (panel_x2, panel_y2),
        color,
        -1
    )

    text_x = panel_x1 + 7

    cv2.putText(
        frame, line1, (text_x, panel_y1 + 16),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
    )
    cv2.putText(
        frame, line2, (text_x, panel_y1 + 34),
        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1
    )
    cv2.putText(
        frame, line3, (text_x, panel_y1 + 52),
        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1
    )
    cv2.putText(
        frame, line4, (text_x, panel_y1 + 70),
        cv2.FONT_HERSHEY_SIMPLEX, 0.39, (255, 255, 255), 1
    )


# ============================================================
# LIVE PIPELINE
# ============================================================

print(f"Starting live processing on {VIDEO_PATH}...")

all_payloads = []
frame_idx = 0
read_failures = 0

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video source: {VIDEO_PATH}"
    )

try:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            read_failures += 1
            print(f"Camera frame read failed ({read_failures}/{MAX_READ_FAILURES})")

            if read_failures >= MAX_READ_FAILURES:
                print("Camera unavailable. Stopping.")
                break

            time.sleep(READ_RETRY_DELAY)
            continue

        read_failures = 0

        # Match zones.yaml coordinate system.
        frame = cv2.resize(
            frame,
            (FRAME_WIDTH, FRAME_HEIGHT)
        )

        t_start = time.perf_counter()

        ppe_results = ppe_model(
            frame,
            conf=CONF_THRESH,
            verbose=False
        )[0]

        # COCO class 0 = person only
        person_results = person_model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            conf=0.4,
            verbose=False
        )[0]

        pose_results = pose_model(
            frame,
            classes=[0],
            conf=0.4,
            verbose=False
        )[0]
        pose_people = extract_pose_people(pose_results)

        persons = []
        ppe_items = []

        if (
            person_results.boxes is not None
            and person_results.boxes.id is not None
        ):
            p_boxes = person_results.boxes.xyxy.cpu().numpy()
            p_ids = person_results.boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(p_boxes, p_ids):
                x1, y1, x2, y2 = box
                area = (x2 - x1) * (y2 - y1)

                if area < MIN_PERSON_AREA:
                    continue

                persons.append({
                    "track_id": int(track_id),
                    "box": box
                })

        if ppe_results.boxes is not None:
            ppe_boxes = ppe_results.boxes.xyxy.cpu().numpy()
            ppe_classes = ppe_results.boxes.cls.cpu().numpy().astype(int)
            ppe_confidences = ppe_results.boxes.conf.cpu().numpy()

            for box, cls_id, confidence in zip(ppe_boxes, ppe_classes, ppe_confidences):
                class_name = ppe_model.names[int(cls_id)]
                ppe_items.append({
                    "class_name": class_name,
                    "box": box,
                    "confidence": float(confidence)
                })

        frame_payload = {
            "frame_id": frame_idx,
            "persons": []
        }

        display_people = {}

        for person in persons:
            track_id = person["track_id"]
            px1, py1, px2, py2 = map(int, person["box"])

            face_touch, face_distance, matched_pose = check_face_touch(
                person["box"],
                pose_people
            )

            draw_face_touch_points(
                frame,
                matched_pose,
                face_touch
            )

            associated_ppe = []

            for item in ppe_items:
                if is_contained(item["box"], person["box"], threshold=0.6):
                    associated_ppe.append(item["class_name"])
                    ix1, iy1, ix2, iy2 = map(int, item["box"])

                    cv2.rectangle(
                        frame,
                        (ix1, iy1),
                        (ix2, iy2),
                        (255, 191, 0),
                        1
                    )
                    cv2.putText(
                        frame,
                        item["class_name"],
                        (ix1, max(15, iy1 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.38,
                        (255, 191, 0),
                        1
                    )

            frame_payload["persons"].append({
                "track_id": track_id,
                "box": [px1, py1, px2, py2],
                "foot_point": (int((px1 + px2) / 2), py2),
                "associated_ppe": associated_ppe,
                "face_touch_flag": bool(face_touch)
            })

            display_people[track_id] = {
                "box": (px1, py1, px2, py2),
                "ppe": associated_ppe,
                "face_touch": bool(face_touch),
                "face_distance": face_distance,
                "matched_pose": matched_pose,
                "handwash_time": None,
                "handwash_done": handwash_completed.get(track_id, False)
            }

        events = process_observation(
            observation=frame_payload,
            adapter=adapter,
            person_states=person_states,
            zones=zones,
            manager=manager,
            frame=frame,
            fixed_zone=FIXED_ZONE,
            use_zones=USE_ZONES,
            face_touch=True,
            required_gowning_order=GOWNING_ORDER
        )

# --------------------------------------------------------
        # HANDWASH SENSITIVITY, WRIST ZONE & OSCILLATORY LOGIC (REAL-TIME DT)
        # --------------------------------------------------------
        # Measure actual elapsed wall-clock time for this frame iteration
        current_frame_time = time.perf_counter()
        frame_dt = current_frame_time - t_start
        # Clamp frame_dt between 0.01s and 0.2s to prevent lag spikes from throwing off the timer
        frame_dt = min(max(frame_dt, 0.01), 0.2)

        handwash_zone_data = zones.get("handwash", {}) if isinstance(zones, dict) else {}
        handwash_polygon = handwash_zone_data.get("polygon", [])

        def is_point_in_polygon(pt, polygon):
            if not polygon:
                return True
            pts = np.array(polygon, dtype=np.int32)
            return cv2.pointPolygonTest(pts, (float(pt[0]), float(pt[1])), False) >= 0

        for track_id, info in display_people.items():
            person_state = person_states.get(track_id)
            if person_state is None:
                continue

            current_zone = person_state.current_zone
            info["current_zone"] = current_zone
            previous_zone = previous_zone_for_handwash.get(track_id)

            matched_pose = info.get("matched_pose")

            if track_id not in handwash_accumulated_time:
                handwash_accumulated_time[track_id] = 0.0
                handwash_completed[track_id] = False
                handwash_wrist_history[track_id] = []

            is_in_person_zone = (current_zone == "handwash")
            is_actively_moving = False
            wrists_in_zone = False

            if matched_pose is not None:
                xy = matched_pose["xy"]
                conf = matched_pose["conf"]
                LWRIST, RWRIST = 9, 10

                current_wrists = []
                for idx in [LWRIST, RWRIST]:
                    if keypoint_valid(xy, conf, idx):
                        current_wrists.append(xy[idx])

                if len(current_wrists) > 0:
                    if handwash_polygon:
                        wrists_in_zone = any(is_point_in_polygon(w, handwash_polygon) for w in current_wrists)
                    else:
                        wrists_in_zone = is_in_person_zone

                    if is_in_person_zone and wrists_in_zone:
                        current_center = np.mean(current_wrists, axis=0)
                        history = handwash_wrist_history[track_id]
                        history.append(current_center)

                        if len(history) > OSCILLATION_WINDOW_FRAMES:
                            history.pop(0)

                        if len(history) >= 5:
                            smoothed_history = []
                            for i in range(len(history)):
                                start_idx = max(0, i - 1)
                                end_idx = min(len(history), i + 2)
                                window_avg = np.mean(history[start_idx:end_idx], axis=0)
                                smoothed_history.append(window_avg)

                            displacements = []
                            sign_changes = 0
                            prev_direction = 0

                            for i in range(1, len(smoothed_history)):
                                diff = smoothed_history[i] - smoothed_history[i - 1]
                                dist = np.linalg.norm(diff)
                                displacements.append(dist)

                                direction = np.sign(diff[0])
                                if prev_direction != 0 and direction != 0 and direction != prev_direction:
                                    sign_changes += 1
                                if direction != 0:
                                    prev_direction = direction

                            avg_movement = np.mean(displacements) if displacements else 0.0

                            if avg_movement >= max(MOVEMENT_SENSITIVITY_THRESHOLD, 2.5) and sign_changes >= MIN_REVERSALS_REQUIRED:
                                is_actively_moving = True

            is_valid_handwash_frame = is_in_person_zone and wrists_in_zone and is_actively_moving

            if is_valid_handwash_frame and not handwash_completed.get(track_id, False):
                # Accumulate true wall-clock time elapsed during this frame
                handwash_accumulated_time[track_id] += frame_dt

                if handwash_accumulated_time[track_id] >= HANDWASH_REQUIRED_SECONDS:
                    handwash_completed[track_id] = True
                    info["handwash_done"] = True

            accumulated = handwash_accumulated_time.get(track_id, 0.0)
            is_done = handwash_completed.get(track_id, False)

            if is_in_person_zone or accumulated > 0.0 or is_done:
                info["handwash_time"] = accumulated
            else:
                info["handwash_time"] = None

            info["handwash_done"] = is_done

            if previous_zone == "handwash" and current_zone != "handwash":
                total_time = accumulated
                if total_time < HANDWASH_REQUIRED_SECONDS and not is_done:
                    active_alerts[track_id] = {
                        "until": frame_idx + ALERT_HOLD_FRAMES,
                        "text": f"Handwash incomplete ({total_time:.1f}s/5s)"
                    }
                    handwash_accumulated_time[track_id] = 0.0
                    handwash_wrist_history[track_id] = []
                    info["handwash_time"] = None

            if current_zone == "production" and not is_done:
                active_alerts[track_id] = {
                    "until": frame_idx + ALERT_HOLD_FRAMES,
                    "text": "Handwash required before production"
                }

            previous_zone_for_handwash[track_id] = current_zone

        for event in events:
            print(f"🚨 CONFIRMED VIOLATION: {event}")
            track_id = event_track_id(event)

            if track_id is not None:
                active_alerts[track_id] = {
                    "until": frame_idx + ALERT_HOLD_FRAMES,
                    "text": event_short_text(event)
                }

        expired = [
            track_id
            for track_id, alert in active_alerts.items()
            if frame_idx > alert["until"]
        ]

        for track_id in expired:
            active_alerts.pop(track_id, None)

        draw_zone_overlay(frame, zones)

        for track_id, info in display_people.items():
            draw_person_panel(
                frame=frame,
                track_id=track_id,
                box=info["box"],
                ppe_list=info["ppe"],
                current_zone=info.get("current_zone"),
                face_touch=info.get("face_touch", False),
                face_distance=info.get("face_distance"),
                handwash_time=info.get("handwash_time"),
                handwash_done=info.get("handwash_done", False),
                alert=active_alerts.get(track_id)
            )

        for track_id, info in display_people.items():
            if info.get("face_distance") is None:
                continue
            px1, py1, _, _ = info["box"]
            raw_text = f"Face d={info['face_distance']:.2f} | {'TOUCH' if info['face_touch'] else 'clear'}"
            raw_color = (0, 0, 255) if info["face_touch"] else (0, 255, 255)
            cv2.putText(
                frame, raw_text,
                (px1, min(FRAME_HEIGHT - 30, py1 + 75)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, raw_color, 1
            )

        t_latency = (time.perf_counter() - t_start) * 1000
        fps = 1000.0 / t_latency if t_latency > 0 else 0.0
        ram_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        zone_mode = "POLYGON" if USE_ZONES else "FIXED FALLBACK"

        cv2.rectangle(frame, (0, 0), (FRAME_WIDTH, 32), (25, 25, 25), -1)
        cv2.putText(
            frame,
            f"Voting {K}/{N} | Zone: {zone_mode} | Face: ENABLED | Handwash: ENABLED",
            (10, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1
        )

        cv2.rectangle(frame, (0, FRAME_HEIGHT - 25), (FRAME_WIDTH, FRAME_HEIGHT), (25, 25, 25), -1)
        cv2.putText(
            frame,
            f"Latency {t_latency:.1f}ms | FPS {fps:.1f} | RAM {ram_mb:.0f}MB",
            (10, FRAME_HEIGHT - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )

        cv2.imshow("LIVE GMP Compliance Monitoring", frame)
        all_payloads.append(frame_payload)

        if hasattr(adapter, "cleanup"):
            adapter.cleanup(frame_idx)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

        frame_idx += 1

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Saving payloads to payloads.json...")
    with open("payloads.json", "w") as f:
        json.dump(all_payloads, f)
    print("Done!")