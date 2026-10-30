import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import os
import time

# =========================================================
# 1. ZONE AND TIMER CONFIGURATION
# =========================================================
USE_CAMERA_ZONES = True 

# Assigns cameras to primary zones
CAMERA_ZONE_MAP = {
    "CAM_01": "corridor",   # Camera 1 will handle Corridor + Sink
    "CAM_02": "production"  # Camera 2 will handle Production
}

# Split the CAM_01 view: Right half (x > 320) is the Sink
SINK_ZONE_X_START = 320  # Assuming 640px width video
HANDWASH_MIN_SECONDS = 20

# Global dictionaries to track state across frames
handwash_timers = {}
wrist_history = {}

# =========================================================
# 2. MODEL SETUP
# =========================================================
# Load your models
ppe_model = YOLO('best.pt')
person_model = YOLO('yolov8n.pt')

# Setup MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

# =========================================================
# 3. HELPER FUNCTIONS
# =========================================================
def check_face_touch(person_crop, tau=0.4):
    if person_crop.size == 0:
        return False
    h, w = person_crop.shape[:2]
    rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_crop)
    if not results.pose_landmarks:
        return False
    lm = results.pose_landmarks.landmark
    
    def pt(idx):
        return np.array([lm[idx].x * w, lm[idx].y * h])
        
    nose = pt(mp_pose.PoseLandmark.NOSE)
    lwrist = pt(mp_pose.PoseLandmark.LEFT_WRIST)
    rwrist = pt(mp_pose.PoseLandmark.RIGHT_WRIST)
    lshoulder = pt(mp_pose.PoseLandmark.LEFT_SHOULDER)
    rshoulder = pt(mp_pose.PoseLandmark.RIGHT_SHOULDER)
    shoulder_width = np.linalg.norm(lshoulder - rshoulder)
    
    if shoulder_width == 0: return False
    
    ldist = np.linalg.norm(nose - lwrist) / shoulder_width
    rdist = np.linalg.norm(nose - rwrist) / shoulder_width
    return min(ldist, rdist) < tau

def check_handwash_motion(person_crop, track_id):
    """
    Detects if hands are close together AND moving (rubbing motion).
    """
    if person_crop.size == 0:
        return False
    h, w = person_crop.shape[:2]
    rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_crop)
    
    if not results.pose_landmarks:
        return False
        
    lm = results.pose_landmarks.landmark
    
    def pt(idx):
        return np.array([lm[idx].x * w, lm[idx].y * h])
        
    # Ensure wrists are visible (visibility > 0.5)
    if lm[mp_pose.PoseLandmark.LEFT_WRIST].visibility < 0.5 or \
       lm[mp_pose.PoseLandmark.RIGHT_WRIST].visibility < 0.5:
        return False

    lwrist = pt(mp_pose.PoseLandmark.LEFT_WRIST)
    rwrist = pt(mp_pose.PoseLandmark.RIGHT_WRIST)
    
    # 1. Are hands close together? (Rubbing distance)
    hand_distance = np.linalg.norm(lwrist - rwrist)
    
    # 2. Are hands moving? (Compare to previous frame)
    current_pos = (lwrist + rwrist) / 2
    movement = 0
    if track_id in wrist_history:
        prev_pos = wrist_history[track_id]
        movement = np.linalg.norm(current_pos - prev_pos)
        
    wrist_history[track_id] = current_pos
    
    # Thresholds (Adjust based on resolution)
    HANDS_CLOSE_THRESHOLD = 120 # Pixels
    MOVEMENT_THRESHOLD = 3      # Pixels per frame
    
    # Return True if hands are close and moving
    return hand_distance < HANDS_CLOSE_THRESHOLD and movement > MOVEMENT_THRESHOLD

def is_contained(item_box, person_box, threshold=0.5):
    ix1 = max(item_box[0], person_box[0])
    iy1 = max(item_box[1], person_box[1])
    ix2 = min(item_box[2], person_box[2])
    iy2 = min(item_box[3], person_box[3])
    item_area = (item_box[2] - item_box[0]) * (item_box[3] - item_box[1])
    inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if item_area == 0: return False
    return (inter_area / item_area) >= threshold

# =========================================================
# 4. MAIN DETECTION FUNCTION
# =========================================================
def detect(frame_content, camera_id, session_id):
    # Decode image
    nparr = np.frombuffer(frame_content, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return []

    # Run inference
    ppe_results = ppe_model(frame, conf=0.15, verbose=False)[0]
    person_results = person_model.track(frame, persist=True, tracker='bytetrack.yaml', verbose=False)[0]

    persons = []
    ppe_items = []

    if person_results.boxes is not None and person_results.boxes.id is not None:
        p_boxes = person_results.boxes.xyxy.cpu().numpy()
        p_ids = person_results.boxes.id.cpu().numpy().astype(int)
        for box, track_id in zip(p_boxes, p_ids):
            persons.append({'track_id': track_id, 'box': box})

    if ppe_results.boxes is not None:
        ppe_boxes = ppe_results.boxes.xyxy.cpu().numpy()
        ppe_classes = ppe_results.boxes.cls.cpu().numpy().astype(int)
        for box, cls_id in zip(ppe_boxes, ppe_classes):
            class_name = ppe_model.names[cls_id]
            ppe_items.append({'class_name': class_name, 'box': box})
            
            # Draw Blue PPE Box
            ix1, iy1, ix2, iy2 = map(int, box)
            cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), (255, 191, 0), 2)
            cv2.putText(frame, class_name, (ix1, iy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 191, 0), 2)

    output_persons = []
    current_time = time.time()

    # Draw the Sink Zone visual on CAM_01
    if camera_id == "CAM_01":
        cv2.rectangle(frame, (SINK_ZONE_X_START, 0), (frame.shape[1], frame.shape[0]), (255, 0, 0), 2)
        cv2.putText(frame, "HANDWASH SINK", (SINK_ZONE_X_START + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    for person in persons:
        track_id = int(person['track_id'])
        px1, py1, px2, py2 = map(int, person['box'])
        person_crop = frame[max(0, py1): min(frame.shape[0], py2), max(0, px1): min(frame.shape[1], px2)]
        
        face_touch = check_face_touch(person_crop)
        associated_ppe = []
        
        for item in ppe_items:
            if is_contained(item['box'], person['box']):
                associated_ppe.append(item['class_name'])
        
        # Remove duplicates
        associated_ppe = list(set(associated_ppe))

        # Base zone
        base_zone = CAMERA_ZONE_MAP.get(camera_id, "production") if USE_CAMERA_ZONES else "production"
        
        foot_x = int((px1 + px2) / 2)
        foot_y = py2
        
        # =========================================================
        # ZONE & HANDWASH STATE MACHINE
        # =========================================================
        current_zone = base_zone
        handwash_status = "not_in_sink"

        if camera_id == "CAM_01":
            if foot_x > SINK_ZONE_X_START:
                current_zone = "sink_zone"
                
                # Check if action is happening
                is_washing = check_handwash_motion(person_crop, track_id)
                
                if is_washing:
                    if track_id not in handwash_timers:
                        handwash_timers[track_id] = current_time
                        handwash_status = "started"
                    else:
                        elapsed = current_time - handwash_timers[track_id]
                        if elapsed >= HANDWASH_MIN_SECONDS:
                            handwash_status = "compliant"
                        else:
                            handwash_status = "washing"
                else:
                    # In zone but not moving hands
                    if track_id in handwash_timers:
                        handwash_status = "paused"
                    else:
                        handwash_status = "waiting_for_motion"
            else:
                # Left the sink zone
                if track_id in handwash_timers:
                    elapsed = current_time - handwash_timers[track_id]
                    if elapsed < HANDWASH_MIN_SECONDS:
                        handwash_status = "FAILED_HANDWASH"
                        print(f"🚨 VIOLATION: Person {track_id} left sink after only {elapsed:.1f}s of washing!")
                    del handwash_timers[track_id]

        # =========================================================
        # DRAWING AND OUTPUT
        # =========================================================
        # If in Corridor, remove "no_" flags so it stays compliant
        if current_zone == "corridor":
            associated_ppe = [item for item in associated_ppe if not item.startswith("no_")]

        box_color = (0, 0, 255) if face_touch else (0, 255, 0)
        cv2.rectangle(frame, (px1, py1), (px2, py2), box_color, 2)
        
        label = f"ID: {track_id} | {current_zone.upper()}"
        
        # Update label based on handwash state
        if handwash_status in ["washing", "started", "paused"]:
            elapsed = current_time - handwash_timers.get(track_id, current_time)
            label += f" | Wash: {elapsed:.1f}s ({handwash_status})"
        elif handwash_status == "compliant":
            label += " | Wash: DONE"
        elif handwash_status == "FAILED_HANDWASH":
            label += " | WASH FAILED!"
            
        cv2.putText(frame, label, (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        output_persons.append({
            "track_id": track_id,
            "box": [px1, py1, px2, py2],
            "foot_point": [foot_x, foot_y], 
            "associated_ppe": associated_ppe,
            "face_touch": bool(face_touch),
            "handwash_status": handwash_status
        })

    # Clean up old timers/wrist data for people who left the frame completely
    active_ids = [p['track_id'] for p in persons]
    for tid in list(handwash_timers.keys()):
        if tid not in active_ids:
            del handwash_timers[tid]
    for tid in list(wrist_history.keys()):
        if tid not in active_ids:
            del wrist_history[tid]

    # Save annotated frame for the monitor
    cv2.imwrite(f"live_{camera_id}.jpg", frame)
    
    if len(output_persons) > 0:
        print(f"DEBUG: {camera_id} ({current_zone}) | PPE: {output_persons[0]['associated_ppe']} | Wash: {output_persons[0]['handwash_status']}")
        
    return output_persons