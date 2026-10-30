import cv2
import numpy as np

mp = None

import time
import psutil
import json
import os
import yaml

from ultralytics import YOLO

# Import rule engine and adapters
from ml_adapter import MLAdapter
from rule_engine.smoothing import ViolationManager
from rule_engine.person_state import PersonState
from rule_engine.zones import load_zones
from replay import process_observation


# ---------------------------------------------------------
# CONFIGURATION LOADING
# ---------------------------------------------------------

print("Loading configuration files...")


# 1. Load thresholds.yaml
try:
    with open("configs/thresholds.yaml", "r") as f:
        thresholds = yaml.safe_load(f)

    K = thresholds.get("voting_k", 3)
    N = thresholds.get("voting_n", 5)
    TAU = thresholds.get("face_touch_threshold", 0.4)
    CONF_THRESH = thresholds.get("detection_confidence", 0.5)

    print(f"   Thresholds loaded: k={K}, n={N}, tau={TAU}")

except Exception as e:
    print(
        f"   Warning: Could not load thresholds.yaml ({e}). "
        "Using defaults."
    )

    K, N, TAU, CONF_THRESH = 3, 5, 0.4, 0.5


# 2. Load sop.yaml
try:
    with open("configs/sop.yaml", "r") as f:
        sop = yaml.safe_load(f)

    GOWNING_ORDER = sop.get(
        "gowning_order",
        ["hairnet", "mask", "gloves"]
    )

    print(f"   Gowning order loaded: {GOWNING_ORDER}")

except Exception as e:
    print(
        f"   Warning: Could not load sop.yaml ({e}). "
        "Using defaults."
    )

    GOWNING_ORDER = ["hairnet", "mask", "gloves"]


# 3. Load zones
try:
    zones = load_zones("configs/zones_review.yaml")

    print("   Zones loaded successfully.")

    USE_ZONES = True

except Exception as e:
    print(
        f"   Warning: Could not load zones ({e}). "
        "Falling back to fixed zone 'production'."
    )

    zones = {}
    USE_ZONES = False


# 4. Model and Video Paths

MODEL_PATH = "best.pt"

# 0 = laptop webcam
# Later this can become an RTSP CCTV URL
VIDEO_PATH = 0


# ---------------------------------------------------------
# SETUP
# ---------------------------------------------------------

print("Loading YOLO models...")

# Custom PPE model
ppe_model = YOLO(MODEL_PATH)

# Pre-trained COCO model
person_model = YOLO("yolov8n.pt")


print("Face-touch detection disabled for current demo.")

mp_pose = None
pose = None


print("Initializing Rule Engine...")

adapter = MLAdapter(hold_frames=5)

manager = ViolationManager(
    k=K,
    n=N,
    cooldown_frames=30
)

person_states = {}


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def is_contained(item_box, person_box, threshold=0.6):
    """
    Check whether enough of a PPE bounding box lies inside
    the person's bounding box.
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

    if item_area == 0:
        return False

    return (inter_area / item_area) >= threshold


def check_face_touch(person_crop, tau=TAU):
    """
    Face-touch branch retained for later use.
    Currently returns False because pose is disabled.
    """

    if pose is None:
        return False

    if person_crop.size == 0:
        return False

    h, w = person_crop.shape[:2]

    rgb_crop = cv2.cvtColor(
        person_crop,
        cv2.COLOR_BGR2RGB
    )

    results = pose.process(rgb_crop)

    if not results.pose_landmarks:
        return False

    lm = results.pose_landmarks.landmark

    def pt(idx):
        return np.array([
            lm[idx].x * w,
            lm[idx].y * h
        ])

    nose = pt(mp_pose.PoseLandmark.NOSE)

    lwrist = pt(
        mp_pose.PoseLandmark.LEFT_WRIST
    )

    rwrist = pt(
        mp_pose.PoseLandmark.RIGHT_WRIST
    )

    lshoulder = pt(
        mp_pose.PoseLandmark.LEFT_SHOULDER
    )

    rshoulder = pt(
        mp_pose.PoseLandmark.RIGHT_SHOULDER
    )

    shoulder_width = np.linalg.norm(
        lshoulder - rshoulder
    )

    if shoulder_width == 0:
        return False

    ldist = (
        np.linalg.norm(nose - lwrist)
        / shoulder_width
    )

    rdist = (
        np.linalg.norm(nose - rwrist)
        / shoulder_width
    )

    return min(ldist, rdist) < tau


# ---------------------------------------------------------
# MAIN PIPELINE LOOP
# ---------------------------------------------------------

print(
    f"Starting video processing on {VIDEO_PATH}..."
)

all_payloads = []
frame_idx = 0


# Open webcam / video / CCTV source
cap = cv2.VideoCapture(VIDEO_PATH)


if not cap.isOpened():
    print(
        f"Error: Could not open video source "
        f"{VIDEO_PATH}"
    )
    exit()


while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        print("Could not read frame.")
        break
read_failures = 0
MAX_READ_FAILURES = 10

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        read_failures += 1

        print(
            f"Camera frame read failed "
            f"({read_failures}/{MAX_READ_FAILURES})"
        )

        if read_failures >= MAX_READ_FAILURES:
            print("Camera unavailable. Stopping.")
            break

        time.sleep(0.2)
        continue

    # Successful frame → reset failure counter
    read_failures = 0

    t_start = time.perf_counter()


    # -----------------------------------------------------
    # 1. RUN BOTH MODELS
    # -----------------------------------------------------

    # Custom PPE detector
    ppe_results = ppe_model(
        frame,
        conf=CONF_THRESH,
        verbose=False
    )[0]


    # PERSON DETECTION + TRACKING
    #
    # IMPORTANT:
    # classes=[0] means ONLY COCO class 0 = PERSON.
    #
    # So chairs, beds, plants, bottles etc.
    # will NOT receive tracking IDs.
    person_results = person_model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],
        conf=0.4,
        verbose=False
    )[0]


    persons = []
    ppe_items = []


    # -----------------------------------------------------
    # 2. EXTRACT PERSONS
    # -----------------------------------------------------

    if (
        person_results.boxes is not None
        and person_results.boxes.id is not None
    ):

        p_boxes = (
            person_results.boxes.xyxy
            .cpu()
            .numpy()
        )

        p_ids = (
            person_results.boxes.id
            .cpu()
            .numpy()
            .astype(int)
        )

        for box, track_id in zip(
            p_boxes,
            p_ids
        ):

            persons.append({
                "track_id": track_id,
                "box": box
            })


    # -----------------------------------------------------
    # 3. EXTRACT PPE
    # -----------------------------------------------------

    if ppe_results.boxes is not None:

        ppe_boxes = (
            ppe_results.boxes.xyxy
            .cpu()
            .numpy()
        )

        ppe_classes = (
            ppe_results.boxes.cls
            .cpu()
            .numpy()
            .astype(int)
        )


        for box, cls_id in zip(
            ppe_boxes,
            ppe_classes
        ):

            class_name = (
                ppe_model.names[cls_id]
            )

            # IMPORTANT:
            # We collect PPE detections here,
            # but DO NOT draw them yet.
            #
            # They will only be drawn if they
            # belong to an actual detected person.
            ppe_items.append({
                "class_name": class_name,
                "box": box
            })


    frame_payload = {
        "frame_id": frame_idx,
        "persons": []
    }


    # -----------------------------------------------------
    # 4. ASSOCIATE PPE WITH EACH PERSON
    # -----------------------------------------------------

    for person in persons:

        px1, py1, px2, py2 = map(
            int,
            person["box"]
        )


        person_crop = frame[
            max(0, py1):
            min(frame.shape[0], py2),

            max(0, px1):
            min(frame.shape[1], px2)
        ]


        face_touch = check_face_touch(
            person_crop,
            tau=TAU
        )


        associated_ppe = []


        # -------------------------------------------------
        # PPE → PERSON ASSOCIATION
        # -------------------------------------------------

        for item in ppe_items:

            if is_contained(
                item["box"],
                person["box"],
                threshold=0.6
            ):

                associated_ppe.append(
                    item["class_name"]
                )


                # Draw PPE box ONLY if it belongs
                # to a detected person
                ix1, iy1, ix2, iy2 = map(
                    int,
                    item["box"]
                )


                # Light blue PPE box
                cv2.rectangle(
                    frame,
                    (ix1, iy1),
                    (ix2, iy2),
                    (255, 191, 0),
                    2
                )


                cv2.putText(
                    frame,
                    item["class_name"],
                    (ix1, max(20, iy1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 191, 0),
                    2
                )


        # -------------------------------------------------
        # DRAW PERSON BOX
        # -------------------------------------------------

        # Red if face-touch violation
        # Green otherwise
        box_color = (
            (0, 0, 255)
            if face_touch
            else (0, 255, 0)
        )


        cv2.rectangle(
            frame,
            (px1, py1),
            (px2, py2),
            box_color,
            2
        )


        # Make PPE text easier to read
        ppe_text = (
            ",".join(associated_ppe)
            if associated_ppe
            else "none"
        )


        label = (
            f"ID: {person['track_id']} "
            f"| PPE: {ppe_text}"
        )


        cv2.putText(
            frame,
            label,
            (
                px1,
                max(25, py1 - 10)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


        # Face-touch overlay
        if face_touch:

            cv2.rectangle(
                frame,
                (0, 0),
                (frame.shape[1], 40),
                (0, 0, 255),
                -1
            )


            cv2.putText(
                frame,
                (
                    "VIOLATION: "
                    "Face touch detected "
                    f"on ID {person['track_id']}"
                ),
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )


        # -------------------------------------------------
        # PAYLOAD FOR RULE ENGINE
        # -------------------------------------------------

        frame_payload["persons"].append({

            "track_id":
                int(person["track_id"]),

            "box":
                [px1, py1, px2, py2],

            "foot_point":
                (
                    int((px1 + px2) / 2),
                    py2
                ),

            "associated_ppe":
                associated_ppe,

            "face_touch_flag":
                bool(face_touch)
        })


    # -----------------------------------------------------
    # 5. RULE ENGINE
    # -----------------------------------------------------

    events = process_observation(

        observation=frame_payload,

        adapter=adapter,

        person_states=person_states,

        zones=zones,

        manager=manager,

        frame=frame,

        fixed_zone="production",

        use_zones=USE_ZONES,

        face_touch=True,

        required_gowning_order=
            GOWNING_ORDER
    )


    # Print violations
    for event in events:

        print(
            f"🚨 VIOLATION DETECTED: "
            f"{event}"
        )


    # Save frame payload
    all_payloads.append(
        frame_payload
    )


    # -----------------------------------------------------
    # 6. DISPLAY & METRICS
    # -----------------------------------------------------

    t_latency = (
        time.perf_counter()
        - t_start
    ) * 1000


    inference_speed = (
        sum(ppe_results.speed.values())
        +
        sum(person_results.speed.values())
    )


    t_latency += inference_speed


    fps = (
        1000 / t_latency
        if t_latency > 0
        else 0
    )


    ram_mb = (
        psutil.Process()
        .memory_info()
        .rss
        / (1024 * 1024)
    )


    metric_text = (
        f"Latency: {t_latency:.1f}ms "
        f"| FPS: {fps:.1f} "
        f"| RAM: {ram_mb:.0f}MB"
    )


    cv2.putText(
        frame,
        metric_text,
        (
            10,
            frame.shape[0] - 15
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2
    )


    cv2.imshow(
        "GMP Compliance Monitoring Pipeline Demo",
        frame
    )


    # Press q to exit
    if (
        cv2.waitKey(1) & 0xFF
        == ord("q")
    ):
        break


    frame_idx += 1


# ---------------------------------------------------------
# CLEANUP
# ---------------------------------------------------------

cap.release()

cv2.destroyAllWindows()


print(
    "Saving payloads to payloads.json..."
)


with open(
    "payloads.json",
    "w"
) as f:

    json.dump(
        all_payloads,
        f
    )


print("Done!")