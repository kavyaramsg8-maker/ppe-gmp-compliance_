
PRESENT = "present"
MISSING = "missing"
NOT_DETECTED = "not_detected"

CLASS_MAP = {
    "glove": ("gloves", PRESENT),
    "no_glove": ("gloves", MISSING),
    "hairnet": ("hairnet", PRESENT),
    "no_hairnet": ("hairnet", MISSING),
    "maskon": ("mask", PRESENT),
    "maskoff": ("mask", MISSING),
}

PPE_ITEMS = (
    "hairnet",
    "mask",
    "goggles",
    "gown",
    "gloves",
    "shoe_cover",
)

ACTIVE_ITEMS = tuple(
    sorted({item for item, state in CLASS_MAP.values()})
)


class MLAdapter:
    def __init__(self, hold_frames=5, stale_after=300):
        self.hold_frames = hold_frames
        self.stale_after = stale_after
        self.last_state = {}
        self.missing_frames = {}
        self.last_seen = {}
        self.unmapped_classes = set()

    def update(self, track_id, detections, frame_id=0):
        if track_id not in self.last_state:
            self.last_state[track_id] = {
                item: NOT_DETECTED for item in PPE_ITEMS
            }

        if track_id not in self.missing_frames:
            self.missing_frames[track_id] = {
                item: 0 for item in PPE_ITEMS
            }

        self.last_seen[track_id] = frame_id
        current_detected = {}

        # Read either string detections or dictionary detections.
        for detection in detections:
            if isinstance(detection, str):
                class_name = detection
            elif isinstance(detection, dict):
                class_name = detection.get("class")
            else:
                continue

            if class_name not in CLASS_MAP:
                if class_name is not None:
                    self.unmapped_classes.add(class_name)
                continue

            item, state = CLASS_MAP[class_name]

            # MISSING takes priority if both states are detected.
            if item not in current_detected:
                current_detected[item] = state
            elif state == MISSING:
                current_detected[item] = MISSING

        final_state = {}

        for item in PPE_ITEMS:
            if item in current_detected:
                # An explicit detection resets this item's counter.
                final_state[item] = current_detected[item]
                self.missing_frames[track_id][item] = 0
            else:
                # Count missing detections independently per PPE item.
                self.missing_frames[track_id][item] += 1

                previous_state = self.last_state[track_id][item]
                missing_count = self.missing_frames[track_id][item]

                if (
                    previous_state != NOT_DETECTED
                    and missing_count <= self.hold_frames
                ):
                    final_state[item] = previous_state
                else:
                    final_state[item] = NOT_DETECTED

        self.last_state[track_id] = final_state.copy()
        return final_state

    def cleanup(self, current_frame):
        stale_tracks = [
            track_id
            for track_id, last_seen_frame in self.last_seen.items()
            if current_frame - last_seen_frame > self.stale_after
        ]

        for track_id in stale_tracks:
            self.last_seen.pop(track_id, None)
            self.last_state.pop(track_id, None)
            self.missing_frames.pop(track_id, None)
