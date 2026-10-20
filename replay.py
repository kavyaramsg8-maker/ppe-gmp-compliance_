
"""
replay.py

GMP PPE Monitoring Replay Pipeline

Features:
- MLAdapter PPE state resolution
- PRESENT / MISSING / NOT_DETECTED
- Temporal PPE holding
- Person tracking state
- Zone detection
- Zone authorization
- Zone transition detection
- Zone PPE compliance
- Gowning-sequence tracking and violation detection
- Face-touch detection
- Temporal smoothing
- Cooldown
- Duplicate prevention
- Snapshots
- SQLite logging
- Dry-run mode
"""

import argparse
import json
import os

import cv2

from ml_adapter import (
    MLAdapter,
    MISSING,
    PRESENT,
    NOT_DETECTED,
)

from rule_engine.rules import (
    check_zone_authorisation,
    update_gowning_sequence,
    check_gowning_sequence,
)

from rule_engine.person_state import PersonState
from rule_engine.zones import (
    load_zones,
    update_person_zone,
)

from rule_engine.smoothing import (
    ViolationManager,
)


# ============================================================
# CONFIGURATION
# ============================================================

GOWNING_ZONE = "gowning_room"

# Example order for testing.
# Replace this with the order approved for your project SOP.
DEFAULT_GOWNING_ORDER = [
    "hairnet",
    "mask",
    "gloves",
]


# ============================================================
# PAYLOAD LOADING
# ============================================================

def load_payloads(path):
    with open(path, "r") as f:
        return json.load(f)


# ============================================================
# VIDEO
# ============================================================

def open_video(path):
    if path is None:
        return None

    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    return cap


def get_video_frame(cap, frame_id):
    if cap is None:
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = cap.read()

    if not ok:
        return None

    return frame


# ============================================================
# DRY RUN MANAGER
# ============================================================

class DryRunManager:
    """
    Prints rule decisions without database or snapshot writes.
    """

    def __init__(
        self,
        k=3,
        n=5,
        cooldown_frames=30,
    ):
        self.k = k
        self.n = n
        self.cooldown_frames = cooldown_frames
        self.events = []

    def process(
        self,
        track_id,
        rule_name,
        condition,
        violation_data,
        frame=None,
    ):
        if not condition:
            return None

        print(
            f"   process("
            f"id={track_id}, "
            f"rule={rule_name}, "
            f"condition={condition}, "
            f"item={violation_data.get('item')}"
            f")"
        )

        return None

    def update(self):
        pass


# ============================================================
# PERSON STATE
# ============================================================

def get_person_state(person_states, track_id):
    if track_id not in person_states:
        person_states[track_id] = PersonState(track_id)

    return person_states[track_id]


# ============================================================
# UPDATE PERSON PPE
# ============================================================

def update_person_ppe(person_state, ppe_state):
    """
    Store the exact ML-derived PPE state.

    PRESENT:
        Detected.

    MISSING:
        Explicitly detected as missing.

    NOT_DETECTED:
        The model did not provide information.
    """

    person_state.current_ppe_state = ppe_state.copy()

    for item, state in ppe_state.items():
        if state == PRESENT:
            person_state.ppe[item] = True

        elif state == MISSING:
            person_state.ppe[item] = False

        # NOT_DETECTED does not modify the legacy boolean.
        # current_ppe_state remains authoritative for rule checks.


# ============================================================
# ZONE PPE
# ============================================================

def check_zone_ppe(person, zone_data):
    """
    Check PPE requirements defined by a zone.

    PRESENT:
        Compliant.

    MISSING:
        Violation.

    NOT_DETECTED:
        Cannot determine, so no violation is generated.
    """

    violations = []

    required_ppe = zone_data.get("required_ppe", [])

    for item in required_ppe:
        state = person.current_ppe_state.get(
            item,
            NOT_DETECTED,
        )

        if state == PRESENT:
            continue

        if state == NOT_DETECTED:
            continue

        if state == MISSING:
            violations.append({
                "track_id": person.track_id,
                "type": "MISSING_PPE",
                "item": item,
                "zone": person.current_zone,
            })

    return violations


# ============================================================
# ZONE TRANSITION
# ============================================================

def process_zone_transition(
    person,
    previous_zone,
    current_zone,
    manager,
    frame,
):
    """
    Trigger a ZONE_ENTRY event when a person moves
    between two known, different zones.
    """

    if (
        previous_zone is None
        or current_zone is None
        or previous_zone == current_zone
    ):
        return []
    if (
        previous_zone == GOWNING_ZONE
        and current_zone != GOWNING_ZONE
        and hasattr(manager, "clear_cooldown")
    ):
        manager.clear_cooldown(
            person.track_id,
            "GOWNING_SEQUENCE",
        )

    violation = {
        "track_id": person.track_id,
        "type": "ZONE_ENTRY",
        "from_zone": previous_zone,
        "to_zone": current_zone,
    }

    event = manager.process(
        track_id=person.track_id,
        rule_name="ZONE_ENTRY",
        condition=True,
        violation_data=violation,
        frame=frame,
    )

    return [event] if event else []


# ============================================================
# GOWNING SEQUENCE
# ============================================================

def process_gowning_sequence(
    person,
    ppe_state,
    previous_zone,
    current_zone,
    required_gowning_order,
    manager,
    frame,
):
    """
    Track positively detected PPE while the person is in
    the gowning room and evaluate the configured sequence.

    The sequence is reset when a person enters the gowning
    room from another zone, so a new visit starts a new sequence.
    """

    # --------------------------------------------------------
    # Reset the sequence at the beginning of a new visit.
    # --------------------------------------------------------

    entering_gowning_room = (
        current_zone == GOWNING_ZONE
        and previous_zone != GOWNING_ZONE
    )

    if entering_gowning_room:
        person.gowning_sequence = []

    # --------------------------------------------------------
    # Record positively detected PPE in the gowning room.
    # --------------------------------------------------------

    if current_zone == GOWNING_ZONE:
        for item, state in ppe_state.items():
            if state == PRESENT:
                update_gowning_sequence(person, item)

        gowning_violations = check_gowning_sequence(
            person,
            required_gowning_order,
        )
    else:
        gowning_violations = []

    # --------------------------------------------------------
    # Send both true and false conditions to the manager.
    # This allows its temporal logic to observe changes.
    # --------------------------------------------------------

    first_violation = (
        gowning_violations[0]
        if gowning_violations
        else {}
    )

    event = manager.process(
        track_id=person.track_id,
        rule_name="GOWNING_SEQUENCE",
        condition=bool(gowning_violations),
        violation_data={
            "track_id": person.track_id,
            "type": "GOWNING_SEQUENCE_VIOLATION",
            "item": first_violation.get("item"),
            "zone": current_zone,
            "confidence": None,
        },
        frame=frame,
    )

    return [event] if event else []


# ============================================================
# FACE TOUCH
# ============================================================

def process_face_touch(
    person,
    observation,
    manager,
    frame,
):
    if not observation.get("face_touch_flag", False):
        return []

    event = manager.process(
        track_id=person.track_id,
        rule_name="FACE_TOUCH",
        condition=True,
        violation_data={
            "track_id": person.track_id,
            "type": "FACE_TOUCH",
            "zone": person.current_zone,
            "item": None,
            "confidence": None,
        },
        frame=frame,
    )

    return [event] if event else []


# ============================================================
# PROCESS ONE OBSERVATION
# ============================================================

def process_observation(
    observation,
    adapter,
    person_states,
    zones,
    manager,
    frame=None,
    fixed_zone=None,
    use_zones=False,
    face_touch=False,
    required_gowning_order=None,
):
    """
    Process one ML payload frame.
    """

    if required_gowning_order is None:
        required_gowning_order = DEFAULT_GOWNING_ORDER

    frame_id = observation.get("frame_id")
    persons = observation.get("persons", [])

    frame_events = []
    display_states = []

    # ========================================================
    # PROCESS EACH PERSON
    # ========================================================

    for raw_person in persons:
        track_id = raw_person.get("track_id")

        if track_id is None:
            continue

        # ----------------------------------------------------
        # Person state
        # ----------------------------------------------------

        person = get_person_state(
            person_states,
            track_id,
        )

        # ----------------------------------------------------
        # ML PPE adapter
        # ----------------------------------------------------

        ppe_state = adapter.update(
            track_id,
            raw_person.get("associated_ppe", []),
            frame_id=frame_id,
        )

        update_person_ppe(
            person,
            ppe_state,
        )

        # ----------------------------------------------------
        # Determine zone
        # ----------------------------------------------------

        previous_zone = person.current_zone

        if use_zones:
            foot_point = raw_person.get("foot_point")

            if foot_point is None:
                continue

            x, y = foot_point

            current_zone = update_person_zone(
                person,
                x,
                y,
                zones,
            )

        else:
            current_zone = fixed_zone

            if current_zone != person.current_zone:
                person.current_zone = current_zone

                if current_zone is not None:
                    person.zone_history.append(current_zone)

        # ----------------------------------------------------
        # Outside all configured zones
        # ----------------------------------------------------

        if use_zones and current_zone is None:
            display_states.append((track_id, ppe_state))

            # Clear the active sequence condition outside
            # the gowning room.
            sequence_events = process_gowning_sequence(
                person=person,
                ppe_state=ppe_state,
                previous_zone=previous_zone,
                current_zone=current_zone,
                required_gowning_order=required_gowning_order,
                manager=manager,
                frame=frame,
            )

            frame_events.extend(sequence_events)
            continue

        # ====================================================
        # ZONE TRANSITION
        # ====================================================

        transition_events = process_zone_transition(
            person=person,
            previous_zone=previous_zone,
            current_zone=current_zone,
            manager=manager,
            frame=frame,
        )

        frame_events.extend(transition_events)

        # ====================================================
        # ZONE AUTHORIZATION
        # ====================================================

        if use_zones:
            authorization_violations = check_zone_authorisation(
                person,
                zones,
            )

            for violation in authorization_violations:
                event = manager.process(
                    track_id=person.track_id,
                    rule_name="UNAUTHORISED_ZONE",
                    condition=True,
                    violation_data={
                        "track_id": person.track_id,
                        "type": "UNAUTHORISED_ZONE",
                        "zone": violation.get("zone"),
                        "item": None,
                        "confidence": None,
                    },
                    frame=frame,
                )

                if event:
                    frame_events.append(event)

        # ====================================================
        # ZONE PPE
        # ====================================================

        if use_zones:
            zone_data = zones.get(current_zone, {})

            ppe_violations = check_zone_ppe(
                person,
                zone_data,
            )

            for violation in ppe_violations:
                item = violation["item"]

                event = manager.process(
                    track_id=person.track_id,
                    rule_name=f"MISSING_PPE_{item}",
                    condition=True,
                    violation_data={
                        "track_id": person.track_id,
                        "type": "MISSING_PPE",
                        "item": item,
                        "zone": current_zone,
                        "confidence": None,
                    },
                    frame=frame,
                )

                if event:
                    frame_events.append(event)

        # ====================================================
        # FIXED ZONE PPE
        # ====================================================

        else:
            if fixed_zone is not None:
                zone_data = zones.get(fixed_zone, {})

                ppe_violations = check_zone_ppe(
                    person,
                    zone_data,
                )

                for violation in ppe_violations:
                    item = violation["item"]

                    event = manager.process(
                        track_id=person.track_id,
                        rule_name=f"MISSING_PPE_{item}",
                        condition=True,
                        violation_data={
                            "track_id": person.track_id,
                            "type": "MISSING_PPE",
                            "item": item,
                            "zone": fixed_zone,
                            "confidence": None,
                        },
                        frame=frame,
                    )

                    if event:
                        frame_events.append(event)

        # ====================================================
        # GOWNING SEQUENCE
        # ====================================================

        sequence_events = process_gowning_sequence(
            person=person,
            ppe_state=ppe_state,
            previous_zone=previous_zone,
            current_zone=current_zone,
            required_gowning_order=required_gowning_order,
            manager=manager,
            frame=frame,
        )

        frame_events.extend(sequence_events)

        # ====================================================
        # FACE TOUCH
        # ====================================================

        if face_touch:
            face_events = process_face_touch(
                person=person,
                observation=raw_person,
                manager=manager,
                frame=frame,
            )

            frame_events.extend(face_events)

        # ====================================================
        # DISPLAY STATE
        # ====================================================

        display_states.append((track_id, ppe_state))

    # ========================================================
    # UPDATE COOLDOWN
    # ========================================================

    manager.update()

    # ========================================================
    # PRINT FRAME STATE
    # ========================================================

    if display_states:
        parts = []

        for track_id, ppe_state in display_states:
            parts.append(f"id{track_id}:{ppe_state}")

        print(f"frame {frame_id}: " + ", ".join(parts))

    else:
        print(f"frame {frame_id}: nobody")

    # ========================================================
    # PRINT EVENTS
    # ========================================================

    for event in frame_events:
        print(f"   EVENT {event}")

    return frame_events


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Replay ML payloads through GMP PPE rule engine."
    )

    # ========================================================
    # INPUT
    # ========================================================

    parser.add_argument(
        "--payloads",
        required=True,
        help="JSON payload file",
    )

    parser.add_argument(
        "--video",
        default=None,
        help="Optional video file",
    )

    # ========================================================
    # ZONES
    # ========================================================

    parser.add_argument(
        "--use-zones",
        action="store_true",
        help="Enable zone detection",
    )

    parser.add_argument(
        "--zones-file",
        default="configs/zones.yaml",
        help="Zone configuration file",
    )

    parser.add_argument(
        "--zone",
        default="production",
        help="Fixed zone when --use-zones is not enabled",
    )

    # ========================================================
    # RULES
    # ========================================================

    parser.add_argument(
        "--face-touch",
        action="store_true",
        help="Enable face-touch rule",
    )

    # ========================================================
    # GOWNING ORDER
    # ========================================================

    parser.add_argument(
        "--gowning-order",
        default="hairnet,mask,gloves",
        help="Comma-separated required gowning order",
    )

    # ========================================================
    # ML HOLD
    # ========================================================

    parser.add_argument(
        "--hold",
        type=int,
        default=5,
        help="Frames to hold previous PPE state",
    )

    # ========================================================
    # SMOOTHING
    # ========================================================

    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="True frames required within smoothing window",
    )

    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Smoothing window size",
    )

    parser.add_argument(
        "--cooldown",
        type=int,
        default=30,
        help="Cooldown frames",
    )

    # ========================================================
    # DRY RUN
    # ========================================================

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print rule decisions without database/snapshot writes",
    )

    args = parser.parse_args()

    # ========================================================
    # VALIDATE GOWNING ORDER
    # ========================================================

    required_gowning_order = [
        item.strip()
        for item in args.gowning_order.split(",")
        if item.strip()
    ]

    if not required_gowning_order:
        parser.error("--gowning-order must contain at least one PPE item")

    # ========================================================
    # LOAD PAYLOADS
    # ========================================================

    if not os.path.exists(args.payloads):
        raise FileNotFoundError(
            f"Payload file not found: {args.payloads}"
        )

    payloads = load_payloads(args.payloads)

    # ========================================================
    # LOAD ZONES
    # ========================================================

    zones = {}

    if args.use_zones:
        zones = load_zones()

        from ml_adapter import ACTIVE_ITEMS

        for zone_name, zone_data in zones.items():
            required = zone_data.get("required_ppe", [])

            not_detectable = [
                item
                for item in required
                if item not in ACTIVE_ITEMS
            ]

            if not_detectable:
                print(
                    f"NOTE: zone '{zone_name}' requires "
                    f"{not_detectable}, which are currently "
                    f"NOT_DETECTED by the ML model. "
                    f"They will not generate false violations."
                )

    # ========================================================
    # ML ADAPTER
    # ========================================================

    adapter = MLAdapter(hold_frames=args.hold)

    # ========================================================
    # PERSON STATES
    # ========================================================

    person_states = {}

    # ========================================================
    # VIDEO
    # ========================================================

    video = open_video(args.video)

    # ========================================================
    # VIOLATION MANAGER
    # ========================================================

    if args.dry_run:
        manager = DryRunManager(
            k=args.k,
            n=args.n,
            cooldown_frames=args.cooldown,
        )

    else:
        manager = ViolationManager(
            k=args.k,
            n=args.n,
            cooldown_frames=args.cooldown,
        )

    # ========================================================
    # REPLAY
    # ========================================================

    total_events = 0

    try:
        for observation in payloads:
            frame_id = observation.get("frame_id", 0)

            # ------------------------------------------------
            # Get video frame
            # ------------------------------------------------

            frame = None

            if video is not None:
                frame = get_video_frame(
                    video,
                    frame_id,
                )

            # ------------------------------------------------
            # Process
            # ------------------------------------------------

            events = process_observation(
                observation=observation,
                adapter=adapter,
                person_states=person_states,
                zones=zones,
                manager=manager,
                frame=frame,
                fixed_zone=args.zone,
                use_zones=args.use_zones,
                face_touch=args.face_touch,
                required_gowning_order=required_gowning_order,
            )

            total_events += len(events)

            # ------------------------------------------------
            # Cleanup stale ML tracks
            # ------------------------------------------------

            adapter.cleanup(frame_id)

    finally:
        if video is not None:
            video.release()

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print(
        f"{len(payloads)} frames, "
        f"{total_events} violation events"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
