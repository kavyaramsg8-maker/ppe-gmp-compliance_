
"""
GMP PPE Monitoring System
Presentation Demo Entry Point

Demonstrates:
1. PPE compliance monitoring
2. Missing PPE detection
3. Temporal smoothing to reduce false alerts
4. Violation logging in SQLite
5. Snapshot capture
6. Alert cooldown and recovery

Run from the project root:
    python main.py
"""

import os
import cv2
import numpy as np

from rule_engine.smoothing import ViolationManager


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

K = 3
N = 5
COOLDOWN_FRAMES = 30

TRACK_ID = 17
ZONE = "production"


# ---------------------------------------------------------
# DISPLAY HELPERS
# ---------------------------------------------------------

def create_demo_frame(frame_number, ppe_status, status_text):
    """Create a clearly labelled presentation frame."""

    frame = np.zeros((520, 900, 3), dtype=np.uint8)
    frame[:] = (35, 30, 25)

    cv2.rectangle(frame, (0, 0), (900, 75), (95, 45, 145), -1)

    cv2.putText(
        frame,
        "GMP PPE MONITORING SYSTEM",
        (25, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        "PRESENTATION DEMO - SIMULATED INPUT",
        (25, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (190, 190, 190),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"Person ID: {TRACK_ID}",
        (35, 165),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        f"Zone: {ZONE.upper()}",
        (35, 205),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (220, 220, 220),
        2,
    )

    y = 265

    for item, status in ppe_status.items():
        compliant = status == "PRESENT"
        color = (70, 210, 100) if compliant else (60, 70, 245)

        cv2.circle(frame, (55, y - 8), 9, color, -1)

        cv2.putText(
            frame,
            f"{item}: {status}",
            (80, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

        y += 48

    alert_color = (
        (70, 210, 100)
        if status_text == "COMPLIANT"
        else (60, 70, 245)
    )

    cv2.rectangle(frame, (25, 430), (875, 485), alert_color, 2)

    cv2.putText(
        frame,
        status_text,
        (45, 466),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        alert_color,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"Demo frame: {frame_number}",
        (700, 510),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (180, 180, 180),
        1,
    )

    return frame


# ---------------------------------------------------------
# DEMO SCENARIO
# ---------------------------------------------------------

def main():
    print("=" * 60)
    print("       GMP PPE MONITORING SYSTEM")
    print("       Presentation Demonstration")
    print("=" * 60)

    print("\nNOTE: This demonstration uses simulated PPE states.")
    print("It does not perform live AI detection.\n")

    manager = ViolationManager(
        k=K,
        n=N,
        cooldown_frames=COOLDOWN_FRAMES,
    )

    scenarios = [
        {
            "name": "Correct PPE",
            "status": {
                "hairnet": "PRESENT",
                "mask": "PRESENT",
                "gloves": "PRESENT",
            },
            "missing_item": None,
            "frames": 8,
        },
        {
            "name": "Missing gloves",
            "status": {
                "hairnet": "PRESENT",
                "mask": "PRESENT",
                "gloves": "MISSING",
            },
            "missing_item": "gloves",
            "frames": 8,
        },
        {
            "name": "PPE restored",
            "status": {
                "hairnet": "PRESENT",
                "mask": "PRESENT",
                "gloves": "PRESENT",
            },
            "missing_item": None,
            "frames": 8,
        },
    ]

    frame_number = 0
    total_alerts = 0

    try:
        for scenario in scenarios:
            print(f"\n--- Scenario: {scenario['name']} ---")

            for _ in range(scenario["frames"]):
                frame_number += 1

                status = scenario["status"]
                missing_item = scenario["missing_item"]

                is_missing = missing_item is not None

                display_status = (
                    f"ALERT: MISSING {missing_item.upper()}"
                    if is_missing
                    else "COMPLIANT"
                )

                frame = create_demo_frame(
                    frame_number,
                    status,
                    display_status,
                )

                violation_data = {
                    "track_id": TRACK_ID,
                    "type": "MISSING_PPE",
                    "item": missing_item,
                    "zone": ZONE,
                    "confidence": None,
                }

                event = manager.process(
                    track_id=TRACK_ID,
                    rule_name="MISSING_PPE_gloves",
                    condition=is_missing,
                    violation_data=violation_data,
                    frame=frame,
                )

                if event:
                    total_alerts += 1
                    print(
                        f"[ALERT] Person #{TRACK_ID} | "
                        f"Zone: {ZONE} | "
                        f"Missing: {missing_item} | "
                        f"Event ID: {event.get('event_id')} | "
                        f"Snapshot: {event.get('snapshot_path')}"
                    )

                manager.update()

                cv2.imshow("GMP PPE Monitoring System", frame)

                # Press Q or ESC to end the demo early.
                key = cv2.waitKey(250) & 0xFF

                if key in (ord("q"), 27):
                    print("\nDemo stopped by user.")
                    return

            print(f"Completed: {scenario['name']}")

        print("\n" + "=" * 60)
        print("DEMONSTRATION SUMMARY")
        print("=" * 60)
        print(f"Frames demonstrated : {frame_number}")
        print(f"New violation alerts: {total_alerts}")
        print("Database            : gmp_monitoring.db")
        print("Snapshots           : snapshots/")
        print("=" * 60)
        print("\nDemo completed successfully.")

        cv2.waitKey(0)

    except KeyboardInterrupt:
        print("\nDemo interrupted.")

    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
