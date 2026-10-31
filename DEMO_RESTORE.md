# Restore the three-region phone demo

The old `detect_and_track_live_FINAL.py` remains unchanged as a standalone reference.
Use `ai_detector:detect` for phone uploads through the backend. Do not run both workers.

## Render configuration (save and redeploy)

Keep the existing shared secret. Set:

```
PPE_CAMERA_IDS=CAM_01
PPE_ZONE_MODE=three
PPE_CHECK_ACTIONS=false
```

Start with actions disabled to verify PPE. The mode is shown on `/camera`.
`CAM_01` now contains all three regions, not a hard-coded corridor/sink split.
To add a second phone later, change `PPE_CAMERA_IDS` to `CAM_01,CAM_02`.
Both views have independent trackers and their own three regions. This is not
cross-camera re-identification or synchronized capture.

## Laptop commands (inside the project folder)

Terminal 1, using the already working CV environment:

```bash
source .venv/bin/activate
# Copy the Render key privately, then:
export PPE_API_KEY="$(pbpaste)"
python -m cloud_backend.worker --url https://ppe-gmp-compliance.onrender.com --detector ai_detector:detect
```

Open `https://ppe-gmp-compliance.onrender.com/camera` on the phone, refresh it,
select CAM_01 and start a NEW session. Hold the phone landscape and fix it in place.
The three equal-width regions are configured demonstration areas, not automatically
recognized rooms. Person foot-point determines zone. Use a wide enough view to move
between the three regions; a seated close-up spanning the image is unsuitable.

Terminal 2:

```bash
source .venv/bin/activate
python monitor.py --cameras CAM_01 --record demo.mp4
```

Press q in the monitor image window to finalize `demo.mp4`. Check disk space first.
This records the monitor, without audio. macOS Shift-Command-5 can record narration
and the other application windows instead.

No new MediaPipe install is needed: pose uses the already committed YOLO pose weights.
The detector still needs ultralytics, OpenCV, NumPy and httpx in this environment.
If ultralytics requests its tracking dependency, use the same environment to install
that dependency and restart the worker.

## What the display means

- One row each for hairnet, mask and gloves. Scores are retained and shown.
- Green person outline: current model evidence verifies all production PPE.
- Red: current model evidence explicitly detects missing production PPE.
- Yellow: unverified or exempt-zone evidence. No missing detection is invented.
- The BACKEND header is the server's temporally confirmed decision, not the outline.
- MONITORING means a person is in gowning/handwash, where PPE is not required.
  This does not claim verified PPE compliance and does not enable overall green.
- Contradictory present/absent scores within 0.10 are unknown. Detections below
  0.50 are not sent as reliable labels. False confident model predictions can remain.
- Leaving a zone closes its incidents with ZONE_EXIT, not a claim that PPE was restored.
- A stopped source becomes UNKNOWN, even if an older JPEG is still on disk.

## Face touch and handwash

These are visible as pose-based heuristics. To include them in production alerts,
set `PPE_CHECK_ACTIONS=true` on Render and redeploy after PPE works.
Handwash is a hand-motion proxy in the middle region, not proof of washing or hygiene.
Only observed active intervals count. Default target is 20 seconds; gaps over 2.5s
are not counted. Low upload/inference rates may make this proxy unreliable. To use
an accelerated demonstration, set `PPE_HANDWASH_SECONDS=5` in the worker terminal
and label it explicitly as an accelerated demo. It is not a five-second hygiene rule.
Completion is remembered only for this camera + session + track. Reconnects, track
loss or another camera do not preserve it; do not claim cross-camera gowning history.
Gowning-order enforcement from the old standalone script is not restored here.

## Validation before recording

1. Empty view -> NO_PERSON (no green).
2. In production, reliable maskoff -> PENDING then VIOLATION after 3/5 frames.
3. Reliable maskon plus other required items -> recovery after 3 present frames.
4. Obscured items -> UNKNOWN; unknown does not clear an active production incident.
5. Move out of production -> MONITORING; no production-only requirement there.
6. Stop phone -> UNKNOWN after 15 seconds; verify ESP32 independently if connected.

The model weights have not been retrained. Verify gloves with real positive/negative
examples; a bare hand labelled glove is a model error, not successful compliance.
Laptop inference + Render backend is a hybrid demo, not cloud-hosted inference.
The Streamlit dashboard's old local-database path does not read Render events.

## Tests

Run `python -m pytest tests_cloud test_smoothing_basic.py -q` with API test
dependencies plus NumPy/OpenCV. The detector tests use controlled model doubles;
they do not establish model accuracy. `test_zones_basic.py` on the original main
branch imports `point_in_polygon`, which the current `rule_engine/zones.py` does
not export. That pre-existing collection failure is outside these cloud-demo fixes.
