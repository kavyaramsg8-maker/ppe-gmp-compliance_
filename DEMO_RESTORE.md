# Correct two-camera layout

This supersedes the previous equal-third layout. The earlier suggestion to divide
a single image into gowning / handwash / production was wrong for this setup.

- CAM_01: corridor / gowning camera, with a handwash region inside the same view.
- CAM_02: separate production camera. Its entire image is production.
- There is no production region on CAM_01 and no corridor/handwash region on CAM_02.
- Two physical phones are needed for simultaneous two-camera capture.

## Apply on laptop and deploy

Stop both worker and monitor, and stop phone uploads. Back up tracked local changes
using `git stash push -m "Before two-camera correction"`, then extract this ZIP into
the project root, keeping its folder structure. No models, keys or video data are
included in the ZIP. It contains replacement source files and regression tests.

Push only the changed source files, not `.venv`, secrets, live frames or recordings.
Deploy the new backend code before running this detector.

Render Environment:

```
PPE_CAMERA_IDS=CAM_01,CAM_02
PPE_ZONE_MODE=cameras
PPE_CHECK_ACTIONS=false
```

Keep the existing API key privately. Save, deploy and wait for Live.
Verify `/camera-config` reports both IDs and `zone_mode: cameras`.

Terminal 1 (project folder):

```bash
source .venv/bin/activate
# If needed, copy the same Render API key privately, then:
export PPE_API_KEY="$(pbpaste)"
python -m cloud_backend.worker --url https://ppe-gmp-compliance.onrender.com --detector ai_detector:detect
```

The worker reads the configured handwash region from the backend at startup.
It refuses to start this detector against an incompatible old layout.

On phone 1, refresh `/camera`, choose CAM_01 and Start. On phone 2, choose CAM_02
and Start. Keep phone cameras fixed, landscape and foregrounded.

Terminal 2:

```bash
source .venv/bin/activate
python monitor.py --cameras CAM_01,CAM_02 --record demo.mp4
```

The monitor displays separate camera views stacked vertically. Press q in its image
window to finalize demo.mp4 (no audio). Without `--record` it only displays video.

## Handwash region

Default: the right half of CAM_01, preserving Ujwal's original arrangement. The
outline is just a configured region; it must cover the actual sink area in the
fixed camera view. Set `PPE_HANDWASH_ROI=[left,top,right,bottom]` on Render to adjust
it, with normalized coordinates between 0 and 1. Example `[0.5,0,1,1]` is the right
half. Redeploy and restart the worker after changing it. Person foot-point assigns
the zone, and both visible wrists must be in the wash region for motion evidence.
There is no automatic sink recognition.

## Decisions and colours

One scored row per PPE item. Missing evidence is yellow/unknown, not automatically
missing PPE. Close contradictory predictions are unknown. Green/red person outlines
represent current PPE model evidence in production. The BACKEND header is the
server's temporally confirmed verdict. CAM_01 ordinarily shows MONITORING, which
means no production PPE requirement applies there, not verified PPE compliance.

A violation on either camera makes the ESP32 command red. Overall green needs
fresh verified production compliance on CAM_02 and a fresh non-violating monitored
CAM_01. Disconnected/unknown cameras never enable green.

## Optional handwash and face-touch alerts

First verify PPE with PPE_CHECK_ACTIONS=false. To exercise action heuristics, set it
true on Render and redeploy. CAM_01 can report an incomplete wash after an observed
partial washing attempt followed by leaving the handwash region. CAM_02 evaluates
PPE and face touch. It does NOT assume CAM_02's track 1 is CAM_01's track 1 or demand
that CAM_02 prove completion from another camera.

Handwashing is a pose/motion proxy, not proof of hygiene. The default target is 20s
of observed active intervals. Missing observations and gaps over 2.5s do not count.
This proxy needs adequate sampling and visibility. No cross-camera handwash sequence
or gowning order is verified. Track loss/reconnect resets camera-local action history.

## Remaining model issue

The model names match the adapter, but this does not establish correct training labels
or accurate predictions. The hairnet/glove confusion still needs raw-image verification
and may require correcting the dataset and retraining. No class names were swapped to
hide it, and no model weights were changed. Keep a real positive/negative PPE test set.

## Validation

22 targeted tests passed in the build environment, using controlled detections for
model-dependent code. They verify layout, backend rules, camera/session isolation,
confidence handling and stale results; they do not verify live detection accuracy.
The full original suite still cannot collect test_zones_basic.py because its existing
rule_engine.zones module does not export point_in_polygon. That legacy issue is separate.
Run `python -m pytest tests_cloud test_smoothing_basic.py -q` with API-test dependencies,
NumPy and OpenCV installed. This patch adds no MediaPipe dependency.

Inference still runs on the laptop while Render hosts the API. This remains a hybrid
demo until model inference is deployed to cloud compute. Physical ESP32 alerts need
independent wiring/firmware testing. The old Streamlit local-DB dashboard does not
read the remote Render database automatically.
