# Team integration contract

Flow: phones upload JPEGs directly to this cloud API; Aravind's worker runs on
cloud compute and submits per-person observations; one ESP32 polls conclusions.
No cross-camera identity matching. All team clients use the shared X-API-Key.
This is a small-team demo trust boundary, not separate production roles.

## Ujju — camera upload
Open the cloud HTTPS address `/camera` on each phone. Choose different IDs and
enter the key privately. Press Start and allow camera access. This sends one
640×360 JPEG per second, directly from the phone. Keep the page foregrounded.
Phone clocks must be accurate; captures older than 15 seconds are rejected.
The browser key is not stored. The server uses a single latest frame per camera.
The configured IDs come from PPE_CAMERA_IDS, default CAM_01,CAM_02.

For a custom sender:
1. POST `/v1/cameras/CAM_01/sessions` → session_id.
2. POST multipart `/v1/cameras/CAM_01/frames`: session_id, integer frame_id
   starting at 0 and increasing, timezone-aware captured_at, image (JPEG).
3. On 429 drop this sample and try again later: detector has a processing lease.
   Do not queue old frames. On 409 obtain a new session (one sender per ID).
4. GET `/v1/cameras/CAM_01` to see its latest decision.

## Aravind — actual cloud detection
Do not import detect_and_track_live_FINAL.py into the API: it opens the webcam,
loads models and runs its main loop during import. Supply a headless callable:

```python
# aravind_inference.py (your implementation, not included as a fake detector)
def infer(jpeg_bytes: bytes, camera_id: str, session_id: str) -> list[dict]:
    # Decode JPEG, run models, associate PPE with tracked persons.
    # Keep tracker/model state separate per (camera_id, session_id).
    # Return actual inference output in this structure:
    return [
        {"track_id": 1, "associated_ppe": [
            {"class": "hairnet", "confidence": 0.92},
            {"class": "maskon", "confidence": 0.88},
            {"class": "no_glove", "confidence": 0.81},
        ]}
    ]
```
The return above only illustrates the schema. Never return those constant labels
as detection. The cloud worker calls the real function with each fresh image.
Use stable per-camera track IDs. A session ID changes on camera reconnect.
A low-confidence or obscured item should produce no reliable class: this becomes
not_detected/UNKNOWN, not a missing-PPE accusation. Never invent a person ID from
an empty image. Return [] only when inference succeeded and found no people;
raise an exception on model errors so the backend expires to UNKNOWN.

Existing MLAdapter class names are reused: hairnet/no_hairnet, maskon/maskoff,
glove/no_glove. Other names are ignored; align model classes before integration.
Dictionary detections below 0.5 are ignored. Existing string class lists are also
accepted, with confidence filtering left to the trusted detector.

Run on the same cloud host (or locally while developing):

```bash
export PPE_DETECTOR=aravind_inference:infer
python -m cloud_backend.worker --url http://127.0.0.1:8000
```

The worker GETs `/v1/cameras`, fetches `/v1/cameras/{id}/frame` with
`X-Claim-Frame: true`, reads X-Frame-ID and X-Session-ID response headers, then
POSTs `/v1/cameras/{id}/observations`. The claim blocks new uploads for up to
10 seconds while inference runs. Only a result matching the latest session/frame
is accepted. It must also arrive within the 15-second freshness limit.
Measure model latency and adjust sampling/compute before changing this limit.

The launcher `python -m cloud_backend.serve` starts the API and, if PPE_DETECTOR
is set, the worker on the same host. Dockerfile.cloud currently builds the API
only. To run the detector in it, Aravind must add his module, model weights and
CV dependencies to that image. Cloud inference is not claimed ready until this
real-model path is exercised. No fake detector is enabled by default.

## Compliance policy used by this API
- Fixed production-camera requirement: hairnet, mask, gloves.
- Three explicit missing observations within five processed frames confirm each
  per-person/per-item violation. Decisions are scoped to camera + session + track.
- One event per confirmed incident; continuous absence does not spam events.
- Three consecutive explicit present observations clear an active incident.
- Unknown observations do not clear a confirmed incident; stale camera decisions
  become UNKNOWN after 15 seconds and must not become green.
- No detected person → NO_PERSON (not verified compliance).
- This network MVP covers PPE only. Polygon zones, face touch, gowning order and
  handwashing logic in the original script are NOT automatically ported here.
  Aravind owns extending the cloud detector/rules if those are required.

## ESP32 — agreed response
Poll GET `/v1/devices/ESP32_01/command` about once per second. Example fields:

```json
{"command_id":"unique-state-id","status":"VIOLATION","alert":true,
 "red_led":true,"green_led":false,"amber_led":false,"buzzer":true,
 "buzzer_duration_ms":1000,"valid_for_seconds":15}
```

A confirmed violation on either configured camera keeps red on. Green is allowed
only when ALL configured cameras have fresh COMPLIANT results. Start with
PPE_CAMERA_IDS=CAM_01 until the second camera is connected. If either is unknown,
no overall green. `amber_led` is a logical status; if you only have two LEDs,
use a distinct non-green blink pattern for unknown.

When buzzer=true, sound one bounded pulse (default 1000 ms), then POST
`/v1/devices/ESP32_01/ack` with `{"command_id":"the-id-received"}`. Remember handled
command IDs locally too, so a lost acknowledgement response doesn't repeat a
pulse on every poll. Backend acknowledgements track incident IDs, so another
camera changing state does not re-trigger the same acknowledged incident.
If ack returns 409 poll again; don't acknowledge a different command blindly.

On HTTP/Wi-Fi failure, ESP32 firmware must use its OWN watchdog: after
valid_for_seconds without a valid response, turn green off and show unknown.
The cloud cannot send a disconnect warning to hardware that is offline.
Use HTTPS with server certificate validation; never put the key in a URL.
No physical firmware/wiring is included in Kavya's backend task.

## Sana — verification
Exercise real present, explicit missing, obscured/unclear, no-person, restored
PPE and two-camera cases. Repeat the same track_id on the second camera to check
isolation. Disconnect each camera/worker/Wi-Fi/ESP32. Events and snapshots are
retrievable with GET `/v1/events` and its snapshot_url fields, all authenticated.
GET `/v1/devices/ESP32_01` reports recent polling activity, not hardware proof.
Measure full capture-to-alert latency including sampling, voting, network,
inference and device polling; no real latency/accuracy is claimed by API tests.
