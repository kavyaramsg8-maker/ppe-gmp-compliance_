# ESP32 handoff (hardware owner)

Software side only. LED and buzzer behaviour on the device has **not** been verified by the software owner.

- **Backend:** `https://ppe-gmp-compliance.onrender.com` (HTTPS only).
- **Auth:** header `X-API-Key: <team key>`. Get the key privately from the team; do not hard-code it in a
  public repository.
- **Device ID:** `ESP32_01`. Any other ID gets a 404.

## Endpoints (unchanged paths and methods)

| Call | Purpose |
|---|---|
| `GET /v1/devices/ESP32_01/command` | Poll about once per second. Also records the device as online |
| `POST /v1/devices/ESP32_01/ack` body `{"command_id":"<id from command>"}` | Silence the buzzer for the current incidents. 409 = command changed, poll again |
| `GET /v1/devices/ESP32_01` | `{"online": bool, "last_seen": ...}`; online means polled within 15 s |

## Command response (fields the device should use)

```json
{"command_id":"32 hex chars","device_id":"ESP32_01","status":"VIOLATION|COMPLIANT|UNKNOWN",
 "alert":true,"red_led":true,"green_led":false,"amber_led":false,
 "buzzer":true,"buzzer_duration_ms":1000,"valid_for_seconds":15,
 "cameras":[...per-camera detail, may be ignored...],"issued_at":"ISO-8601 UTC"}
```

- Drive the LEDs directly from `red_led`, `green_led` and `amber_led`; exactly one is true.
- Sound the buzzer for `buzzer_duration_ms` when `buzzer` is true, then POST the ack with that
  `command_id`. A **new** incident buzzes again; an acknowledged one does not.
- **Fail safe:** if the backend is unreachable, or nothing has been received for more than
  `valid_for_seconds`, show amber or off, **never green**.

## Compatibility changes in this release

1. The top-level fields are unchanged.
2. `cameras[].status` can now also be `MONITORING` (corridor: no PPE requirement applies) or `PENDING`
   (missing PPE seen but not yet confirmed). Firmware that reads only the top-level LED fields is unaffected.
3. Green now requires every camera to be fresh and not violating, and at least one camera COMPLIANT.
   An empty corridor (`NO_PERSON`) no longer blocks green.
4. The response is larger: about **2 KB with four people in view**, because `cameras[].persons` carries more
   fields. With ArduinoJson, use a buffer of at least 4 KB, or a filter that keeps only the top-level
   fields above.
5. `?observer=true` is for monitoring tools only. The ESP32 must **not** send it, or it will show as offline.

## Hardware test to report back (owner: hardware teammate)

While the software demo runs:
- Mask removed on CAM_02 → red LED and one buzz.
- Ack → buzz stops.
- Mask back on → green, or amber if CAM_01 is unknown.
- Stop a phone → amber within about 15 s.
- Unplug Wi-Fi → amber or off, never green.
- Record the delay from the event appearing on the monitor to the LED changing.
