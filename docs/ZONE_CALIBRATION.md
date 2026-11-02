# Zone calibration

Layout: **two physical cameras, three logical areas.**

| Camera | View | Zones |
|---|---|---|
| CAM_01 | corridor / gowning | `corridor` everywhere, except the `handwash` polygon around the sink |
| CAM_02 | production | `production` everywhere (no regions) |

Neither image is split into equal columns, and nothing is copied between cameras. Track IDs, handwash history and
PPE votes belong to one camera session only, because cross-camera person matching is out of scope. A wash
seen on CAM_01 never counts as production compliance on CAM_02.

## Where the geometry lives

`cloud_backend/config/camera_zones.yaml` is the single source of truth. The backend uses it for verdicts and
serves it at `GET /camera-config`. The Mac worker downloads it at start-up, and the detector uses it for
drawing and sink-motion evidence.

- Coordinates are **normalised to the camera picture**: `[x, y]` with `0,0` top-left and `1,1` bottom-right.
  They are independent of upload resolution.
- The phone page sends the native aspect ratio with **no letterbox bars**. If a custom sender letterboxes,
  it must send `content_box=left,top,right,bottom` (pixels); points on the bars are then `outside` and never
  silently mapped into a zone.
- A person's zone is the **bottom-centre of their box** (foot point). Regions are tested in order; the first
  match wins; otherwise the camera's `default_zone` applies.
- A zone change must persist for 2 consecutive processed frames (hysteresis), so foot-point jitter on a
  boundary does not reset votes or close incidents.
- Required PPE and action checks are per zone (`required_ppe`, `action_checks`). The model supports only
  `hairnet`, `mask` and `gloves`; the config refuses any other item.

## Calibrating against the real views (10 minutes per camera)

1. Mount each phone in its **final position and orientation**. Rotating a phone later changes the picture,
   so the polygon must be redrawn.
2. Start both phones and the worker. The detector writes the processed still to `live_CAM_01.jpg`. You can
   also use any photo taken by that phone in that position, in the same orientation.
3. Measure the sink polygon:
   ```bash
   source .venv/bin/activate
   cp live_CAM_01.jpg calib_CAM_01.jpg     # freeze one frame
   python calibrate_zones.py --camera CAM_01 --image calib_CAM_01.jpg
   ```
   Click the corners of the floor area where a person stands while washing. That is where their feet are,
   not the basin itself. Copy the printed `polygon:` line into `camera_zones.yaml` under
   `CAM_01 → regions → handwash`.
4. Check the result without a window:
   `python calibrate_zones.py --camera CAM_01 --image calib_CAM_01.jpg --save-preview zones_CAM_01.png`
5. Walk the boundary. In the monitor, the label above the person must change between `CORRIDOR` and
   `HANDWASH` where you expect. The wrists must also be inside the polygon for wash motion to count, so the
   polygon must cover the hands' position above the basin as well as the standing area.
6. Commit the YAML, redeploy the backend, and restart the worker; it loads the layout at start-up.

`PPE_HANDWASH_ROI=[left,top,right,bottom]` on Render still works for the old rectangle setting. It replaces
the handwash polygon with that rectangle. Delete that variable to use the YAML polygon.

## Policy items flagged as unclear (not invented)

- **PPE at the sink:** `handwash.required_ppe` is empty, following the earlier `configs/zones.yaml`.
  Confirm the site SOP.
- **Face touch and handwash alerts** are evaluated only with `PPE_CHECK_ACTIONS=true`. The deployed backend
  had them off. With them off, the values are still reported per person in the API and monitor, but they
  raise no alert.
- **Handwash requirement for production entry** cannot be enforced: identities are not matched across cameras.
