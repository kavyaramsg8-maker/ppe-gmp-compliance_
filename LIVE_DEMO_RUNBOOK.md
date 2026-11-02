# Live two-camera demo: runbook (branch `fix/live-pipeline-zones`)

Phones → Render backend → Mac worker → Render backend → ESP32 is unchanged. This branch makes that flow
fresh, steady and zone-correct. Supersedes the zone and handwash notes in `PERFORMANCE_START_HERE.md`.

## 1. What was wrong (measured)

Your recorded live run (`performance_run/live.jsonl`, before this branch): **0.60 / 0.63 processed FPS**,
result interval p50 ≈1.5 s, p95 ≈2.3 s, CV 0.33; detector only **≈70–77 ms** of an **≈800 ms** cycle.

1. **Render runs `main`, not `demo-setup`.** The deployed phone page is capped at **1 upload/s** and waits
   for a status GET after every upload. The 3/5/8 FPS selector was never deployed.
2. **The backend serialised everything.** It returned 429 to every upload while a frame was claimed, so a
   new frame could enter only after the previous result came back.
3. **The worker was lock-stepped.** It listed cameras, then downloaded, inferred and submitted, then waited
   for *both* cameras. Each request costs **≈270 ms TTFB to Render** (measured; TLS to the edge is ≈30 ms,
   so the origin is far away).
4. **Phones in portrait were letterboxed** into 640×360. The picture used ≈200 px of width, and the `x>320`
   sink rule covered half a black bar.

## 2. Fixes

| Area | Change | Files |
|---|---|---|
| Freshness | Latest-frame slot. Uploads are never blocked (no 429); a newer upload replaces an unclaimed one and is counted. Long-poll claim `GET …/frame?after=&session=&wait=` (204 if nothing new). Concurrent claims get distinct frames. A result is accepted for any claimed frame newer than the last decision and <15 s old; older or superseded results get 409. Snapshots are the **claimed** frame's own JPEG | `cloud_backend/app.py` |
| Worker | Per camera: 2 fetch threads, 1 inference thread, 2 submit threads, own HTTP connections. Shared models behind a FIFO gate (no camera starves). Latest-only hand-offs with drop counters. Out-of-order frames are skipped, never fed to the tracker backwards. Timeouts with back-off, rate-limited warnings, fatal on a bad key. Falls back to the old polling loop automatically against an old backend | `cloud_backend/worker.py` |
| Phone page | Rate selector 3/5/8. Fixed capture clock with ≤2 uploads in flight (skips and counts when busy, never queues). Native aspect ratio, longest side 640, no letterbox. New session only when *this* session expired. Screen wake lock; stops if backgrounded. Visible page version `2026-11-02.2` | `cloud_backend/camera.html` |
| Monitor | Shows only processed frames, each tied to its frame ID and capture time by JPEG hash. Aspect ratio kept. Shows processed FPS, gap p95, capture→screen age, backend verdict and its frame. Dims with "NO FRESH PROCESSED FRAMES – UNKNOWN" after 15 s. Optional playout buffer (`--playout auto`) re-times **real** frames to capture cadence and **states the added delay**; it never interpolates or replays frames | `monitor.py` |
| Zones | Per-camera normalised polygons and per-zone `required_ppe` / `action_checks` in one YAML. The backend derives zones (with 2-frame hysteresis) and serves the layout to the worker | `cloud_backend/zone_policy.py`, `cloud_backend/config/camera_zones.yaml`, `docs/ZONE_CALIBRATION.md`, `calibrate_zones.py` |
| PPE evidence | Scored detections, 0.5 decision threshold. A close positive/negative conflict → unknown. One owner per box, head-band plausibility, too-small person → unknown. See `performance/MODEL_EVALUATION.md` | `ai_detector.py` |
| Rules | `face_touch` and `handwash_status` are consumed (alerting only with `PPE_CHECK_ACTIONS=true`). Status set: VIOLATION / PENDING / UNKNOWN / COMPLIANT / MONITORING / NO_PERSON. Incidents are closed as `ZONE_EXIT`, `TRACK_LOST` or `SESSION_RESET`, never as a fake `RECOVERY`. Events gain a `zone` column (auto-migrated) | `cloud_backend/app.py` |
| Handwash | Time comes from **capture timestamps**. Only intervals between two consecutive active samples ≤1.0 s apart count; pauses, gaps, tracking loss, leaving the sink and stale or out-of-order frames add nothing. Active means: both wrists visible, inside the sink polygon, **below the shoulders**, hands together, back-and-forth motion. "Incomplete" requires 2 frames away after partial washing. Per camera, session and track only | `ai_detector.py`, `HandwashTimer` |
| ESP32 | Green requires every camera fresh and in COMPLIANT/MONITORING/NO_PERSON, with at least one COMPLIANT. Unknown, stale or no-camera → amber. One buzzer per incident until acknowledged. Payload shape is unchanged (≈1.9 KB with 4 people) | `cloud_backend/app.py` |
| Tests | 49 tests: latest-frame/claim/snapshot, long-poll, two-camera fairness and freshness at 0/150 ms RTT, backend-restart recovery, zones and letterbox, uncertainty, hysteresis, incident closure, action checks, ESP32, DB migration, handwash FPS-invariance, real-weight detector contract. Also fixes the long-broken `test_zones_basic.py` | `tests_cloud/*` |

## 3. Measured before/after

**A. Live pipeline A/B on this Mac.** Real detector on MPS, real backend process, two simulated phones
replaying `review_video.mp4`, **270 ms ± 60 ms injected RTT on every request**. Source:
`performance_run/ab/*.json` (`python performance/live_ab.py --mode … --duration 60`). The simulation
reproduces your real "before" (0.54 vs 0.60 FPS measured live).

| per camera (mean of both) | before: deployed page (1/s) | before: 5/s page | **after (5/s)** |
|---|---:|---:|---:|
| processed FPS | 0.55 | 0.88 | **5.01** |
| uploads rejected 429 | 28% | 38% | **0%** |
| uploaded frames never processed (60 s) | 51 | 252 | **0** |
| result interval p50 / p95 | 2088 / 2197 ms | 1145 / 1264 ms | **198 / 274 ms** |
| capture→decision mean / p50 / p95 | 1285 / 1337 / 1677 ms | 933 / 939 / 1110 ms | **760 / 757 / 833 ms** |
| detector p50 / p95 | 84 / 103 ms | 84 / 102 ms | 86 / 109 ms |
| screen-update interval CV (monitor) | (old monitor) | (old monitor) | 0.20 off → **0.06** with `--playout auto` (+86 ms stated) |
| harness process CPU (worker + simulated phones) | 172% | 216% | **692%** |
| harness process peak RSS | 712 MiB | 545 MiB | 504 MiB |
| torch MPS allocator (current / driver) | 43 / 99 MiB | 43 / 99 MiB | 42 / 91 MiB |

Not covered by this simulation: real Render, phone JPEG encoding, mobile-radio variance and bandwidth, and
phone clock offset (all phones were the Mac clock here).

**B. Detector-only, same 60-frame CAM_01 clip** (`benchmark_performance.py`, MPS, run back-to-back under
the same background load): previous detector **88.0 ms mean / 116 ms p95 / 11.37 FPS** → this branch
**86.9 / 113 / 11.51 FPS**. YOLO: 7.221 GFLOPs per frame (PPE 640 + person 416, batch 1; excludes
MediaPipe, tracking and I/O), 6,169,218 parameters (weights unchanged). Portrait uploads letterbox inside
YOLO to the same pixel count, so FLOPs are the same. Your earlier figures (3.38 → 4.75 FPS CPU,
12.39 FPS MPS) were taken without the background worker and monitor and are not directly comparable to
the second table.

**C. Not yet measured (needs the real deployment):** live accepted FPS, frame age and capture→decision
on Render with real phones, and physical ESP32 alert delay. After deploying, collect them with the
commands in §5 and `performance_run/live.summary.json`.

**Cost to know:** 10 processed FPS in total pushes CPU to ≈7 cores on a fanless MacBook Air. If the
monitor shows detector time rising after a few minutes (thermal throttling), choose **3 frames/s** on
both phones.

## 4. Render (needs your go-ahead: push and deploy)

1. Push this branch, then either point the Render service at `fix/live-pipeline-zones` or merge it into
   `main` with a PR. Render currently deploys `main` (commit `fb8e995`).
2. Build command `pip install -r requirements-cloud.txt` (now includes PyYAML). Start command
   `python -m cloud_backend.serve`. Health check `/healthz`. Instance count **1**.
3. Environment: keep `PPE_API_KEY`; set `PPE_CAMERA_IDS=CAM_01,CAM_02`. `PPE_CHECK_ACTIONS=false` gives
   the PPE-only demo; `true` alerts on face touch and incomplete handwash. **Delete `PPE_HANDWASH_ROI`**
   unless you want it to override the YAML sink polygon. `PPE_ZONE_MODE` is no longer read.
4. Verify after "Live":
   ```bash
   curl -s https://ppe-gmp-compliance.onrender.com/healthz            # api_version 2026-11-02.latest-frame-zones
   curl -s https://ppe-gmp-compliance.onrender.com/camera | grep -o 'ppe-page-version" content="[^"]*'
   ```
   On each phone, reload `/camera`; the footer must say **Page version 2026-11-02.2**.
5. **Storage:** without a persistent disk (paid) mounted at `PPE_DATA_DIR=/var/data`, Render's filesystem
   is ephemeral. Events and snapshots are **lost on every redeploy, restart or free-tier spin-down**. Live
   state is always in memory; after any restart all cameras are UNKNOWN until fresh results arrive (tested).
6. **Latency floor:** about 270 ms per round trip to the current region. Capture→decision cannot go much
   below about 3 half-trips plus inference (≈0.6–0.75 s) from here. A service in a nearer region (for
   example Singapore) would lower it; that is an estimate, not a measurement. Render cannot change the
   region of an existing service.

## 5. Mac commands (after the backend is deployed)

Stop the old worker and monitor first (Ctrl+C in their terminals, or press `q` in the monitor window to
finalise its MP4).

```bash
# Terminal 1 — worker
cd ~/Desktop/ppe-gmp-compliance && source .venv/bin/activate
git branch --show-current                      # fix/live-pipeline-zones
export PPE_API_KEY="$(pbpaste)"                # copy the Render key to the clipboard first
python -m cloud_backend.worker --url https://ppe-gmp-compliance.onrender.com \
  --detector ai_detector:detect --metrics performance_run/live.jsonl
# Expect: "mode pipelined long-poll", cameras ['CAM_01','CAM_02'], then a status line every 10 s.

# Terminal 2 — monitor (and recording)
cd ~/Desktop/ppe-gmp-compliance && source .venv/bin/activate
python monitor.py --cameras CAM_01,CAM_02 --record demo_$(date +%H%M).mp4
#   --playout off   → lowest latency, raw arrival cadence
#   --playout auto  → smooth (default); the added delay is printed on screen

# After Ctrl+C on the worker: per-camera FPS, interval CV, latency, drops
python -c "import json;print(json.dumps(json.load(open('performance_run/live.summary.json'))['per_camera'],indent=1))"
```

**Phones:** open `https://ppe-gmp-compliance.onrender.com/camera` and check the page version. Phone 1
selects CAM_01 and phone 2 selects CAM_02. Rate 5 (or 3 if the Mac gets hot). Paste the key, press Start,
and keep the page in the foreground. Mount the phones fixed, in the orientation used for calibration
(`docs/ZONE_CALIBRATION.md`).

If the backend is **not** redeployed, the new worker detects the old API and falls back to the old polling
loop. It runs, but the 1/s phone page and 429 behaviour remain, and the old backend's 640×360 zone
assumptions apply.

## 6. Two-camera demo checklist (≈10 min)

1. Monitor shows both tiles near the phone rate. `curl …/v1/cameras/CAM_02 -H "X-API-Key: $PPE_API_KEY"`
   shows `pipeline.superseded_before_claim` staying low.
2. **CAM_02, all PPE:** COMPLIANT (green only if CAM_01 is not violating). Expect UNKNOWN if gloves are not
   recognised (see the model limitation).
3. **CAM_02, remove the mask:** PENDING → VIOLATION after 3 of 5 processed frames; red LED and buzzer
   once; acknowledge; put the mask back → RECOVERY after 3 clear frames.
4. **CAM_02, turn away / cover the head:** UNKNOWN (amber), never COMPLIANT.
5. **CAM_01:** walk the corridor → MONITORING, label CORRIDOR. Step to the sink → HANDWASH; rub hands
   below the shoulders → `wash x/20s washing`; pause → `paused` (timer stops). With
   `PPE_CHECK_ACTIONS=true`: leave early → `incomplete` → VIOLATION `handwash`.
6. **Disconnect:** stop the CAM_02 phone → within 15 s CAM_02 is UNKNOWN, the tile is dimmed and the
   ESP32 is not green. Restart → a new session and fresh IDs.
7. **Worker crash:** Ctrl+C the worker → both cameras are UNKNOWN within 15 s. Restart → it resumes.
8. **Events:** `curl -s -H "X-API-Key: $PPE_API_KEY" "https://ppe-gmp-compliance.onrender.com/v1/events?limit=10" | python -m json.tool`
   shows VIOLATION / RECOVERY with snapshot URLs, and TRACK_LOST / SESSION_RESET without.

## 7. Verified here vs. needs real-world testing

**Verified on this Mac:**
- All 49 tests pass, and the local launcher passes the end-to-end smoke test.
- The A/B pipeline measurement with injected RTT.
- The real-weight replay through real backend rules (`performance/replay_validation.py`): gloves violation
  correct; mask violation when removed; hairnet violation delayed by the model's false positives;
  raised-hand false washing removed.
- The DB migration on an old-schema database.
- Detector speed is unchanged.

**Not verified (no access, or needs your hardware):**
- The deployed Render service running this code.
- Real phones (encoding time, clock offset, Safari/Chrome wake-lock and visibility behaviour).
- The ESP32 firmware (not in the repo) and its JSON buffer size, physical alert delay, and ESP32
  disconnect/reconnect.
- The real sink polygon calibration.
- Detection accuracy on site (no labelled data); handwash validity (motion near a sink, not soap, water
  or technique).

**Known limitations:**
- Weights: hairnet false positives on hair and weak glove recall.
- No cross-camera identity.
- The k-of-n vote counts frames, so time-to-alert scales with processed FPS (≈0.6 s of frames at 5 FPS
  vs ≈3 s at 1 FPS).
- Handwash evidence needs ≳2 processed FPS on CAM_01; below that, the 1 s gap rule correctly counts nothing.
- Event storage is ephemeral on Render without a disk.
- `dashboard/app.py` reads only a local SQLite file, not Render.
