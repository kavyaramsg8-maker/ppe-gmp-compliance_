# PPE performance update — demo-setup / Mac

This is a small overlay, **not a full repository**. It targets demo-setup commit `2ee36267bd7129ae2d50700e80bb653cea895f8e`. Keep your existing weights and working Python 3.11 virtual environment. No model weights or credentials are included.

## 1. Apply safely (10 minutes)

Stop the local worker and monitor (Ctrl+C). Extract this ZIP separately, e.g. `~/Downloads/ppe-performance-update`. In your existing project:

```bash
cd ~/Desktop/ppe-gmp-compliance
source .venv/bin/activate
git branch --show-current
git status --short
```

The branch must be `demo-setup`. If another branch is open, save your existing work before switching; do not use reset/force-push. Run the installer from your **project terminal** (adjust the Downloads directory if Finder added a suffix):

```bash
python ~/Downloads/ppe-performance-update/apply_performance_update.py --repo "$PWD"
python -m pip install --no-cache-dir -r requirements-performance.txt
python -c "import torch, mediapipe as mp; print('MPS available:', torch.backends.mps.is_available()); print('Legacy pose available:', hasattr(mp, 'solutions'))"
```

The installer refuses to overwrite changed source files and saves original files in a timestamped backup folder. If it refuses, send its message; don't manually overwrite a teammate's changes. If `Legacy pose available` is false, retain/restore your known working MediaPipe environment before benchmarking. This update does not migrate MediaPipe APIs or require a new Torch download.

## 2. Capture a fair before/after benchmark (20–40 minutes)

Use one existing representative video containing a visible person, PPE and hand motion. Replace `review_video.mp4` if your video has a different name. Both commands use the **same clip, CPU, 640px PPE input and number of frames**; baseline keeps the old person model at 640, optimized uses 416. Baseline source is preserved in `performance/baseline_detector.py`. Run separately, not simultaneously with the worker. Warmup frames are excluded. Ensure the clip has at least 65 frames.

```bash
python benchmark_performance.py --variant baseline --source review_video.mp4 --device cpu --output performance_run/baseline_cpu.json
python benchmark_performance.py --variant optimized --source review_video.mp4 --device cpu --output performance_run/optimized_cpu.json
```

If the earlier check says MPS is available, also test Apple GPU:

```bash
python benchmark_performance.py --variant optimized --source review_video.mp4 --device mps --output performance_run/optimized_mps.json
```

Use whichever device actually performs better; MPS can have overhead with small inputs. For an ablation isolating shared-pose/network changes from person resolution, rerun optimized with `--person-imgsz 640`. Test CAM_02 independently using `--camera CAM_02` and a production clip. Do not compare two different clips or call a CPU-to-MPS improvement an algorithm-only speedup.

Generate a ready-to-use table after the runs:

```bash
python compare_performance.py performance_run/baseline_cpu.json performance_run/optimized_cpu.json > performance_run/comparison.md
```

Read `performance_run/*.json`. Do not fill in numbers before running. Reports provide:

| Metric | What it means |
|---|---|
| `models[].gflops` | YOLO forward-pass GFLOPs at the stated letterboxed shape, batch 1. Excludes MediaPipe and application work. Null means unsupported/missing profiler. |
| `models[].parameters` | Learned parameter count per YOLO network. Weights are unchanged: parameter count is expected to remain unchanged. |
| `detector_throughput_fps` | Sequential detector calls / detector wall time after warmup. No cloud/network. |
| `detector_latency_ms` | Mean, median and p95 detector wall time, with GPU synchronization. Includes decoding, tracking, pose and preview writes. |
| `peak_sampled_process_rss_mib` | Peak sampled process host RAM, including models and encoded test inputs. Not GPU/unified-memory allocation attribution. |

FLOPs are not latency: measured elapsed time is the deployment evidence. Total system FLOPs/MediaPipe parameter count are **not** claimed by the YOLO profiler. Two cameras load two person-tracker model instances (more resident RAM), while the PPE and pose model are shared. FLOPs depend on people count/pose workload; YOLO figures alone are not the full pipeline cost.

## 3. Publish the phone-page change to Render (10–20 minutes)

After local validation, review `git diff`. Stage only the update files, not virtual environments, recordings, secrets, generated reports or backups:

```bash
git add ai_detector.py monitor.py cloud_backend/worker.py cloud_backend/camera.html performance_tools.py benchmark_performance.py compare_performance.py requirements-performance.txt performance tests_cloud/test_performance.py PERFORMANCE_START_HERE.md
git commit -m "Reduce PPE frame bottlenecks and add measured performance reporting"
git push origin demo-setup
```

In Render → your service → Settings, confirm **Branch = demo-setup**. Deploy latest commit. Changing these settings only changes the upload page/backend checkout; inference still runs on your Mac. On both phones reload the HTTPS `/camera` page. The new page must show **Target upload rate**. Start at 5 FPS; 8 FPS is optional if it actually helps. CAM_01 = corridor/gowning, with sink region in right half; CAM_02 = separate production phone. Use landscape and fixed camera mounts. This update preserves those regions rather than introducing three artificial columns.

## 4. Run the live demo and measure actual accepted throughput (20–40 minutes)

Terminal 1, existing project with its API key set locally (never commit/paste the key into a report):

```bash
source .venv/bin/activate
export PPE_DEVICE=cpu
python -m cloud_backend.worker --url https://ppe-gmp-compliance.onrender.com --detector ai_detector:detect --metrics performance_run/live.jsonl
```

Set `PPE_DEVICE=mps` instead only if your benchmark is faster and valid. `auto` also selects available GPU, but an explicit measured choice is easier to reproduce. Keep **one worker process**; two network threads share a serial inference lock.

Terminal 2:

```bash
source .venv/bin/activate
python monitor.py --cameras CAM_01,CAM_02 --record demo_optimized.mp4
```

Start both phones, each with a different camera ID. Run for at least 60 seconds with the same scenario. Press **q in the monitor window** to save recording, then Ctrl+C in worker to write `performance_run/live.summary.json`. The worker does not require the monitor to run. The recording has 15 FPS playback; repeated frames are expected whenever processing is slower. The overlay shows the actual arrival rate of new processed images; it does not manufacture intermediate detections.

`live.summary.json` gives **accepted processed FPS**, per-camera FPS, detector/cycle latency and capture-to-response age. Live throughput uses complete elapsed run time, including idle periods; start cameras promptly. Capture age relies on synchronized phone/Mac clocks and includes upload, queueing, inference and result submission. It is not physical LED/buzzer delay. Use a timed physical test for that.

## What changed and why

- Original phone page waited one second per upload and fetched status inline on every frame. Now it targets 3/5/8 FPS, polls status separately, preserves aspect ratio and retains only one in-flight upload. Busy frames are dropped, not queued.
- Network I/O for two cameras can overlap while shared model execution remains serial. Original lease/backpressure and observation checks remain intact.
- Pose evidence is computed once per person and reused by face-touch and sink-motion checks (previously twice in the sink). Face-touch and handwash rules remain present.
- Person tracking filters the person class and uses a smaller configurable input; this needs detection/ID stability validation. PPE resolution remains 640 and class names/weights are unchanged.
- Trackers and action history are scoped per camera/session. JPEGs publish atomically, so the monitor never reads a partly written file. Per-frame debug spam is disabled by default (`PPE_DEBUG=1` restores it).
- Monitor remains responsive, shows stale/unknown status and records optionally. A smooth UI refresh is not a claim of 30 processed FPS.

**Expected limitation:** Cloud round trips, inference and the existing claim lease still bound processed FPS. If each detector call takes one second, two cameras cannot each deliver five processed FPS. This patch removes avoidable work; actual improvement must come from the reports. Don't reduce PPE resolution or skip action checks solely to get a flattering number. If testing `PPE_IMGSZ=480`, evaluate small gloves/hairnets against the 640 baseline first.

## Accuracy and feature gate (45–60 minutes)

For each camera test: PPE present, each missing item, obscured PPE, empty scene, two people, entering/leaving sink, face touch, session restart and disconnect. Compare the same annotated frames before/after. Count TP/FP/FN per PPE class; check person ID switches. Confirm violations, recovery, stale handling and physical alerts using actual hardware. Keep person size at 640 (`PPE_PERSON_IMGSZ=640`) if 416 loses distant people. Existing temporal voting means a changed sample rate can change time-to-alert even with unchanged vote settings: measure it; do not silently change compliance thresholds.

Important existing branch limitations: handwash/face-touch fields are produced by the detector but the current backend Person schema consumes only `track_id` and `associated_ppe`; those action fields are not a confirmed backend compliance verdict. Sink geometry is the branch's existing fixed right-half rule, not physical calibration. The original handwash timer is a heuristic, not validated hygiene verification. This performance update does not claim to repair or validate those separate functional limitations.

## Test status and remaining six-hour plan

Validated here with dependency doubles (no user weights or Mac GPU available): targeted backend, worker, smoothing and performance regressions. `tests_cloud/test_performance.py` checks pose reuse, camera isolation, overlapping downloads with serial inference, atomic writes and honest throughput accounting. Exact pass count is in `performance/VALIDATION.md`. Full pytest on the inspected files has an existing import error: `test_zones_basic.py` imports `point_in_polygon`, absent from `rule_engine/zones.py`. No all-suite success claim is made.

Budget the remaining time: 30 min apply/preflight, 60 min CPU/MPS benchmarks, 45 min two-camera cloud test, 60 min accuracy/alert regression, 45 min demo recording/report table, 120 min buffer for model/environment/network issues. Prioritize a reproducible measured demo over adding features. A grade depends on the rubric, accuracy, integration and your explanation; these changes cannot guarantee 30 marks.
