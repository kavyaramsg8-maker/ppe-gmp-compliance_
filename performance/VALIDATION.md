# Validation — 2026-11-02

Source: demo-setup commit 2ee36267bd7129ae2d50700e80bb653cea895f8e.

Executed in Linux/Python 3.12, temporary isolated test dependencies:

```
PYTHONPATH=/tmp/ppe-perf-deps:. python -m pytest tests_cloud test_smoothing_basic.py -q
17 passed, 1 warning
```

Tests cover backend authentication/frame lifecycle/events/stale handling, worker submission, camera network overlap with serial inference, one pose pass per tracked person, independent camera trackers, atomic images, throughput accounting, and benchmark CLI/report structure for baseline/optimized variants using model doubles. The warning is a Starlette/httpx deprecation in the temporary test environment.

Python compilation and browser script syntax check pass. These are software checks, not evidence of model accuracy, real FPS, GPU compatibility or physical alerts.

`pytest -q` on the inspected source set fails collection because the existing `test_zones_basic.py` imports a missing `point_in_polygon` from `rule_engine/zones.py`. It is not silently excluded in an all-tests claim. No full repository test pass is claimed.

Real weights, MediaPipe inference, camera feeds, Render latency, Mac MPS, physical ESP32, and before/after measurements were unavailable here. Run the included benchmark on the user's Mac and perform live acceptance testing. The reports intentionally contain no invented measurement results.

Installer tested against a reconstructed matching branch: 14 files applied, backups created, repeated application safely refused.
