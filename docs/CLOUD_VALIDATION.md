# Validation performed

- 13 automated tests passed, including the three existing smoothing/zone tests.
- A live Uvicorn HTTP server completed the explicit synthetic backend smoke test:
  COMPLIANT → VIOLATION → COMPLIANT, with event/snapshot storage and simulated
  ESP32 polling/acknowledgement.
- Tests cover API-key rejection, camera/session separation, stale timestamps,
  duplicate frames/results, unknown observations, recovery, event persistence,
  incident acknowledgements and worker upload backpressure.
- Dependency emits a Starlette/httpx deprecation warning during TestClient use;
  this did not fail tests or the live HTTP smoke test.

Not validated here: real model inference, phone camera permission/streaming on a
physical device, Docker image build, paid cloud deployment, physical ESP32 or
end-to-end detection accuracy/latency.
