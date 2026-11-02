"""Worker pipeline against a real uvicorn backend, two simulated phones and network delay."""
from io import BytesIO
import threading
import time

import pytest
from PIL import Image

from cloud_backend.app import Settings, create_app
from cloud_backend.worker import FairGate, LatestSlot, call_detector, run_pipelines
from performance_tools import Recorder
from live_harness import Backend, DelayTransport, Phone
from test_backend import KEY, ALL_PRODUCTION


def jpeg_bytes(shade):
    out = BytesIO(); Image.new('RGB', (64, 48), (shade, shade, shade)).save(out, format='JPEG'); return out.getvalue()


FRAMES = [jpeg_bytes(s) for s in range(0, 250, 25)]


def test_latest_slot_keeps_newest_and_counts_drops():
    slot = LatestSlot()
    slot.put(1); slot.put(2); slot.put(3)
    assert slot.get(0.01) == 3 and slot.replaced == 2
    assert slot.get(0.01) is None


def test_fair_gate_serves_in_arrival_order():
    gate = FairGate(); order = []; started = threading.Event()

    def hold():
        with gate:
            started.set(); time.sleep(0.15)

    def user(name):
        with gate:
            order.append(name)

    threading.Thread(target=hold).start(); started.wait()
    threads = []
    for name in 'ABCD':
        t = threading.Thread(target=user, args=(name,)); t.start(); threads.append(t); time.sleep(0.02)
    for t in threads:
        t.join()
    assert order == list('ABCD')


def test_detector_receives_identity_only_when_it_accepts_it():
    class F:  # minimal frame
        jpeg, camera_id, session_id, frame_id, content_box = b'x', 'CAM_01', 's', 7, (0, 0, 4, 4)
        captured_at = None
    assert call_detector(lambda b, c, s: (b, c, s), F) == (b'x', 'CAM_01', 's')
    assert call_detector(lambda b, c, s, frame_id=None, **kw: (frame_id, kw['content_box']), F) == (7, (0, 0, 4, 4))


@pytest.mark.parametrize('rtt_ms', [0, 150])
def test_two_cameras_flow_fairly_without_backlog_or_blocking(tmp_path, monkeypatch, rtt_ms):
    monkeypatch.setenv('PPE_MONITOR_DIR', str(tmp_path))
    active = []; peak = [0]; guard = threading.Lock(); calls = {'CAM_01': 0, 'CAM_02': 0}

    def detector(image, camera_id, session_id, frame_id=None, captured_at=None, content_box=None):
        with guard:
            active.append(1); peak[0] = max(peak[0], len(active)); calls[camera_id] += 1
        time.sleep(0.06)  # shared-model inference time
        with guard:
            active.pop()
        assert captured_at is not None and content_box == (0, 0, 64, 48)
        return [{'track_id': 1, 'associated_ppe': [{'class': c, 'confidence': .9} for c in ('hairnet', 'maskon', 'glove')]}]

    app = create_app(Settings(api_key=KEY, data_dir=tmp_path, zone_config=ALL_PRODUCTION))
    recorder = Recorder(tmp_path / 'live.jsonl')
    with Backend(app) as backend:
        stop = threading.Event()
        phones = [Phone(backend.url, KEY, cam, lambda t: FRAMES[int(t * 5) % len(FRAMES)], fps=5, rtt_ms=rtt_ms, stop=stop)
                  for cam in ('CAM_01', 'CAM_02')]
        for p in phones:
            p.start()
        time.sleep(0.3)
        pipelines, _ = run_pipelines(backend.url, KEY, detector, ['CAM_01', 'CAM_02'], recorder, wait=2.0, stop=stop,
                                     transport=lambda: DelayTransport(rtt_ms))
        time.sleep(4.0)
        stop.set()
        for p in phones:
            p.join(3)
        time.sleep(0.3)
    summary = recorder.close()
    assert peak[0] == 1  # shared models never run concurrently
    for p in phones:
        assert p.counts['busy_429'] == 0 and p.counts['errors'] == 0
        assert p.counts['accepted'] >= 15  # ~5/s uploads were not throttled by inference
    per = summary['per_camera']
    fps = [per[c]['accepted_fps'] for c in ('CAM_01', 'CAM_02')]
    assert min(fps) >= 3.0, per  # near the 5 FPS upload rate, not ~1 FPS
    assert min(fps) / max(fps) > 0.7  # neither camera starves the other
    for c in ('CAM_01', 'CAM_02'):
        # Fresh: decisions are about the latest uploads, no growing queue.
        assert per[c]['capture_to_decision_ms']['p95'] < 400 + 3 * rtt_ms, per[c]
        assert pipelines[0].counts['errors'] == 0 and pipelines[1].counts['errors'] == 0


def test_worker_recovers_from_backend_restart_and_never_reports_stale_as_fresh(tmp_path, monkeypatch):
    monkeypatch.setenv('PPE_MONITOR_DIR', str(tmp_path))
    detector = lambda image, camera_id, session_id, **kw: [{'track_id': 1, 'associated_ppe': [
        {'class': c, 'confidence': .9} for c in ('hairnet', 'maskon', 'glove')]}]
    settings = Settings(api_key=KEY, data_dir=tmp_path, zone_config=ALL_PRODUCTION)
    first = Backend(create_app(settings))
    stop = threading.Event()
    with first:
        port = first.port
        phone = Phone(first.url, KEY, 'CAM_02', lambda t: FRAMES[0], fps=5, stop=threading.Event()); phone.start()
        pipelines, _ = run_pipelines(first.url, KEY, detector, ['CAM_02'], wait=1.0, stop=stop)
        time.sleep(1.5); phone.stop.set(); phone.join(2)
        before = pipelines[0].counts['accepted']
    assert before > 0
    time.sleep(1.0)  # backend down: worker keeps retrying with back-off, no crash
    with Backend(create_app(settings), port=port) as second:
        import httpx
        # Restarted backend: in-memory state is gone, so the camera is UNKNOWN until fresh results.
        assert httpx.get(second.url + '/v1/cameras/CAM_02', headers={'X-API-Key': KEY}).json()['status'] == 'UNKNOWN'
        phone = Phone(second.url, KEY, 'CAM_02', lambda t: FRAMES[1], fps=5, stop=threading.Event()); phone.start()
        time.sleep(2.0); phone.stop.set(); phone.join(2)
        stop.set()
        assert pipelines[0].counts['accepted'] > before + 3  # resumed on the new session
        assert pipelines[0].counts['errors'] > 0 and pipelines[0].fatal is None
