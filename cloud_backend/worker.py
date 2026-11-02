"""Pipelined inference worker: per-camera fetch → shared detector → submit.

Each camera has its own threads and HTTP connections, so a slow upload or network
stall on one camera cannot hold up the other. The shared models run one frame at a
time behind a first-come-first-served gate, so neither camera can starve. Each stage
keeps only the newest item (no backlog); every dropped item is counted, never hidden.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
import importlib
import inspect
import json
import logging
import os
from pathlib import Path
import threading
import time
import httpx
from performance_tools import Recorder, atomic_bytes, rss_mib

log = logging.getLogger(__name__)
_infer_lock = threading.Lock()


class LatestSlot:
    """Single-item hand-off: a new item replaces an unconsumed one."""

    def __init__(self):
        self._item = None
        self._cond = threading.Condition()
        self.replaced = 0

    def put(self, item):
        with self._cond:
            if self._item is not None:
                self.replaced += 1
            self._item = item
            self._cond.notify()

    def get(self, timeout):
        with self._cond:
            if self._item is None:
                self._cond.wait(timeout)
            item, self._item = self._item, None
            return item


class FairGate:
    """FIFO mutex: waiters enter in arrival order (threading.Lock gives no such promise)."""

    def __init__(self):
        self._cond = threading.Condition()
        self._next_ticket = 0
        self._serving = 0

    def __enter__(self):
        with self._cond:
            ticket = self._next_ticket
            self._next_ticket += 1
            while ticket != self._serving:
                self._cond.wait()

    def __exit__(self, *exc):
        with self._cond:
            self._serving += 1
            self._cond.notify_all()


@dataclass
class Frame:
    camera_id: str
    session_id: str
    frame_id: int
    jpeg: bytes
    captured_at: datetime | None
    content_box: tuple | None
    metrics: dict = field(default_factory=dict)
    persons: list | None = None


def _parse_time(value):
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def _ms_between(later, earlier):
    return round((later - earlier).total_seconds() * 1000, 1) if later and earlier else None


def call_detector(detector, frame):
    """Pass frame identity/timestamps to detectors that accept them (older ones take 3 args)."""
    try:
        params = inspect.signature(detector).parameters
    except (TypeError, ValueError):
        params = {}
    extra = {'frame_id': frame.frame_id,
             'captured_at': frame.captured_at.timestamp() if frame.captured_at else None,
             'content_box': frame.content_box}
    accepts_any = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    kwargs = {k: v for k, v in extra.items() if accepts_any or k in params}
    return detector(frame.jpeg, frame.camera_id, frame.session_id, **kwargs)


def write_status(camera_id, value):
    # No API key or image data in this status file.
    safe = ''.join(c for c in camera_id if c.isalnum() or c in '_-')
    folder = Path(os.environ.get('PPE_MONITOR_DIR', str(Path(__file__).resolve().parent.parent)))
    atomic_bytes(folder / f'live_{safe}.status.json', json.dumps(value).encode())


class CameraPipeline:
    def __init__(self, camera_id, make_client, detector, gate, recorder=None, wait=5.0, stop=None, connections=2):
        self.camera_id, self.make_client, self.detector = camera_id, make_client, detector
        self.gate, self.recorder, self.wait = gate, recorder, wait
        # Concurrent fetches/submits per camera: with one of each, throughput is capped at 1/RTT.
        self.connections = connections
        self.stop = stop or threading.Event()
        self.frames, self.results = LatestSlot(), LatestSlot()
        self.last_inferred = (None, -1)
        self.counts = {'claimed': 0, 'inferred': 0, 'accepted': 0, 'rejected': 0, 'errors': 0, 'out_of_order': 0}
        self.fatal = None
        self._last_log = {}

    def _warn(self, key, message, *args):
        # Rate-limited: a disconnected phone or network must not flood the terminal.
        if time.monotonic() - self._last_log.get(key, 0) > 5:
            self._last_log[key] = time.monotonic()
            log.warning(f'{self.camera_id}: ' + message, *args)

    def _backoff(self, attempt):
        self.stop.wait(min(5.0, 0.25 * 2 ** min(attempt, 5)))

    def fetch_loop(self):
        after, session, failures = -1, None, 0
        with self.make_client(read_timeout=self.wait + 15) as client:
            while not self.stop.is_set():
                try:
                    started = time.perf_counter()
                    params = {'after': after, 'wait': self.wait}
                    if session:
                        params['session'] = session
                    r = client.get(f'/v1/cameras/{self.camera_id}/frame', params=params, headers={'X-Claim-Frame': 'true'})
                    if r.status_code == 401:
                        self.fatal = 'API key rejected'; self.stop.set(); return
                    if r.status_code == 204:
                        failures = 0; continue
                    if r.status_code in (404, 410):
                        self._warn('nofresh', 'no fresh frame (phone stopped or disconnected?)')
                        self.stop.wait(0.5); continue
                    r.raise_for_status()
                    fetched = time.perf_counter(); failures = 0
                    session, after = r.headers['X-Session-ID'], int(r.headers['X-Frame-ID'])
                    captured = _parse_time(r.headers.get('X-Captured-At'))
                    received = _parse_time(r.headers.get('X-Received-At'))
                    served = _parse_time(r.headers.get('X-Server-Time'))
                    box = tuple(float(v) for v in r.headers['X-Content-Box'].split(',')) if r.headers.get('X-Content-Box') else None
                    now = datetime.now(timezone.utc)
                    frame = Frame(self.camera_id, session, after, r.content, captured, box, metrics={
                        'camera_id': self.camera_id, 'frame_id': after, 'session_id': session,
                        'fetch_ms': round((fetched - started) * 1000, 1),
                        # Phone clock → server clock: includes any phone clock offset.
                        'upload_ms': _ms_between(received, captured),
                        # Same server clock: time the frame waited for this claim.
                        'server_wait_ms': _ms_between(served, received),
                        'age_at_fetch_ms': _ms_between(now, captured),
                        'jpeg_bytes': len(r.content), '_fetched': fetched})
                    self.counts['claimed'] += 1
                    self.frames.put(frame)
                except (httpx.HTTPError, KeyError, ValueError) as e:
                    self.counts['errors'] += 1; failures += 1
                    self._warn('fetch', 'frame fetch failed (%s); retrying', type(e).__name__)
                    self._backoff(failures)

    def infer_loop(self):
        while not self.stop.is_set():
            frame = self.frames.get(timeout=0.5)
            if frame is None:
                continue
            # Concurrent fetches can land out of order; never run the tracker backwards.
            if frame.session_id == self.last_inferred[0] and frame.frame_id <= self.last_inferred[1]:
                self.counts['out_of_order'] += 1
                continue
            self.last_inferred = (frame.session_id, frame.frame_id)
            queued = time.perf_counter()
            try:
                with self.gate:
                    start = time.perf_counter()
                    frame.persons = call_detector(self.detector, frame)
                    end = time.perf_counter()
            except Exception:
                self.counts['errors'] += 1
                log.exception('%s: detector failed on frame %s; no result submitted', self.camera_id, frame.frame_id)
                continue
            frame.metrics.update(local_queue_ms=round((queued - frame.metrics['_fetched']) * 1000, 1),
                                 gate_wait_ms=round((start - queued) * 1000, 1),
                                 detector_ms=round((end - start) * 1000, 1), _inferred=end)
            self.counts['inferred'] += 1
            self.results.put(frame)

    def submit_loop(self):
        failures = 0
        with self.make_client(read_timeout=15) as client:
            while not self.stop.is_set():
                frame = self.results.get(timeout=0.5)
                if frame is None:
                    continue
                started = time.perf_counter()
                try:
                    r = client.post(f'/v1/cameras/{self.camera_id}/observations',
                                    json={'session_id': frame.session_id, 'frame_id': frame.frame_id, 'persons': frame.persons})
                    failures = 0
                except httpx.HTTPError as e:
                    self.counts['errors'] += 1; failures += 1
                    self._warn('submit', 'result submit failed (%s); result dropped, no compliant substitute', type(e).__name__)
                    self._backoff(failures)
                    continue
                end = time.perf_counter()
                m = frame.metrics
                accepted = r.status_code == 200
                m.update(accepted=accepted, status_code=r.status_code, submit_ms=round((end - started) * 1000, 1),
                         local_cycle_ms=round((end - m.pop('_fetched')) * 1000, 1),
                         age_at_decision_ms=_ms_between(datetime.now(timezone.utc), frame.captured_at),
                         dropped_before_inference=self.frames.replaced, dropped_before_submit=self.results.replaced,
                         rss_mib=rss_mib(), recorded_at=datetime.now(timezone.utc).isoformat())
                m.pop('_inferred', None)
                if accepted:
                    self.counts['accepted'] += 1
                    value = r.json()
                    m['server_capture_to_decision_ms'] = (value.get('pipeline') or {}).get('capture_to_decision_ms')
                    value['worker_metrics'] = m
                    write_status(self.camera_id, value)
                elif r.status_code == 409:
                    self.counts['rejected'] += 1
                    log.info('%s: frame %s result rejected as superseded/expired', self.camera_id, frame.frame_id)
                else:
                    self.counts['errors'] += 1
                    self._warn('submit_status', 'result submit returned HTTP %s', r.status_code)
                if self.recorder:
                    self.recorder.add(m)

    def start(self):
        roles = ['fetch_loop'] * self.connections + ['infer_loop'] + ['submit_loop'] * self.connections
        self.threads = [threading.Thread(target=getattr(self, name), name=f'{self.camera_id}-{name}-{i}', daemon=True)
                        for i, name in enumerate(roles)]
        for t in self.threads:
            t.start()
        return self


def run_pipelines(base_url, key, detector, cameras, recorder=None, wait=5.0, stop=None, transport=None, connections=2):
    stop = stop or threading.Event()
    gate = FairGate()

    def make_client(read_timeout):
        return httpx.Client(base_url=base_url.rstrip('/'), headers={'X-API-Key': key},
                            timeout=httpx.Timeout(10.0, read=read_timeout), transport=transport() if transport else None)

    pipelines = [CameraPipeline(c, make_client, detector, gate, recorder, wait, stop, connections).start() for c in cameras]
    return pipelines, stop


# ---- Legacy one-shot polling (old backends without long-poll claims; tests) ----

def _process_camera(client, detector, camera, recorder=None):
    if camera['latest_frame_id'] is None or camera['latest_frame_id'] == camera['processed_frame_id']:
        return 0
    camera_id = camera['camera_id']; start = time.perf_counter()
    frame = client.get(f'/v1/cameras/{camera_id}/frame', headers={'X-Claim-Frame': 'true'})
    if frame.status_code in (404, 410):
        return 0
    frame.raise_for_status(); session_id = frame.headers['X-Session-ID']
    fetched = time.perf_counter()
    captured = _parse_time(frame.headers.get('X-Captured-At'))
    item = Frame(camera_id, session_id, int(frame.headers['X-Frame-ID']), frame.content, captured, None)
    try:
        with _infer_lock:
            infer_start = time.perf_counter(); persons = call_detector(detector, item)
            infer_end = time.perf_counter()
        result = client.post(f'/v1/cameras/{camera_id}/observations', json={'session_id': session_id, 'frame_id': item.frame_id, 'persons': persons})
        end = time.perf_counter()
        if result.status_code != 409:
            result.raise_for_status()
        accepted = result.status_code != 409
        row = {'camera_id': camera_id, 'frame_id': item.frame_id, 'accepted': accepted,
               'detector_ms': (infer_end - infer_start) * 1000, 'fetch_ms': (fetched - start) * 1000,
               'gate_wait_ms': (infer_start - fetched) * 1000, 'submit_ms': (end - infer_end) * 1000,
               'local_cycle_ms': (end - start) * 1000, 'age_at_decision_ms': _ms_between(datetime.now(timezone.utc), captured),
               'rss_mib': rss_mib(), 'recorded_at': datetime.now(timezone.utc).isoformat()}
        if recorder:
            recorder.add(row)
        if not accepted:
            log.warning('Dropped expired/superseded result for %s', camera_id); return 0
        value = result.json(); value['worker_metrics'] = row
        write_status(camera_id, value)
        return 1
    except Exception:
        if recorder:
            recorder.add({'camera_id': camera_id, 'accepted': False, 'error': True, 'local_cycle_ms': (time.perf_counter() - start) * 1000, 'rss_mib': rss_mib()})
        raise


def process_once(client, detector, executor=None, recorder=None):
    response = client.get('/v1/cameras'); response.raise_for_status()
    cameras = response.json()['cameras']
    if executor is None:
        return sum(_process_camera(client, detector, c, recorder) for c in cameras)
    futures = [executor.submit(_process_camera, client, detector, c, recorder) for c in cameras]
    count = 0; error = None
    for f in futures:
        try:
            count += f.result()
        except Exception as e:
            error = e
    if error:
        raise error
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default=os.environ.get('PPE_BACKEND_URL', 'http://127.0.0.1:8000'))
    parser.add_argument('--detector', default=os.environ.get('PPE_DETECTOR'))
    parser.add_argument('--metrics', default='performance_run/live.jsonl')
    parser.add_argument('--cameras', help='Comma-separated IDs (default: all cameras configured on the backend)')
    parser.add_argument('--wait', type=float, default=5.0, help='Long-poll seconds per frame request')
    parser.add_argument('--connections', type=int, choices=[1, 2, 3], default=2, help='Concurrent fetches and submits per camera')
    parser.add_argument('--legacy-poll', action='store_true', help='Use the old list/claim polling loop')
    args = parser.parse_args(); key = os.environ.get('PPE_API_KEY')
    if not key or not args.detector or ':' not in args.detector:
        parser.error('Set PPE_API_KEY and --detector module:function')
    module_name, function = args.detector.split(':', 1)
    module = importlib.import_module(module_name); detector = getattr(module, function)
    if not callable(detector):
        parser.error('Detector must be callable')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    logging.getLogger('httpx').setLevel(logging.WARNING)
    with httpx.Client(base_url=args.url.rstrip('/'), timeout=15) as probe:
        config = probe.get('/camera-config'); config.raise_for_status(); config = config.json()
    cameras = [c.strip() for c in args.cameras.split(',')] if args.cameras else list(config['cameras'])
    if unknown := set(cameras) - set(config['cameras']):
        parser.error(f'Backend does not configure {sorted(unknown)}; it has {config["cameras"]}')
    legacy = args.legacy_poll or 'api_version' not in config
    log.info('Backend %s | cameras %s | actions enforced %s | mode %s', config.get('api_version', 'OLD (no long-poll)'),
             cameras, config.get('check_actions'), 'legacy polling' if legacy else 'pipelined long-poll')
    if 'zone_config' in config and hasattr(module, 'configure'):
        module.configure(config['zone_config'])  # backend layout is authoritative
    elif hasattr(module, 'configure'):
        log.warning('Backend sends no zone layout; detector uses its local camera_zones.yaml')
    recorder = Recorder(args.metrics)
    try:
        if legacy:
            with httpx.Client(base_url=args.url.rstrip('/'), headers={'X-API-Key': key}, timeout=30) as client, ThreadPoolExecutor(max_workers=2) as pool:
                while True:
                    try:
                        process_once(client, detector, pool, recorder)
                    except (httpx.HTTPError, RuntimeError, ValueError, KeyError, TypeError):
                        log.exception('Worker cycle failed; no compliant substitute')
                    time.sleep(.02)
        pipelines, stop = run_pipelines(args.url, key, detector, cameras, recorder, args.wait, connections=args.connections)
        last = time.monotonic()
        while not stop.wait(1.0):
            if time.monotonic() - last >= 10:
                last = time.monotonic()
                log.info(' | '.join(f'{p.camera_id}: {p.counts["accepted"]} ok, {p.counts["rejected"]} rejected, {p.counts["out_of_order"]} out-of-order, '
                                    f'{p.frames.replaced}+{p.results.replaced} dropped, {p.counts["errors"]} errors' for p in pipelines))
        raise SystemExit(next((p.fatal for p in pipelines if p.fatal), 'stopped'))
    except KeyboardInterrupt:
        pass
    finally:
        log.info('Live metrics: %s', json.dumps(recorder.close(), indent=1))


if __name__ == '__main__':
    main()
