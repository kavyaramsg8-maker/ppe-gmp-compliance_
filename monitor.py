"""Display processed frames per camera with honest timing; never show display refresh as inference FPS.

Only frames the detector actually processed are shown, each paired (by JPEG hash) with
its frame ID and capture time. The optional playout buffer re-times those real frames
to their capture cadence, so irregular network arrival does not look like stutter; it
adds a stated delay and never invents, interpolates or repeats-as-new any frame.
"""
import argparse
from collections import deque
from hashlib import sha256
import json
import os
from pathlib import Path
import statistics
import time

import cv2
import numpy as np

TILE_W, TILE_H, HEADER = 640, 480, 58


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    i = (len(values) - 1) * q
    lo = int(i); hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (i - lo)


class CameraFeed:
    """Reads processed frames for one camera and decides what to show when."""

    def __init__(self, root, camera_id, playout, max_extra=1.0):
        self.camera_id = camera_id
        self.jpg = root / f'live_{camera_id}.jpg'
        self.meta = root / f'live_{camera_id}.frame.json'
        self.status = root / f'live_{camera_id}.status.json'
        self.playout, self.max_extra = playout, max_extra
        self.stamp = None
        self.buffer = deque(maxlen=90)
        self.arrivals = deque(maxlen=60)
        self.latency = deque(maxlen=60)      # arrival − capture (includes phone/Mac clock offset)
        self.shown = None
        self.shown_log = []                   # (display time, frame_id) for the summary
        self.mismatches = 0

    def poll(self):
        try:
            stamp = self.jpg.stat().st_mtime_ns
        except FileNotFoundError:
            return
        if stamp == self.stamp:
            return
        data = self.jpg.read_bytes()
        try:
            meta = json.loads(self.meta.read_text())
        except (OSError, ValueError):
            meta = {}
        if meta.get('jpeg_sha256') and meta['jpeg_sha256'] != sha256(data).hexdigest():
            self.mismatches += 1  # image and metadata from different frames: retry next tick
            return
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return
        self.stamp = stamp
        now = time.time()
        if self.buffer and meta.get('session_id') != self.buffer[-1]['session_id']:
            self.buffer.clear(); self.latency.clear()  # a new phone session restarts timing
        entry = {'frame_id': meta.get('frame_id'), 'session_id': meta.get('session_id'),
                 'captured': meta.get('captured_at'), 'arrival': now, 'image': image}
        self.buffer.append(entry); self.arrivals.append(now)
        if entry['captured']:
            self.latency.append(now - entry['captured'])

    def delay(self):
        """Display deadline offset after capture, or None for 'show newest'."""
        if self.playout is None or len(self.latency) < 5:
            return None
        base = percentile(self.latency, 0.5)
        if self.playout == 'auto':
            target = percentile(self.latency, 0.9) + 0.05
        else:
            target = base + self.playout
        return min(target, base + self.max_extra)

    def choose(self, now):
        if not self.buffer:
            return None
        d = self.delay()
        if d is None or self.buffer[-1]['captured'] is None:
            pick = self.buffer[-1]
        else:
            ready = [e for e in self.buffer if e['captured'] is not None and e['captured'] + d <= now]
            pick = ready[-1] if ready else (self.shown or self.buffer[0])
            # Frames older than the one now shown are no longer needed.
            while self.buffer and self.buffer[0] is not pick and self.buffer[0]['arrival'] <= pick['arrival']:
                self.buffer.popleft()
        if pick is not self.shown:
            self.shown = pick
            self.shown_log.append((now, pick['frame_id']))
        return pick

    def stats(self):
        gaps = [(b - a) * 1000 for a, b in zip(self.arrivals, list(self.arrivals)[1:])]
        span = self.arrivals[-1] - self.arrivals[0] if len(self.arrivals) > 1 else 0
        return {'fps': (len(self.arrivals) - 1) / span if span > 0 else 0.0,
                'p50': percentile(gaps, 0.5), 'p95': percentile(gaps, 0.95),
                'extra_ms': (self.delay() - percentile(self.latency, 0.5)) * 1000 if self.delay() is not None else 0.0}


def backend_verdict(path, stale_seconds, now):
    try:
        value = json.loads(path.read_text()); age = now - path.stat().st_mtime
    except (OSError, ValueError):
        return 'UNKNOWN (no backend decision yet)', None
    if age > stale_seconds or value.get('stale'):
        return 'UNKNOWN (backend decision stale)', None
    return value.get('status', 'UNKNOWN'), value.get('processed_frame_id')


def render_tile(feed, pick, stale_seconds, now):
    tile = np.full((TILE_H + HEADER, TILE_W, 3), 18, np.uint8)
    if pick is not None:
        img = pick['image']; h, w = img.shape[:2]; s = min(TILE_W / w, TILE_H / h)
        img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))))
        y, x = HEADER + (TILE_H - img.shape[0]) // 2, (TILE_W - img.shape[1]) // 2
        tile[y:y + img.shape[0], x:x + img.shape[1]] = img
    st = feed.stats()
    newest = feed.arrivals[-1] if feed.arrivals else None
    stale = newest is None or now - newest > stale_seconds
    verdict, verdict_frame = backend_verdict(feed.status, stale_seconds, now)
    if pick is None:
        line1 = f'{feed.camera_id} | waiting for processed frames'
    else:
        age = f"{now - pick['captured']:.2f}s" if pick['captured'] else 'n/a'
        p95 = f"{st['p95']:.0f}" if st['p95'] is not None else '-'
        line1 = (f"{feed.camera_id} | processed {st['fps']:.1f} FPS | gap p95 {p95} ms | "
                 f"frame {pick['frame_id']} | capture->screen {age}")
    line2 = f'Backend: {verdict}' + (f' (frame {verdict_frame})' if verdict_frame is not None else '')
    if st['extra_ms'] > 0:
        line2 += f" | smoothing buffer +{st['extra_ms']:.0f} ms"
    if stale:
        tile[HEADER:] = (tile[HEADER:] * 0.35).astype(np.uint8)
        cv2.putText(tile, 'NO FRESH PROCESSED FRAMES - UNKNOWN', (40, HEADER + TILE_H // 2), cv2.FONT_HERSHEY_SIMPLEX, .8, (60, 60, 230), 2)
    cv2.putText(tile, line1, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .47, (255, 255, 255), 1)
    colour = {'VIOLATION': (60, 60, 230), 'COMPLIANT': (70, 190, 70), 'MONITORING': (220, 170, 70)}.get(verdict, (0, 200, 255))
    cv2.putText(tile, line2, (8, 46), cv2.FONT_HERSHEY_SIMPLEX, .5, colour, 1)
    return tile


def summary(feeds):
    out = {}
    for f in feeds:
        times = [t for t, _ in f.shown_log]
        gaps = [(b - a) * 1000 for a, b in zip(times, times[1:])]
        arrivals = list(f.arrivals)
        a_gaps = [(b - a) * 1000 for a, b in zip(arrivals, arrivals[1:])]
        out[f.camera_id] = {
            'frames_shown': len(times),
            'shown_interval_ms': {'p50': percentile(gaps, .5), 'p95': percentile(gaps, .95),
                                  'cv': statistics.pstdev(gaps) / statistics.mean(gaps) if len(gaps) > 1 else None},
            'arrival_interval_ms_last60': {'p50': percentile(a_gaps, .5), 'p95': percentile(a_gaps, .95),
                                           'cv': statistics.pstdev(a_gaps) / statistics.mean(a_gaps) if len(a_gaps) > 1 else None},
            'smoothing_extra_ms': f.stats()['extra_ms'], 'metadata_mismatch_retries': f.mismatches}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cameras', default='CAM_01,CAM_02')
    p.add_argument('--record', help='Optional side-by-side MP4 recording (finalised when you press q)')
    p.add_argument('--stale-seconds', type=float, default=15)
    p.add_argument('--playout', default='auto',
                   help="'auto' (re-time to capture cadence, delay ~p90 latency), 'off' (newest immediately), or extra seconds e.g. 0.3")
    p.add_argument('--headless', action='store_true', help='No window; for measurement runs')
    p.add_argument('--duration', type=float, help='Stop after this many seconds')
    p.add_argument('--summary', default='performance_run/monitor_summary.json')
    args = p.parse_args()
    root = Path(os.environ.get('PPE_MONITOR_DIR', str(Path(__file__).resolve().parent)))
    playout = None if args.playout == 'off' else 'auto' if args.playout == 'auto' else float(args.playout)
    feeds = [CameraFeed(root, c.strip(), playout) for c in args.cameras.split(',') if c.strip()]
    writer = None; next_record = 0; started = time.monotonic()
    print('Processed-frame monitor. Click the image window and press q to stop.')
    try:
        while args.duration is None or time.monotonic() - started < args.duration:
            now = time.time()
            for f in feeds:
                f.poll()
            image = np.hstack([render_tile(f, f.choose(now), args.stale_seconds, now) for f in feeds])
            if args.record and time.monotonic() >= next_record:
                if writer is None:
                    writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*'mp4v'), 15, (image.shape[1], image.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError('Could not open recording output')
                writer.write(image); next_record = time.monotonic() + 1 / 15
            if args.headless:
                time.sleep(0.01)
                continue
            cv2.imshow('PPE processed monitor', image)
            if cv2.waitKey(10) & 255 in (ord('q'), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        if writer:
            writer.release(); print(f'Saved {args.record} (15 FPS container; processed FPS is printed on the video).')
        result = summary(feeds)
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(json.dumps(result, indent=1))
        print(json.dumps(result, indent=1))
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
