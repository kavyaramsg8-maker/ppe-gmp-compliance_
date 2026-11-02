"""Shared helpers: run the real backend in-process and simulate phones over a slow network."""
from datetime import datetime, timezone
import random
import socket
import threading
import time

import httpx


class DelayTransport(httpx.HTTPTransport):
    """Adds a round-trip delay (plus optional uniform jitter) to every request."""

    def __init__(self, rtt_ms=0, jitter_ms=0, **kw):
        super().__init__(**kw)
        self.rtt, self.jitter = rtt_ms / 1000, jitter_ms / 1000

    def handle_request(self, request):
        delay = self.rtt + random.uniform(0, self.jitter)
        time.sleep(delay / 2)
        response = super().handle_request(request)
        response.read()
        time.sleep(delay / 2)
        return response


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


class Backend:
    def __init__(self, app, port=None):
        import uvicorn
        self.port = port or free_port()
        self.url = f'http://127.0.0.1:{self.port}'
        self.server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=self.port, log_level='warning'))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self):
        self.thread.start()
        for _ in range(200):
            try:
                httpx.get(self.url + '/healthz', timeout=1); return self
            except httpx.HTTPError:
                time.sleep(0.05)
        raise RuntimeError('backend did not start')

    def __exit__(self, *exc):
        self.server.should_exit = True
        self.thread.join(5)


class Phone(threading.Thread):
    """Simulated phone page.

    mode='pipelined': current camera.html (fixed cadence, <= max_in_flight uploads, skips when busy).
    mode='serial': earlier page (await each upload, then sleep the rest of the interval).
    status_after_upload: the deployed main-branch page also awaited a status GET after each upload.
    """

    def __init__(self, url, key, camera, frames, fps=5.0, rtt_ms=0, mode='pipelined', max_in_flight=2,
                 status_after_upload=False, stop=None, jitter_ms=0):
        super().__init__(daemon=True)
        self.url, self.key, self.camera, self.frames, self.fps = url, key, camera, frames, fps
        self.rtt_ms, self.mode, self.max_in_flight, self.status_after = rtt_ms, mode, max_in_flight, status_after_upload
        self.jitter_ms = jitter_ms
        self.stop = stop or threading.Event()
        self.counts = {'sent': 0, 'accepted': 0, 'busy_429': 0, 'rejected_409': 0, 'skipped': 0, 'errors': 0}
        self.lock = threading.Lock()
        self.in_flight = 0

    def client(self):
        return httpx.Client(base_url=self.url, headers={'X-API-Key': self.key}, timeout=10, transport=DelayTransport(self.rtt_ms, self.jitter_ms))

    def session(self, client):
        self.sid = client.post(f'/v1/cameras/{self.camera}/sessions').json()['session_id']
        self.frame_id = 0

    def frame_bytes(self, elapsed):
        return self.frames(elapsed)

    def upload(self, client, data, fid, sid):
        with self.lock:
            self.counts['sent'] += 1
        try:
            r = client.post(f'/v1/cameras/{self.camera}/frames', data={'session_id': sid, 'frame_id': fid,
                            'captured_at': datetime.now(timezone.utc).isoformat()}, files={'image': ('f.jpg', data, 'image/jpeg')})
            key = 'accepted' if r.status_code == 201 else 'busy_429' if r.status_code == 429 else 'rejected_409' if r.status_code == 409 else 'errors'
        except httpx.HTTPError:
            key = 'errors'
        with self.lock:
            self.counts[key] += 1
        if self.status_after:
            try:
                client.get(f'/v1/cameras/{self.camera}')
            except httpx.HTTPError:
                pass

    def run(self):
        started = time.monotonic()
        with self.client() as client:
            self.session(client)
        clients = [self.client() for _ in range(self.max_in_flight)]
        free = list(clients)
        interval = 1 / self.fps
        next_tick = time.monotonic()
        try:
            while not self.stop.is_set():
                tick_start = time.monotonic()
                # Read the frame at capture time; encoding happens before upload, as on the phone.
                data = self.frame_bytes(tick_start - started)
                fid = self.frame_id; self.frame_id += 1
                if self.mode == 'serial':
                    self.upload(clients[0], data, fid, self.sid)
                    time.sleep(max(0.01, interval - (time.monotonic() - tick_start)))
                    continue
                with self.lock:
                    busy = self.in_flight >= self.max_in_flight
                    if not busy:
                        self.in_flight += 1; c = free.pop()
                if busy:
                    self.frame_id -= 1
                    with self.lock:
                        self.counts['skipped'] += 1
                else:
                    def go(c=c, data=data, fid=fid):
                        try:
                            self.upload(c, data, fid, self.sid)
                        finally:
                            with self.lock:
                                self.in_flight -= 1; free.append(c)
                    threading.Thread(target=go, daemon=True).start()
                next_tick += interval
                time.sleep(max(0.0, next_tick - time.monotonic()))
        finally:
            time.sleep(0.2)
            for c in clients:
                c.close()
