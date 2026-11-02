"""Before/after live-pipeline measurement on this Mac with a simulated network.

Runs the real backend (separate process), two simulated phones replaying a clip at
capture time, the real detector and worker, and (after-mode) the headless monitor.
Every HTTP request gets an injected round-trip delay (default: 270 ms measured to the
Render service) plus jitter. Phone clocks are the Mac clock, so ages have no skew.

This isolates pipeline architecture; it is NOT a measurement of the real Render
service, phone encoding, mobile radio variance or bandwidth limits.

  before-deployed : snapshot commit backend/worker/detector + deployed 1 FPS serial page
  before-overlay  : snapshot commit backend/worker/detector + 5 FPS serial page (429 drops)
  after           : this branch, pipelined page at 5 FPS
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = '588f596'  # performance overlay as it was running before this work
OLD_FILES = ['cloud_backend/__init__.py', 'cloud_backend/app.py', 'cloud_backend/worker.py', 'ai_detector.py',
             'ml_adapter.py', 'performance_tools.py']
KEY = 'local-ab-test-key-with-at-least-32-characters'


def prepare_old(dest):
    for f in OLD_FILES:
        out = dest / f; out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{SNAPSHOT}:{f}']))
    for w in ('best.pt', 'yolov8n.pt'):
        (dest / w).symlink_to(ROOT / w)


def start_backend(code_dir, data_dir, port):
    env = dict(os.environ, PPE_API_KEY=KEY, PPE_DATA_DIR=str(data_dir), PPE_CAMERA_IDS='CAM_01,CAM_02', PYTHONPATH=str(code_dir))
    env.pop('PPE_HANDWASH_ROI', None)
    proc = subprocess.Popen([str(ROOT / '.venv-cloud/bin/python'), '-m', 'uvicorn', 'cloud_backend.app:create_app', '--factory',
                             '--host', '127.0.0.1', '--port', str(port), '--log-level', 'warning'], cwd=code_dir, env=env)
    import httpx
    for _ in range(100):
        try:
            httpx.get(f'http://127.0.0.1:{port}/healthz', timeout=1); return proc
        except httpx.HTTPError:
            time.sleep(0.1)
    proc.kill(); raise RuntimeError('backend did not start')


def clip_frames(path, offset_s):
    import cv2
    cap = cv2.VideoCapture(str(path)); fps = cap.get(cv2.CAP_PROP_FPS) or 30
    jpegs = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        jpegs.append(cv2.imencode('.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes())
    return lambda t: jpegs[int((t + offset_s) * fps) % len(jpegs)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', choices=['before-deployed', 'before-overlay', 'after'], required=True)
    ap.add_argument('--duration', type=float, default=60)
    ap.add_argument('--rtt-ms', type=float, default=270)
    ap.add_argument('--jitter-ms', type=float, default=60)
    ap.add_argument('--fps', type=float, default=5)
    ap.add_argument('--clip', default=str(ROOT / 'review_video.mp4'))
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    args.output = str(Path(args.output).resolve())
    old = args.mode.startswith('before')
    work = Path(tempfile.mkdtemp(prefix='ppe_ab_'))
    code = work / 'code' if old else ROOT
    if old:
        prepare_old(code)
    live = work / 'live'; live.mkdir()
    os.environ['PPE_MONITOR_DIR'] = str(live)
    sys.path.insert(0, str(code)); sys.path.append(str(ROOT / 'tests_cloud'))
    os.chdir(code)
    import httpx, psutil, torch
    from live_harness import DelayTransport, Phone, free_port
    import ai_detector
    from performance_tools import Recorder
    assert Path(ai_detector.__file__).resolve().parent == code.resolve(), 'wrong detector version imported'
    port = free_port(); url = f'http://127.0.0.1:{port}'
    backend = start_backend(code, work / 'data', port)
    monitors = []
    try:
        recorder = Recorder(work / 'live.jsonl')
        stop = threading.Event()
        serial = old
        fps = 1.0 if args.mode == 'before-deployed' else args.fps
        phones = [Phone(url, KEY, cam, clip_frames(args.clip, off), fps=fps, rtt_ms=args.rtt_ms, jitter_ms=args.jitter_ms,
                        mode='serial' if serial else 'pipelined', status_after_upload=args.mode == 'before-deployed', stop=stop)
                  for cam, off in (('CAM_01', 0.0), ('CAM_02', 3.5))]
        transport = lambda: DelayTransport(args.rtt_ms, args.jitter_ms)
        # Warm the models outside the measured window.
        import cv2, numpy as np
        warm = cv2.imencode('.jpg', np.zeros((360, 640, 3), np.uint8))[1].tobytes()
        for _ in range(3):
            ai_detector.detect(warm, 'WARM', 'warm')
        ai_detector._sessions.pop('WARM', None)
        if not old:
            with httpx.Client(base_url=url) as c:
                ai_detector.configure(c.get('/camera-config').json()['zone_config'])
            for mode in ('off', 'auto'):
                monitors.append(subprocess.Popen([sys.executable, str(ROOT / 'monitor.py'), '--headless', '--playout', mode,
                                                  '--duration', str(args.duration + 3), '--summary', str(work / f'monitor_{mode}.json')],
                                                 env=dict(os.environ), stdout=subprocess.DEVNULL))
        proc = psutil.Process(); proc.cpu_percent()
        for p in phones:
            p.start()
        started = time.monotonic()
        if old:
            from cloud_backend.worker import process_once
            from concurrent.futures import ThreadPoolExecutor
            def loop():
                with httpx.Client(base_url=url, headers={'X-API-Key': KEY}, timeout=30, transport=transport()) as client, \
                        ThreadPoolExecutor(max_workers=2) as pool:
                    while not stop.is_set():
                        try:
                            process_once(client, ai_detector.detect, pool, recorder)
                        except Exception as e:  # mirror the old main loop: log and continue
                            print('cycle failed:', type(e).__name__, e)
                        time.sleep(.02)
            threading.Thread(target=loop, daemon=True).start()
        else:
            from cloud_backend.worker import run_pipelines
            run_pipelines(url, KEY, ai_detector.detect, ['CAM_01', 'CAM_02'], recorder, wait=5.0, stop=stop, transport=transport)
        time.sleep(args.duration)
        stop.set()
        cpu = proc.cpu_percent()
        for p in phones:
            p.join(5)
        time.sleep(1.0)
        recorder.close()
        # Summarise both modes with this branch's statistics code.
        import importlib.util
        spec = importlib.util.spec_from_file_location('new_tools', ROOT / 'performance_tools.py')
        tools = importlib.util.module_from_spec(spec); spec.loader.exec_module(tools)
        rows = [json.loads(line) for line in (work / 'live.jsonl').read_text().splitlines() if line.strip()]
        summary = tools.summarize(rows, time.monotonic() - started)
        mps = None
        if torch.backends.mps.is_available():
            mps = {'current_allocated_mib': torch.mps.current_allocated_memory() / 2**20,
                   'driver_allocated_mib': torch.mps.driver_allocated_memory() / 2**20}
        for m in monitors:
            m.wait(30)
        result = {'mode': args.mode, 'duration_s': args.duration, 'rtt_ms': args.rtt_ms, 'jitter_ms': args.jitter_ms,
                  'phone_target_fps': fps, 'phones': {p.camera: p.counts for p in phones},
                  'worker': summary, 'worker_process_cpu_percent': cpu, 'mps_memory': mps,
                  'monitor': {m: json.loads((work / f'monitor_{m}.json').read_text()) for m in ('off', 'auto')
                              if (work / f'monitor_{m}.json').exists()},
                  'scope': __doc__.strip().splitlines()[0] + ' Simulated network; same Mac for phones, backend and worker.'}
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=1))
        print(json.dumps({k: result[k] for k in ('mode', 'phones')}, indent=1))
        print(json.dumps(summary['per_camera'], indent=1)[:3000])
    finally:
        backend.terminate(); backend.wait(5)
        for m in monitors:
            if m.poll() is None:
                m.kill()
        shutil.rmtree(work, ignore_errors=True)


if __name__ == '__main__':
    main()
