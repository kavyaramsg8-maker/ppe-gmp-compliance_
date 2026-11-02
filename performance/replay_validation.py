"""Replay a clip through the real detector and real backend rules; print the decision timeline.

Frames are uploaded in real time (default 5 FPS) to a local backend process, so voting,
recovery and events behave as live. Use it to inspect compliant / missing / unclear
behaviour on representative clips; it does not measure accuracy against labels.

  python performance/replay_validation.py --clip review_video.mp4 --camera CAM_02 --output performance_run/replay_CAM_02.json
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path += [str(ROOT / 'performance'), str(ROOT / 'tests_cloud')]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--clip', default=str(ROOT / 'review_video.mp4'))
    ap.add_argument('--camera', default='CAM_02')
    ap.add_argument('--fps', type=float, default=5)
    ap.add_argument('--check-actions', action='store_true')
    ap.add_argument('--rotate', choices=['none', 'cw'], default='none', help='Simulate a portrait phone')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    args.output = str(Path(args.output).resolve())
    import cv2, httpx
    from live_ab import KEY, start_backend
    work = Path(tempfile.mkdtemp(prefix='ppe_replay_'))
    os.environ['PPE_MONITOR_DIR'] = str(work)
    if args.check_actions:
        os.environ['PPE_CHECK_ACTIONS'] = 'true'
    import ai_detector
    from live_harness import free_port
    port = free_port(); url = f'http://127.0.0.1:{port}'
    backend = start_backend(ROOT, work / 'data', port)
    timeline = []
    try:
        c = httpx.Client(base_url=url, headers={'X-API-Key': KEY}, timeout=10)
        ai_detector.configure(c.get('/camera-config').json()['zone_config'])
        sid = c.post(f'/v1/cameras/{args.camera}/sessions').json()['session_id']
        cap = cv2.VideoCapture(args.clip); clip_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        step = max(1, round(clip_fps / args.fps)); i = fid = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            i += 1
            if (i - 1) % step:
                continue
            if args.rotate == 'cw':
                f = cv2.rotate(f, cv2.ROTATE_90_CLOCKWISE)
            tick = time.monotonic(); captured = datetime.now(timezone.utc)
            data = cv2.imencode('.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()
            c.post(f'/v1/cameras/{args.camera}/frames', data={'session_id': sid, 'frame_id': fid, 'captured_at': captured.isoformat()},
                   files={'image': ('f.jpg', data, 'image/jpeg')}).raise_for_status()
            r = c.get(f'/v1/cameras/{args.camera}/frame', headers={'X-Claim-Frame': 'true'}); r.raise_for_status()
            people = ai_detector.detect(r.content, args.camera, sid, frame_id=fid, captured_at=captured.timestamp())
            out = c.post(f'/v1/cameras/{args.camera}/observations', json={'session_id': sid, 'frame_id': fid, 'persons': people}).json()
            timeline.append({'frame_id': fid, 'clip_frame': i - 1, 'status': out['status'],
                             'active': sorted(f"{v['track_id']}:{v['item']}" for v in out['active_violations']),
                             'persons': [{'id': p['track_id'], 'zone': p['zone'], 'ppe': p['ppe'], 'evaluated': p['evaluated'],
                                          'wash': p['handwash_status'], 'face': p['face_touch']} for p in out['persons']],
                             'evidence': [{'id': p['track_id'], 'ppe': p['associated_ppe'], 'reliable': p['ppe_reliable']} for p in people]})
            fid += 1
            time.sleep(max(0, 1 / args.fps - (time.monotonic() - tick)))
        events = c.get('/v1/events', params={'limit': 200}).json()['events']
    finally:
        backend.terminate(); backend.wait(5)
    result = {'clip': args.clip, 'camera': args.camera, 'fps': args.fps, 'rotate': args.rotate, 'timeline': timeline,
              'events': [{k: e[k] for k in ('type', 'track_id', 'missing_items', 'frame_id', 'zone')} for e in reversed(events)]}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=1))
    last = None
    for row in timeline:
        line = (row['status'], tuple(row['active']), tuple((p['id'], p['zone'], tuple(sorted(p['ppe'].items()))) for p in row['persons']))
        if line != last:
            print(f"f{row['frame_id']:>3} (clip {row['clip_frame']:>3}) {row['status']:<10} active={row['active']} "
                  + ' '.join(f"[ID{p['id']} {p['zone']} {','.join(k[0]+'='+v[:4] for k, v in p['ppe'].items())} wash={p['wash']}]" for p in row['persons']))
            last = line
    print('events:', [(e['type'], e['track_id'], e['missing_items'][0], e['frame_id']) for e in result['events']])


if __name__ == '__main__':
    main()
