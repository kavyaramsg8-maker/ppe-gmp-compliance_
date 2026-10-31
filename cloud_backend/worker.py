"""Run Aravind's detector on the cloud host; no laptop webcam or display calls."""
import argparse
import importlib
import logging
import os
import time
import json
from pathlib import Path
from hashlib import sha256
import httpx

log = logging.getLogger(__name__)


def process_once(client, detector):
    response = client.get('/v1/cameras')
    response.raise_for_status()
    processed = 0
    for camera in response.json()['cameras']:
        if camera['latest_frame_id'] is None or camera['latest_frame_id'] == camera['processed_frame_id']:
            continue
        camera_id = camera['camera_id']
        frame = client.get(f'/v1/cameras/{camera_id}/frame', headers={'X-Claim-Frame': 'true'})
        if frame.status_code in (404, 410):
            continue
        frame.raise_for_status()
        session_id = frame.headers['X-Session-ID']
        persons = detector(frame.content, camera_id, session_id)
        result = client.post(f'/v1/cameras/{camera_id}/observations', json={
            'session_id': session_id, 'frame_id': int(frame.headers['X-Frame-ID']), 'persons': persons})
        if result.status_code == 409:
            log.info('Dropped expired or superseded result for %s', camera_id)
            continue
        result.raise_for_status()
        # Monitor consumes the actual server verdict, matched to this JPEG.
        folder=Path(os.environ.get('PPE_MONITOR_DIR',str(Path(__file__).resolve().parent.parent)))
        folder.mkdir(parents=True,exist_ok=True)
        name=''.join(ch for ch in camera_id if ch.isalnum() or ch in '_-')
        path=folder/f'live_{name}.decision.json'; tmp=path.with_suffix('.tmp')
        value=result.json(); value['source_sha256']=sha256(frame.content).hexdigest()
        tmp.write_text(json.dumps(value)); os.replace(tmp,path)
        processed += 1
    return processed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default=os.environ.get('PPE_BACKEND_URL', 'http://127.0.0.1:8000'))
    parser.add_argument('--detector', default=os.environ.get('PPE_DETECTOR'))
    args = parser.parse_args()
    key = os.environ.get('PPE_API_KEY')
    if not key or not args.detector or ':' not in args.detector:
        parser.error('Set PPE_API_KEY and --detector module:function (or PPE_DETECTOR)')
    module, function = args.detector.split(':', 1)
    detector = getattr(importlib.import_module(module), function)
    if not callable(detector):
        parser.error('Detector must be callable')
    logging.basicConfig(level=logging.INFO)
    with httpx.Client(base_url=args.url.rstrip('/'), headers={'X-API-Key': key}, timeout=30) as client:
        config=client.get('/camera-config')
        config.raise_for_status()
        config=config.json()
        if 'handwash_roi' in config:
            os.environ['PPE_HANDWASH_ROI']=json.dumps(config['handwash_roi'])
        log.info('Server layout: %s; cameras: %s',config.get('zone_mode','production'),config.get('cameras'))
        if args.detector=='ai_detector:detect' and config.get('zone_mode')!='cameras':
            parser.error('This detector uses camera-based zones. Set PPE_ZONE_MODE=cameras on the backend, redeploy and restart.')
        while True:
            try:
                process_once(client, detector)
            except (httpx.HTTPError, RuntimeError, ValueError, KeyError, TypeError):
                # Failed inference never sends a compliant fallback. Other polling cycles retry.
                log.exception('Worker cycle failed; no substitute detection submitted')
            time.sleep(.25)


if __name__ == '__main__':
    main()

