"""Inference worker with overlapping camera network I/O and measured throughput."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import importlib
import json
import logging
import os
from pathlib import Path
import threading
import time
import httpx
from performance_tools import Recorder,atomic_bytes,rss_mib

log=logging.getLogger(__name__)
_infer_lock=threading.Lock()


def _process_camera(client,detector,camera,recorder=None):
    if camera['latest_frame_id'] is None or camera['latest_frame_id']==camera['processed_frame_id']:return 0
    camera_id=camera['camera_id'];start=time.perf_counter()
    frame=client.get(f'/v1/cameras/{camera_id}/frame',headers={'X-Claim-Frame':'true'})
    if frame.status_code in (404,410):return 0
    frame.raise_for_status();session_id=frame.headers['X-Session-ID']
    fetched=time.perf_counter()
    try:
        with _infer_lock:
            infer_start=time.perf_counter();persons=detector(frame.content,camera_id,session_id)
            infer_end=time.perf_counter()
        result=client.post(f'/v1/cameras/{camera_id}/observations',json={'session_id':session_id,'frame_id':int(frame.headers['X-Frame-ID']),'persons':persons})
        end=time.perf_counter()
        if result.status_code!=409:result.raise_for_status()
        accepted=result.status_code!=409
        age=None
        try:age=(datetime.now(timezone.utc)-datetime.fromisoformat(frame.headers['X-Captured-At'])).total_seconds()*1000
        except (KeyError,ValueError):pass
        row={'camera_id':camera_id,'frame_id':int(frame.headers['X-Frame-ID']),'accepted':accepted,
             'detector_ms':(infer_end-infer_start)*1000,'download_ms':(fetched-start)*1000,
             'lock_wait_ms':(infer_start-fetched)*1000,'submit_ms':(end-infer_end)*1000,
             'cycle_ms':(end-start)*1000,'capture_age_ms':age,'rss_mib':rss_mib(),'recorded_at':datetime.now(timezone.utc).isoformat()}
        if recorder:recorder.add(row)
        if not accepted:
            log.warning('Dropped expired/superseded result for %s',camera_id);return 0
        # No API key or image data in this status file.
        safe=''.join(c for c in camera_id if c.isalnum() or c in '_-')
        folder=Path(os.environ.get('PPE_MONITOR_DIR',str(Path(__file__).resolve().parent.parent)))
        value=result.json();value['worker_metrics']=row
        atomic_bytes(folder/f'live_{safe}.status.json',json.dumps(value).encode())
        return 1
    except Exception:
        if recorder:recorder.add({'camera_id':camera_id,'accepted':False,'error':True,'cycle_ms':(time.perf_counter()-start)*1000,'rss_mib':rss_mib()})
        raise


def process_once(client,detector,executor=None,recorder=None):
    response=client.get('/v1/cameras');response.raise_for_status()
    cameras=response.json()['cameras']
    if executor is None:return sum(_process_camera(client,detector,c,recorder) for c in cameras)
    futures=[executor.submit(_process_camera,client,detector,c,recorder) for c in cameras]
    count=0;error=None
    for f in futures:
        try:count+=f.result()
        except Exception as e:error=e
    if error:raise error
    return count


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default=os.environ.get('PPE_BACKEND_URL','http://127.0.0.1:8000'))
    parser.add_argument('--detector',default=os.environ.get('PPE_DETECTOR'))
    parser.add_argument('--metrics',default='performance_run/live.jsonl')
    parser.add_argument('--network-workers',type=int,choices=[1,2],default=2)
    args=parser.parse_args();key=os.environ.get('PPE_API_KEY')
    if not key or not args.detector or ':' not in args.detector:parser.error('Set PPE_API_KEY and --detector module:function')
    module,function=args.detector.split(':',1);detector=getattr(importlib.import_module(module),function)
    if not callable(detector):parser.error('Detector must be callable')
    logging.basicConfig(level=logging.INFO);logging.getLogger('httpx').setLevel(logging.WARNING)
    recorder=Recorder(args.metrics)
    try:
        with httpx.Client(base_url=args.url.rstrip('/'),headers={'X-API-Key':key},timeout=30) as client,ThreadPoolExecutor(max_workers=args.network_workers) as pool:
            while True:
                try:process_once(client,detector,pool,recorder)
                except (httpx.HTTPError,RuntimeError,ValueError,KeyError,TypeError):log.exception('Worker cycle failed; no compliant substitute')
                time.sleep(.02)
    except KeyboardInterrupt:pass
    finally:
        log.info('Live metrics: %s',recorder.close())


if __name__=='__main__':main()
