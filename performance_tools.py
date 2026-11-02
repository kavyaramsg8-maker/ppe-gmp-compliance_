"""Timing, atomic preview writes, and transparent performance summaries."""
from pathlib import Path
from datetime import datetime,timezone
import json
import math
import os
import statistics
import tempfile
import threading
import time


def atomic_bytes(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:f.write(data)
        os.replace(temp,path)
    finally:
        try:os.unlink(temp)
        except FileNotFoundError:pass


def rss_mib():
    try:
        import psutil
        return psutil.Process().memory_info().rss/1024**2
    except ImportError:return None


def describe(values):
    a=sorted(float(x) for x in values if x is not None and math.isfinite(float(x)))
    if not a:return {'mean':None,'p50':None,'p95':None,'max':None}
    def percentile(q):
        i=(len(a)-1)*q;lo=int(i);hi=min(lo+1,len(a)-1)
        return a[lo]+(a[hi]-a[lo])*(i-lo)
    return {'mean':statistics.mean(a),'p50':percentile(.5),'p95':percentile(.95),'max':a[-1]}


def _first(row,*keys):
    return next((row[k] for k in keys if row.get(k) is not None),None)


def camera_summary(rows):
    """Steadiness of accepted results for one camera, from their recorded_at times."""
    accepted=[r for r in rows if r.get('accepted',True)]
    times=sorted(datetime.fromisoformat(r['recorded_at']).timestamp() for r in accepted if r.get('recorded_at'))
    gaps=[(b-a)*1000 for a,b in zip(times,times[1:])]
    interval=describe(gaps)
    spread=statistics.pstdev(gaps) if len(gaps)>1 else None
    ids=sorted(r['frame_id'] for r in accepted if isinstance(r.get('frame_id'),int))
    return {'accepted_frames':len(accepted),'rejected_or_failed':len(rows)-len(accepted),
            'accepted_fps':(len(times)-1)/(times[-1]-times[0]) if len(times)>1 and times[-1]>times[0] else None,
            'interval_ms':interval,'interval_cv':spread/interval['mean'] if spread is not None and interval['mean'] else None,
            # Uploaded frame IDs never processed (phone uploads faster than the pipeline, or network loss).
            'uploaded_frames_skipped':sum(b-a-1 for a,b in zip(ids,ids[1:]) if b>a),
            'detector_ms':describe([r.get('detector_ms') for r in accepted]),
            'fetch_ms':describe([_first(r,'fetch_ms','download_ms') for r in accepted]),
            'submit_ms':describe([r.get('submit_ms') for r in accepted]),
            'gate_wait_ms':describe([_first(r,'gate_wait_ms','lock_wait_ms') for r in accepted]),
            'server_wait_ms':describe([r.get('server_wait_ms') for r in accepted]),
            'age_at_fetch_ms':describe([r.get('age_at_fetch_ms') for r in accepted]),
            # Worker-side clock after the decision returns: comparable across old/new workers.
            'capture_to_decision_ms':describe([_first(r,'age_at_decision_ms','capture_age_ms') for r in accepted]),
            'server_capture_to_decision_ms':describe([r.get('server_capture_to_decision_ms') for r in accepted]),
            'dropped_before_inference':max([r.get('dropped_before_inference',0) for r in rows] or [0]),
            'dropped_before_submit':max([r.get('dropped_before_submit',0) for r in rows] or [0])}


def summarize(rows,elapsed_seconds):
    accepted=[r for r in rows if r.get('accepted',True)]
    memory=[r['rss_mib'] for r in rows if r.get('rss_mib') is not None]
    cameras=sorted({r.get('camera_id','unknown') for r in rows})
    return {'attempted_frames':len(rows),'accepted_frames':len(accepted),
            'elapsed_seconds':elapsed_seconds,'accepted_fps':len(accepted)/elapsed_seconds if elapsed_seconds>0 else None,
            'per_camera_accepted_fps':{c:sum(r.get('camera_id')==c for r in accepted)/elapsed_seconds for c in cameras} if elapsed_seconds>0 else {},
            'detector_ms':describe([r.get('detector_ms') for r in accepted]),
            'cycle_ms':describe([_first(r,'local_cycle_ms','cycle_ms') for r in accepted]),
            'capture_age_ms':describe([_first(r,'age_at_decision_ms','capture_age_ms') for r in accepted]),
            'inference_lock_wait_ms':describe([_first(r,'gate_wait_ms','lock_wait_ms') for r in rows]),
            'per_camera':{c:camera_summary([r for r in rows if r.get('camera_id','unknown')==c]) for c in cameras},
            'peak_rss_mib':max(memory) if memory else None}


class Recorder:
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.rows=[];self.started=time.perf_counter();self.lock=threading.Lock()
        self.file=self.path.open('w',encoding='utf8')
    def add(self,row):
        with self.lock:
            self.rows.append(row)
            self.file.write(json.dumps(row)+'\n');self.file.flush()
    def close(self):
        with self.lock:
            self.file.close();summary=summarize(self.rows,time.perf_counter()-self.started)
            atomic_bytes(self.path.with_suffix('.summary.json'),json.dumps(summary,indent=2).encode())
            return summary
