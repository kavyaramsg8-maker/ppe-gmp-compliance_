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


def summarize(rows,elapsed_seconds):
    accepted=[r for r in rows if r.get('accepted',True)]
    memory=[r['rss_mib'] for r in rows if r.get('rss_mib') is not None]
    return {'attempted_frames':len(rows),'accepted_frames':len(accepted),
            'elapsed_seconds':elapsed_seconds,'accepted_fps':len(accepted)/elapsed_seconds if elapsed_seconds>0 else None,
            'per_camera_accepted_fps':{c:sum(r.get('camera_id')==c for r in accepted)/elapsed_seconds for c in sorted({r.get('camera_id','unknown') for r in rows})} if elapsed_seconds>0 else {},
            'detector_ms':describe([r.get('detector_ms') for r in accepted]),
            'cycle_ms':describe([r.get('cycle_ms') for r in accepted]),
            'capture_age_ms':describe([r.get('capture_age_ms') for r in accepted]),
            'inference_lock_wait_ms':describe([r.get('lock_wait_ms') for r in rows]),
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
