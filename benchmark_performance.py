"""Run in a fresh process for each variant, using the same local clip and settings."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import tempfile
import threading
import time
from performance_tools import describe,rss_mib,atomic_bytes

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant',choices=['baseline','optimized'],required=True)
    p.add_argument('--source',required=True,help='Same local test video in both runs')
    p.add_argument('--device',default='cpu',help='cpu, mps, or cuda:0; keep matched for comparison')
    p.add_argument('--frames',type=int,default=60)
    p.add_argument('--warmup',type=int,default=5)
    p.add_argument('--camera',choices=['CAM_01','CAM_02'],default='CAM_01')
    p.add_argument('--ppe-imgsz',type=int,default=640)
    p.add_argument('--person-imgsz',type=int,default=416)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    if args.frames<1 or args.warmup<0:p.error('frames must be positive; warmup nonnegative')
    if args.variant=='baseline' and args.ppe_imgsz!=640:p.error('Baseline PPE input is 640')
    os.environ.update(PPE_DEVICE=args.device,PPE_IMGSZ=str(args.ppe_imgsz),PPE_PERSON_IMGSZ=str(args.person_imgsz))
    import cv2
    import torch
    from ultralytics import YOLO
    source=str(Path(args.source).resolve());capture=cv2.VideoCapture(source)
    if not capture.isOpened():raise RuntimeError(f'Cannot open clip: {source}')
    frames=[]
    for _ in range(args.frames+args.warmup):
        ok,image=capture.read()
        if not ok:break
        h,w=image.shape[:2];scale=min(640/w,360/h)
        import numpy as np
        canvas=np.zeros((360,640,3),np.uint8);small=cv2.resize(image,(round(w*scale),round(h*scale)))
        y=(360-small.shape[0])//2;x=(640-small.shape[1])//2
        canvas[y:y+small.shape[0],x:x+small.shape[1]]=small
        ok,encoded=cv2.imencode('.jpg',canvas,[cv2.IMWRITE_JPEG_QUALITY,80]);frames.append(encoded.tobytes())
    capture.release()
    if len(frames)<args.frames+args.warmup:raise ValueError(f'Clip needs {args.frames+args.warmup} frames, has {len(frames)}; use --frames lower')
    # Input decode preparation is excluded from detector timing in BOTH variants.
    stop=threading.Event();samples=[]
    def sample():
        while not stop.is_set():
            value=rss_mib()
            if value is not None:samples.append(value)
            stop.wait(.02)
    thread=threading.Thread(target=sample,daemon=True);thread.start()
    def sync():
        if args.device=='mps':torch.mps.synchronize()
        elif args.device.startswith('cuda'):torch.cuda.synchronize()
    with tempfile.TemporaryDirectory(prefix='ppe_benchmark_') as folder:
        os.environ['PPE_MONITOR_DIR']=folder
        loaded=time.perf_counter()
        if args.variant=='optimized':
            import ai_detector as module
        else:
            # Preserve original algorithm; only pin device/weight paths and redirect preview I/O.
            import types
            module=types.ModuleType('baseline_detector');module.__file__=str(ROOT/'performance/baseline_detector.py')
            text=(ROOT/'performance/baseline_detector.py').read_text()
            text=text.replace("YOLO('best.pt')",f'YOLO({str(ROOT/"best.pt")!r})').replace("YOLO('yolov8n.pt')",f'YOLO({str(ROOT/"yolov8n.pt")!r})')
            text=text.replace('verbose=False','verbose=False, device='+repr(args.device))
            exec(compile(text,module.__file__,'exec'),module.__dict__)
        load_ms=(time.perf_counter()-loaded)*1000
        old_cwd=os.getcwd();os.chdir(folder)
        times=[];counts=[];stages=[]
        try:
            for i,frame in enumerate(frames):
                sync();start=time.perf_counter();result=module.detect(frame,args.camera,'benchmark-session');sync()
                elapsed=(time.perf_counter()-start)*1000
                if i>=args.warmup:
                    times.append(elapsed);counts.append(len(result))
                    if hasattr(module,'get_metrics'):stages.append(module.get_metrics(args.camera))
        finally:os.chdir(old_cwd);stop.set();thread.join()
        # Profiling may copy model weights; exclude that extra allocation from measured RSS.
        models=[('ppe',module.ppe_model,args.ppe_imgsz)]
        if args.variant=='baseline':models.append(('person',module.person_model,640))
        else:models.append(('person',module._sessions[args.camera]['person_model'],args.person_imgsz))
        profiles=[]
        for name,model,size in models:
            entry={'name':name,'parameters':sum(x.numel() for x in model.model.parameters()),'nominal_imgsz':size}
            try:
                from ultralytics.utils.torch_utils import get_flops
                # Letterboxed input, batch=1. Confirm actual predictor shape from model log if changing aspect ratio.
                height=math.ceil((360/640*size)/32)*32
                value=get_flops(model.model,imgsz=[height,size])
                entry.update(input_shape=[1,3,height,size],gflops=float(value) if value>0 else None)
                if value<=0:entry['flops_note']='Profiler unavailable/unsupported; install ultralytics-thop and rerun'
            except Exception as e:entry.update(gflops=None,flops_note=str(e))
            profiles.append(entry)
        result={'variant':args.variant,'hardware':platform.platform(),'processor':platform.processor(),'torch':torch.__version__,'device':args.device,
                'source':source,'camera':args.camera,'warmup_frames':args.warmup,'measured_frames':len(times),'model_load_ms':load_ms,
                'detector_latency_ms':describe(times),'detector_throughput_fps':1000*len(times)/sum(times),
                'peak_sampled_process_rss_mib':max(samples) if samples else None,'detected_people_per_frame':describe(counts),
                'models':profiles,'unique_yolo_parameters':sum(x['parameters'] for x in profiles),
                'stage_latency_ms':{k:describe([r.get(k) for r in stages]) for k in ['ppe_ms','person_ms','pose_ms']},
                'scope':'Single-camera sequential detector incl JPEG decode, tracking, pose, annotation and preview writes. No cloud/network/ESP32. YOLO FLOPs exclude MediaPipe, tracking, I/O and backend. RSS is process host memory, not GPU memory. Preloaded encoded input frames included in RSS. No accuracy claim.'}
        atomic_bytes(args.output,json.dumps(result,indent=2).encode());print(json.dumps(result,indent=2))

if __name__=='__main__':main()
