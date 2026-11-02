"""Display the latest processed frames; never pretend display refresh is inference FPS."""
import argparse
from collections import deque
import json
import os
from pathlib import Path
import time
import cv2
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cameras',default='CAM_01,CAM_02')
    p.add_argument('--record',help='Optional side-by-side MP4 recording')
    p.add_argument('--stale-seconds',type=float,default=15)
    args=p.parse_args();root=Path(os.environ.get('PPE_MONITOR_DIR',str(Path(__file__).resolve().parent)))
    cameras=args.cameras.split(',');state={c:{'stamp':None,'image':None,'updates':deque(maxlen=30)} for c in cameras}
    writer=None;next_record=0
    print('Processed-frame monitor. Click image window and press q to stop. Waiting for worker output...')
    try:
        while True:
            now=time.time();tiles=[]
            for camera in cameras:
                s=state[camera];path=root/f'live_{camera}.jpg'
                try:
                    stamp=path.stat().st_mtime_ns
                    if stamp!=s['stamp']:
                        frame=cv2.imread(str(path))
                        if frame is not None:
                            s.update(stamp=stamp,image=frame);s['updates'].append(time.monotonic())
                except FileNotFoundError:pass
                frame=s['image'].copy() if s['image'] is not None else np.zeros((360,640,3),np.uint8)
                frame=cv2.resize(frame,(640,360));age=now-path.stat().st_mtime if path.exists() else float('inf')
                updates=s['updates'];fps=(len(updates)-1)/(updates[-1]-updates[0]) if len(updates)>1 else 0
                if age>args.stale_seconds:fps=0
                cv2.rectangle(frame,(0,0),(640,60),(20,20,20),-1)
                label=f'{camera} | processed arrivals {fps:.1f} FPS | age {age:.1f}s' if age!=float('inf') else f'{camera} | waiting for processed image'
                cv2.putText(frame,label,(8,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
                verdict='UNKNOWN / STALE' if age>args.stale_seconds else 'Raw detector preview (not backend verdict)'
                try:
                    status_path=root/f'live_{camera}.status.json';status=json.loads(status_path.read_text())
                    # A server response is independently timestamped; don't label raw boxes as confirmed PPE results.
                    status_age=now-status_path.stat().st_mtime
                    if status_age<=args.stale_seconds and age<=args.stale_seconds:
                        verdict=f"Backend: {status.get('status','UNKNOWN')} | response age {status_age:.1f}s"
                except (FileNotFoundError,ValueError):pass
                cv2.putText(frame,verdict,(8,47),cv2.FONT_HERSHEY_SIMPLEX,.46,(0,200,255),1)
                tiles.append(frame)
            image=np.hstack(tiles)
            cv2.imshow('PPE processed monitor',image)
            if args.record and time.monotonic()>=next_record:
                if writer is None:
                    writer=cv2.VideoWriter(args.record,cv2.VideoWriter_fourcc(*'mp4v'),15,(image.shape[1],image.shape[0]))
                    if not writer.isOpened():raise RuntimeError('Could not open recording output')
                writer.write(image);next_record=time.monotonic()+1/15
            if cv2.waitKey(16)&255 in (ord('q'),27):break
    except KeyboardInterrupt:pass
    finally:
        if writer:writer.release();print(f'Saved {args.record}; 15 FPS playback, processed rate shown on video.')
        cv2.destroyAllWindows()

if __name__=='__main__':main()
