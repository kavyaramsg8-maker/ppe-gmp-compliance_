"""Show current annotated frames with authoritative, fresh backend decisions."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import cv2
import numpy as np


def confirmed_status(decision,evidence,now=None):
    now=now or datetime.now(timezone.utc)
    try:
        captured=datetime.fromisoformat(decision['captured_at'])
        if captured.tzinfo is None or not -5<=(now-captured).total_seconds()<=15:return 'UNKNOWN'
        if decision.get('stale'):return 'UNKNOWN'
        if decision.get('source_sha256')!=evidence.get('source_sha256'):return 'AWAITING_RESULT'
        return decision['status']
    except (ValueError,KeyError,TypeError):return 'UNKNOWN'


def read_json(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cameras',default='CAM_01',help='Comma-separated IDs')
    parser.add_argument('--record',help='Optional output MP4; press q to finalize it')
    args=parser.parse_args();cameras=[c.strip() for c in args.cameras.split(',') if c.strip()]
    if not 1<=len(cameras)<=2:parser.error('Use one or two camera IDs')
    if any(not all(ch.isalnum() or ch in '_-' for ch in c) for c in cameras):parser.error('Invalid camera ID')
    folder=Path(os.environ.get('PPE_MONITOR_DIR',str(Path(__file__).resolve().parent)))
    writer=None;fps=5
    colors={'COMPLIANT':(60,190,70),'VIOLATION':(50,50,220),'MONITORING':(180,140,60)}
    print('Monitor running. Press q in the image window to stop and save recording.')
    try:
        while True:
            start=time.monotonic();tiles=[]
            for cam in cameras:
                file=folder/f'live_{cam}.jpg'
                frame=cv2.imread(str(file)) if file.exists() else None
                if frame is None:
                    frame=np.full((440,1000,3),24,dtype=np.uint8)
                    cv2.putText(frame,'Waiting for a processed phone frame',(40,180),cv2.FONT_HERSHEY_SIMPLEX,.7,(230,230,230),1)
                frame=cv2.resize(frame,(1000,440))
                decision=read_json(folder/f'live_{cam}.decision.json');evidence=read_json(folder/f'live_{cam}.evidence.json')
                status=confirmed_status(decision,evidence)
                if file.exists() and time.time()-file.stat().st_mtime>15:status='UNKNOWN'
                cv2.rectangle(frame,(0,0),(999,47),(24,24,24),-1)
                cv2.putText(frame,f'{cam} | BACKEND: {status} | frame {decision.get("processed_frame_id","-")}',(10,27),cv2.FONT_HERSHEY_SIMPLEX,.58,colors.get(status,(0,190,240)),1)
                tiles.append(frame)
            canvas=np.vstack(tiles)
            cv2.imshow('GMP Compliance - q to stop',canvas)
            if args.record:
                if writer is None:
                    writer=cv2.VideoWriter(args.record,cv2.VideoWriter_fourcc(*'mp4v'),fps,(canvas.shape[1],canvas.shape[0]))
                    if not writer.isOpened():raise RuntimeError('Cannot open video output; check folder and free disk space')
                writer.write(canvas)
            if cv2.waitKey(max(1,int((1/fps-(time.monotonic()-start))*1000))) & 0xff in (ord('q'),27):break
    finally:
        if writer:writer.release()
        cv2.destroyAllWindows()
        if args.record:print('Recording saved:',args.record)


if __name__=='__main__':main()
