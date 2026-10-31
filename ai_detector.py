"""Headless cloud-worker detector with isolated trackers and three demo regions.
No webcam is opened here. Raw model evidence is not a confirmed backend verdict.
"""
from collections import deque
from pathlib import Path
import os
import json
from hashlib import sha256
import re
import time
import cv2
import numpy as np
from ultralytics import YOLO
from demo_policy import PPE_ITEMS, PPE_CLASSES, WashTimer, zone_at, select_ppe, ppe_states

ROOT=Path(__file__).resolve().parent
_sessions={}
_ppe_model=None
_pose_model=None
COLORS={'present':(70,190,70),'missing':(60,60,230),'unknown':(0,190,240)}


def _session_for(camera_id,session_id):
    """A new session resets tracker IDs, timers and motion history for this camera."""
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',camera_id):raise ValueError('Invalid camera ID')
    if camera_id not in _sessions and len(_sessions)>=8:raise ValueError('Too many active camera trackers')
    old=_sessions.get(camera_id)
    if old is None or old['session_id']!=session_id:
        _sessions[camera_id]={'session_id':session_id,'person_model':YOLO(str(ROOT/'yolov8n.pt')),'people':{}}
    return _sessions[camera_id]


def _pose_evidence(crop):
    global _pose_model
    if _pose_model is None:_pose_model=YOLO(str(ROOT/'yolov8n-pose.pt'))
    r=_pose_model(crop,classes=[0],conf=.4,verbose=False)[0]
    if r.keypoints is None or len(r.keypoints.xy)==0:return None,None
    # The crop belongs to one tracked person; use the highest-confidence pose.
    k=int(r.boxes.conf.cpu().numpy().argmax()) if r.boxes is not None else 0
    xy=r.keypoints.xy[k].cpu().numpy();conf=r.keypoints.conf[k].cpu().numpy()
    if len(xy)<13 or not all(conf[i]>=.5 for i in [0,5,6,9,10]):return None,None
    shoulder=float(np.linalg.norm(xy[5]-xy[6]))
    if shoulder<5:return None,None
    touch=min(np.linalg.norm(xy[0]-xy[9]),np.linalg.norm(xy[0]-xy[10]))/shoulder < .4
    wrists=(xy[9],xy[10],shoulder)
    return bool(touch),wrists


def _draw(frame,camera_id,people):
    """Keep evidence in a side panel instead of stacking labels over the image."""
    top=52
    canvas=np.full((max(440,top+frame.shape[0]),1000,3),24,dtype=np.uint8)
    canvas[top:top+360,:640]=frame
    for i,(name,c) in enumerate([('GOWNING',(180,150,60)),('HANDWASH',(180,100,50)),('PRODUCTION',(80,120,190))]):
        x1=round(i*640/3);x2=round((i+1)*640/3)-1
        cv2.rectangle(canvas,(x1,top),(x2,top+359),c,1)
        cv2.putText(canvas,name,(x1+5,top+18),cv2.FONT_HERSHEY_SIMPLEX,.43,c,1)
    cv2.putText(canvas,f'{camera_id} | MODEL EVIDENCE - server verdict in monitor header',(10,26),cv2.FONT_HERSHEY_SIMPLEX,.48,(230,230,230),1)
    for i,p in enumerate(people):
        states=ppe_states(p['associated_ppe'])
        production=p['zone']=='production'
        state='missing' if production and 'missing' in states.values() else 'present' if production and all(v=='present' for v in states.values()) else 'unknown'
        c=COLORS[state];x1,y1,x2,y2=p['box']
        cv2.rectangle(canvas,(x1,y1+top),(x2,y2+top),c,2)
        cv2.putText(canvas,f"ID {p['track_id']}",(x1,min(top+350,max(top+36,y1+top-5))),cv2.FONT_HERSHEY_SIMPLEX,.45,c,1)
        if i>=3:continue
        y=70+i*120
        cv2.putText(canvas,f"ID {p['track_id']} | {p['zone'].upper()}",(650,y),cv2.FONT_HERSHEY_SIMPLEX,.45,(235,235,235),1)
        scores={PPE_CLASSES[d['class']][0]:d['confidence'] for d in p['associated_ppe']}
        for j,item in enumerate(PPE_ITEMS):
            value=states[item];score=f" {scores[item]:.0%}" if item in scores else ''
            cv2.putText(canvas,f'{item}: {value.upper()}{score}',(650,y+20+j*18),cv2.FONT_HERSHEY_SIMPLEX,.44,COLORS[value],1)
        cv2.putText(canvas,f"Wash motion: {p['handwash_seconds']:.1f}s / {p['handwash_required_seconds']:.0f}s",(650,y+82),cv2.FONT_HERSHEY_SIMPLEX,.39,(210,210,210),1)
        face='UNKNOWN' if p['face_touch'] is None else 'TOUCH' if p['face_touch'] else 'CLEAR'
        cv2.putText(canvas,f'Face: {face}',(650,y+100),cv2.FONT_HERSHEY_SIMPLEX,.4,(200,200,200),1)
    cv2.putText(canvas,'Yellow = unverified / non-production. Regions are configured demo areas.',(10,432),cv2.FONT_HERSHEY_SIMPLEX,.43,(200,200,200),1)
    target=Path(os.environ.get('PPE_MONITOR_DIR',str(ROOT)));target.mkdir(parents=True,exist_ok=True)
    tmp=target/f'.live_{camera_id}.tmp.jpg'
    if not cv2.imwrite(str(tmp),canvas):raise RuntimeError('Could not write annotated monitor frame')
    os.replace(tmp,target/f'live_{camera_id}.jpg')


def detect(frame_content,camera_id,session_id):
    global _ppe_model
    raw=cv2.imdecode(np.frombuffer(frame_content,np.uint8),cv2.IMREAD_COLOR)
    if raw is None:raise ValueError('Invalid JPEG; inference did not succeed')
    frame=cv2.resize(raw,(640,360))
    session=_session_for(camera_id,session_id)
    if _ppe_model is None:
        _ppe_model=YOLO(str(ROOT/'best.pt'))
        print('PPE model classes:',_ppe_model.names)
    threshold=float(os.environ.get('PPE_CONFIDENCE','.50'))
    if not .5<=threshold<=1:raise ValueError('PPE_CONFIDENCE must be between 0.5 and 1')
    required=float(os.environ.get('PPE_HANDWASH_SECONDS','20'))
    if not 0 < required <= 300:raise ValueError('Invalid PPE_HANDWASH_SECONDS')
    ppes=_ppe_model(frame,conf=threshold,iou=.4,verbose=False)[0]
    tracked=session['person_model'].track(frame,classes=[0],conf=.5,persist=True,tracker='bytetrack.yaml',verbose=False)[0]
    candidates=[]
    if ppes.boxes is not None:
        for b,cls,score in zip(ppes.boxes.xyxy.cpu().numpy(),ppes.boxes.cls.cpu().numpy(),ppes.boxes.conf.cpu().numpy()):
            candidates.append({'box':b,'class':_ppe_model.names[int(cls)],'confidence':float(score)})
    people=[];now=time.monotonic()
    if tracked.boxes is not None and tracked.boxes.id is not None:
        for b,tid in zip(tracked.boxes.xyxy.cpu().numpy(),tracked.boxes.id.cpu().numpy()):
            x1,y1,x2,y2=map(int,b);x1=max(0,x1);y1=max(0,y1);x2=min(639,x2);y2=min(359,y2)
            if (x2-x1)*(y2-y1)<2000:continue
            tid=int(tid);foot=[(x1+x2)/2,y2];zone=zone_at(*foot)
            people.append({'track_id':tid,'box':[x1,y1,x2,y2],'foot_point':foot,'zone':zone,'associated_ppe':[]})
    # Assign an item to one person, not every overlapping person box.
    for d in candidates:
        b=d['box'];area=max(0,b[2]-b[0])*max(0,b[3]-b[1]);matches=[]
        if not area:continue
        for p in people:
            x1,y1,x2,y2=p['box'];inter=max(0,min(b[2],x2)-max(b[0],x1))*max(0,min(b[3],y2)-max(b[1],y1))
            if inter/area>=.6:matches.append((inter/area,-(x2-x1)*(y2-y1),p))
        if matches:
            owner=max(matches,key=lambda m:(m[0],m[1]))[2]
            owner['associated_ppe'].append({'class':d['class'],'confidence':d['confidence']})
    for p in people:
        p['associated_ppe']=select_ppe(p['associated_ppe'],threshold)
        st=session['people'].setdefault(p['track_id'],{'timer':WashTimer(required),'history':deque(maxlen=5),'seen':now})
        if now-st['seen']>15:st.update(timer=WashTimer(required),history=deque(maxlen=5))
        st['seen']=now
        x1,y1,x2,y2=p['box'];crop=frame[y1:y2,x1:x2]
        face,wrists=_pose_evidence(crop) if crop.size else (None,None)
        active=False
        if p['zone']=='handwash' and wrists is not None:
            a,b,shoulder=wrists
            both_in_zone=all(zone_at(float(w[0]+x1),float(w[1]+y1))=='handwash' for w in (a,b))
            center=(a+b)/2/shoulder
            st['history'].append((now,center))
            h=list(st['history'])
            if both_in_zone and np.linalg.norm(a-b)/shoulder<1.2 and len(h)>=3 and now-h[0][0]<=8:
                delta=[float(h[i][1][0]-h[i-1][1][0]) for i in range(1,len(h))]
                active=any(abs(d)>.02 for d in delta) and any(delta[i]*delta[i-1]<0 for i in range(1,len(delta)))
        else:st['history'].clear()
        st['timer'].update(now,active)
        p.update(face_touch=face,handwash_done=st['timer'].done,handwash_seconds=round(st['timer'].elapsed,2),handwash_required_seconds=required)
    session['people']={tid:st for tid,st in session['people'].items() if now-st['seen']<=15}
    _draw(frame,camera_id,people)
    target=Path(os.environ.get('PPE_MONITOR_DIR',str(ROOT)))
    meta=target/f'live_{camera_id}.evidence.json'
    tmp=meta.with_suffix('.tmp')
    tmp.write_text(json.dumps({'source_sha256':sha256(frame_content).hexdigest(),'session_id':session_id}))
    os.replace(tmp,meta)
    print(f"{camera_id}: "+', '.join(f"ID {p['track_id']} {p['zone']} {p['associated_ppe']}" for p in people))
    return people
