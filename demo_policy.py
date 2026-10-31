"""Shared, model-independent policy for the three-region demonstration."""
from dataclasses import dataclass
import math

PPE_CLASSES = {'hairnet':('hairnet','present'),'no_hairnet':('hairnet','missing'),
               'maskon':('mask','present'),'maskoff':('mask','missing'),
               'glove':('gloves','present'),'no_glove':('gloves','missing')}
PPE_ITEMS = ('hairnet','mask','gloves')
ZONE_NAMES = ('gowning_room','handwash','production')


def zone_at(x, y, width=640, height=360):
    if not (math.isfinite(x) and math.isfinite(y)) or not (0 <= x < width and 0 <= y <= height):
        return 'outside'
    return ZONE_NAMES[min(2,int(x * 3 / width))]


def select_ppe(detections, threshold=.5, conflict_margin=.10):
    """One reliable state per item. Close contradictory predictions stay unknown."""
    best={}
    for d in detections:
        name=d['class']; score=float(d['confidence'])
        if name not in PPE_CLASSES or not math.isfinite(score) or not threshold <= score <= 1:
            continue
        if name not in best or score > best[name]['confidence']:
            best[name]={'class':name,'confidence':score}
    result=[]
    for item in PPE_ITEMS:
        choices=sorted((d for name,d in best.items() if PPE_CLASSES[name][0]==item),key=lambda d:d['confidence'],reverse=True)
        if len(choices)>1 and choices[0]['confidence']-choices[1]['confidence'] < conflict_margin:
            continue
        if choices:result.append(choices[0])
    return result


def ppe_states(detections):
    result={item:'unknown' for item in PPE_ITEMS}
    for d in detections:
        if d['class'] in PPE_CLASSES:
            item,state=PPE_CLASSES[d['class']];result[item]=state
    return result


@dataclass
class WashTimer:
    required_seconds: float = 20
    elapsed: float = 0
    last_time: float | None = None
    last_active: bool = False

    @property
    def done(self):
        return self.elapsed >= self.required_seconds

    def update(self, now, active):
        if self.last_time is not None and active and self.last_active:
            dt=now-self.last_time
            if 0 < dt <= 2.5:self.elapsed=min(self.required_seconds,self.elapsed+dt)
        self.last_time=now;self.last_active=active
        return self.elapsed


def camera_zone(camera_id,x=None,y=None,roi=(.5,0,1,1)):
    """CAM_01 corridor + handwash ROI; all of CAM_02 is production."""
    if camera_id=='CAM_02':return 'production'
    if camera_id!='CAM_01' or x is None or y is None:return 'outside'
    if not(math.isfinite(x) and math.isfinite(y)) or not(0<=x<640 and 0<=y<=360):return 'outside'
    a,b,c,d=roi
    return 'handwash' if a<=x/640<=c and b<=y/360<=d else 'corridor'


def handwash_roi():
    import os,json
    roi=tuple(float(v) for v in json.loads(os.environ.get('PPE_HANDWASH_ROI','[0.5,0,1,1]')))
    if len(roi)!=4 or not(0<=roi[0]<roi[2]<=1 and 0<=roi[1]<roi[3]<=1):
        raise ValueError('PPE_HANDWASH_ROI must be [left,top,right,bottom] between 0 and 1')
    return roi
