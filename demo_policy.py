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
