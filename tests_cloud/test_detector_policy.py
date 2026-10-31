import importlib
import importlib.util
from pathlib import Path
import pytest


def policy():
    assert Path('demo_policy.py').exists(), 'Shared zone/PPE policy is missing'
    return importlib.import_module('demo_policy')


def test_ambiguous_or_low_confidence_is_unknown_not_compliant():
    p=policy()
    assert p.select_ppe([{'class':'glove','confidence':.2}])==[]
    assert p.select_ppe([{'class':'maskon','confidence':.8},{'class':'maskoff','confidence':.79}])==[]
    assert p.select_ppe([{'class':'maskon','confidence':.9},{'class':'maskon','confidence':.7}])==[{'class':'maskon','confidence':.9}]


def test_three_zones_scale_with_resolution():
    p=policy()
    assert [p.zone_at(x,300) for x in [100,320,550]]==['gowning_room','handwash','production']
    assert p.zone_at(1100,600,1280,720)=='production'
    assert p.zone_at(-2,300)=='outside'


def test_handwash_only_accumulates_observed_active_intervals():
    p=policy();t=p.WashTimer(required_seconds=2)
    assert t.update(0,True)==0
    assert t.update(1,True)==1
    t.update(2,False)
    assert t.update(20,True)==1
    assert t.update(21,True)==2
    assert t.done


def test_session_and_camera_have_distinct_trackers(monkeypatch):
    import sys,types,numpy as np
    created=[]
    class Model:
        def __init__(self,path):self.path=path;created.append(self)
    monkeypatch.setitem(sys.modules,'ultralytics',types.SimpleNamespace(YOLO=Model))
    sys.modules.pop('ai_detector',None)
    detector=importlib.import_module('ai_detector')
    assert hasattr(detector,'_session_for'), 'Detector has no camera/session isolation'
    a=detector._session_for('CAM_01','a')
    b=detector._session_for('CAM_02','b')
    assert a['person_model'] is not b['person_model']
    assert detector._session_for('CAM_01','new') is not a
    with pytest.raises(ValueError):detector.detect(b'not a jpeg','CAM_01','new')


def test_detector_returns_one_scored_label_per_item_and_tracks_people_only(monkeypatch,tmp_path):
    import sys,types,numpy as np,cv2
    class Array:
        def __init__(self,x):self.x=np.array(x)
        def cpu(self):return self
        def numpy(self):return self.x
    class Model:
        names={0:'maskon',1:'maskoff',2:'glove'}
        def __init__(self,path):self.path=path
        def __call__(self,frame,**kwargs):
            if 'pose' in self.path:return [types.SimpleNamespace(keypoints=None)]
            return [types.SimpleNamespace(boxes=types.SimpleNamespace(xyxy=Array([[450,50,470,70],[452,51,472,71],[460,80,480,100]]),cls=Array([0,0,2]),conf=Array([.9,.8,.3])))]
        def track(self,frame,**kwargs):
            assert kwargs['classes']==[0]
            return [types.SimpleNamespace(boxes=types.SimpleNamespace(xyxy=Array([[430,20,620,350]]),id=Array([1])))]
    monkeypatch.setitem(sys.modules,'ultralytics',types.SimpleNamespace(YOLO=Model))
    monkeypatch.setenv('PPE_MONITOR_DIR',str(tmp_path))
    sys.modules.pop('ai_detector',None);d=importlib.import_module('ai_detector')
    ok,encoded=cv2.imencode('.jpg',np.zeros((360,640,3),dtype=np.uint8));assert ok
    people=d.detect(encoded.tobytes(),'CAM_01','s')
    assert len(people)==1 and people[0]['zone']=='production'
    assert people[0]['associated_ppe']==[{'class':'maskon','confidence':.9}]
    assert (tmp_path/'live_CAM_01.jpg').exists()
