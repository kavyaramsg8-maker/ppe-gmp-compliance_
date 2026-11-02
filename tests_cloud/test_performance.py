import importlib
from pathlib import Path
import ast
import time


def tools_module():
    assert Path('performance_tools.py').exists(), 'Missing performance instrumentation'
    return importlib.import_module('performance_tools')


def test_throughput_is_accepted_frames_over_elapsed_not_display_fps():
    p=tools_module()
    s=p.summarize([{'accepted':True,'camera_id':'CAM_01','detector_ms':100,'cycle_ms':500,'capture_age_ms':800,'rss_mib':20},
                   {'accepted':False,'camera_id':'CAM_02','detector_ms':100,'cycle_ms':500,'capture_age_ms':800,'rss_mib':30}],elapsed_seconds=2)
    assert s['accepted_fps']==.5
    assert s['attempted_frames']==2
    assert s['peak_rss_mib']==30
    assert s['detector_ms']['p95']==100


def test_atomic_preview_never_publishes_half_a_jpeg(tmp_path):
    p=tools_module();dest=tmp_path/'a.jpg'
    p.atomic_bytes(dest,b'first');p.atomic_bytes(dest,b'second')
    assert dest.read_bytes()==b'second'
    assert not list(tmp_path.glob('*.tmp'))


def test_camera_scheduling_has_no_one_second_cap():
    source=Path('cloud_backend/camera.html').read_text()
    assert 'id="fps"' in source, 'Upload rate must be visible and adjustable'
    assert '1000/Number(el(\'fps\').value)' in source


def test_detector_reuses_pose_and_keeps_camera_trackers_separate(monkeypatch,tmp_path):
    import sys
    import types
    import numpy as np
    import cv2
    calls=[];track_settings=[]
    class Array:
        def __init__(self,value):self.value=np.array(value)
        def cpu(self):return self
        def numpy(self):return self.value
    class Model:
        names={0:'glove',1:'hairnet'}
        def __init__(self,*args):pass
        def __call__(self,*args,**kwargs):return [types.SimpleNamespace(boxes=None)]
        def track(self,*args,**kwargs):
            track_settings.append(kwargs)
            return [types.SimpleNamespace(boxes=types.SimpleNamespace(xyxy=Array([[350,10,620,350]]),id=Array([1])))]
    class Pose:
        def __init__(self,**kwargs):pass
        def process(self,image):calls.append(image.shape);return types.SimpleNamespace(pose_landmarks=None)
    monkeypatch.setitem(sys.modules,'ultralytics',types.SimpleNamespace(YOLO=Model))
    monkeypatch.setitem(sys.modules,'mediapipe',types.SimpleNamespace(solutions=types.SimpleNamespace(pose=types.SimpleNamespace(Pose=Pose))))
    monkeypatch.setitem(sys.modules,'torch',types.SimpleNamespace())
    monkeypatch.setenv('PPE_DEVICE','cpu');monkeypatch.setenv('PPE_MONITOR_DIR',str(tmp_path))
    spec=importlib.util.spec_from_file_location('detector_under_test',Path('ai_detector.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    _,encoded=cv2.imencode('.jpg',np.zeros((360,640,3),np.uint8))
    first=module.detect(encoded.tobytes(),'CAM_01','a')
    assert len(first)==1 and len(calls)==1  # Face and sink share one pose result.
    module.detect(encoded.tobytes(),'CAM_02','b')
    assert len(calls)==2
    assert module._sessions['CAM_01']['person_model'] is not module._sessions['CAM_02']['person_model']
    assert all(k['classes']==[0] for k in track_settings)
    assert (tmp_path/'live_CAM_01.jpg').exists()


def test_camera_downloads_overlap_but_shared_detector_is_serial(tmp_path,monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    import httpx
    from cloud_backend.worker import process_once
    barrier=threading.Barrier(2);active=0;peak=0;guard=threading.Lock()
    def request(req):
        path=req.url.path
        if path=='/v1/cameras':
            return httpx.Response(200,json={'cameras':[{'camera_id':c,'latest_frame_id':1,'processed_frame_id':None} for c in ['CAM_01','CAM_02']]})
        if path.endswith('/frame'):
            barrier.wait(timeout=3)
            return httpx.Response(200,content=b'jpeg',headers={'X-Session-ID':'s','X-Frame-ID':'1'})
        return httpx.Response(200,json={'status':'COMPLIANT'})
    def detector(*args):
        nonlocal active,peak
        with guard:active+=1;peak=max(peak,active)
        time.sleep(.02)
        with guard:active-=1
        return []
    monkeypatch.setenv('PPE_MONITOR_DIR',str(tmp_path))
    with httpx.Client(base_url='https://test',transport=httpx.MockTransport(request)) as client,ThreadPoolExecutor(max_workers=2) as pool:
        assert process_once(client,detector,pool)==2
    assert peak==1
    assert len(list(tmp_path.glob('*.status.json')))==2


def test_benchmark_writes_scoped_report_for_both_variants(monkeypatch,tmp_path):
    """Exercise CLI/report assembly with deterministic model doubles, not real speed claims."""
    import sys
    import types
    import cv2
    import numpy as np
    from benchmark_performance import main
    source=tmp_path/'clip.avi'
    writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'MJPG'),5,(64,36))
    assert writer.isOpened()
    for i in range(4):writer.write(np.full((36,64,3),i*40,np.uint8))
    writer.release()
    class Model:
        names={}
        def __init__(self,*args):self.model=types.SimpleNamespace(parameters=lambda:[types.SimpleNamespace(numel=lambda:10)])
        def __call__(self,*args,**kwargs):return [types.SimpleNamespace(boxes=None)]
        track=__call__
    monkeypatch.setitem(sys.modules,'torch',types.SimpleNamespace(__version__='double'))
    monkeypatch.setitem(sys.modules,'ultralytics',types.SimpleNamespace(YOLO=Model))
    monkeypatch.setitem(sys.modules,'ultralytics.utils',types.ModuleType('ultralytics.utils'))
    monkeypatch.setitem(sys.modules,'ultralytics.utils.torch_utils',types.SimpleNamespace(get_flops=lambda *a,**k:2.0))
    monkeypatch.setitem(sys.modules,'mediapipe',types.SimpleNamespace(solutions=types.SimpleNamespace(pose=types.SimpleNamespace(Pose=lambda **k:object()))))
    model=Model()
    optimized=types.SimpleNamespace(ppe_model=model,_sessions={'CAM_01':{'person_model':model}},detect=lambda *a:[],get_metrics=lambda c:{'ppe_ms':1,'person_ms':1,'pose_ms':0})
    monkeypatch.setitem(sys.modules,'ai_detector',optimized)
    import json
    for variant in ['baseline','optimized']:
        dest=tmp_path/f'{variant}.json'
        monkeypatch.setattr(sys,'argv',['benchmark','--variant',variant,'--source',str(source),'--frames','2','--warmup','1','--output',str(dest)])
        main();result=json.loads(dest.read_text())
        assert result['measured_frames']==2
        assert result['unique_yolo_parameters']==20
        assert result['models'][0]['input_shape']==[1,3,384,640]
        assert 'No cloud/network/ESP32' in result['scope']
