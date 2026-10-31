from datetime import datetime,timezone
from fastapi.testclient import TestClient
from cloud_backend.app import Settings,create_app
from test_backend import KEY,HEADERS,jpeg
import demo_policy


def test_production_camera_is_not_split_into_thirds():
    assert hasattr(demo_policy,'camera_zone'), 'Camera-based layout missing'
    assert demo_policy.camera_zone('CAM_02',20,300)=='production'
    assert demo_policy.camera_zone('CAM_02',600,300)=='production'
    assert demo_policy.camera_zone('CAM_01',100,300)=='corridor'
    assert demo_policy.camera_zone('CAM_01',500,300)=='handwash'


def test_two_camera_policy_never_requires_cam01_handwash_identity_on_cam02(tmp_path):
    settings=Settings(api_key=KEY,data_dir=tmp_path,zone_mode='cameras',check_actions=True)
    with TestClient(create_app(settings)) as c:
        def send(cam,point,classes,**extra):
            sid=c.post(f'/v1/cameras/{cam}/sessions',headers=HEADERS).json()['session_id']
            c.post(f'/v1/cameras/{cam}/frames',headers=HEADERS,data={'session_id':sid,'frame_id':0,'captured_at':datetime.now(timezone.utc).isoformat()},files={'image':('a.jpg',jpeg(),'image/jpeg')})
            return c.post(f'/v1/cameras/{cam}/observations',headers=HEADERS,json={'session_id':sid,'frame_id':0,'persons':[{'track_id':1,'foot_point':point,'associated_ppe':classes,**extra}]}).json()
        first=send('CAM_01',[100,300],[],face_touch=False,handwash_incomplete=False)
        assert first['status']=='MONITORING'
        second=send('CAM_02',[20,300],['hairnet','maskon','glove'],face_touch=False,handwash_done=False)
        assert second['status']=='COMPLIANT'
        assert second['persons'][0]['zone']=='production'
        assert 'handwash' not in second['persons'][0]['evaluated']
        assert c.get('/v1/devices/ESP32_01/command',headers=HEADERS).json()['green_led']
