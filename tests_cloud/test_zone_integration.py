from datetime import datetime, timezone
from fastapi.testclient import TestClient
from cloud_backend.app import Settings, create_app
from test_backend import KEY, HEADERS, jpeg


def test_three_zone_server_derives_zone_and_production_requires_ppe(tmp_path):
    settings=Settings(api_key=KEY,data_dir=tmp_path,cameras=('CAM_01',),zone_mode='three')
    with TestClient(create_app(settings)) as c:
        sid=c.post('/v1/cameras/CAM_01/sessions',headers=HEADERS).json()['session_id']
        def send(n,x,classes):
            r=c.post('/v1/cameras/CAM_01/frames',headers=HEADERS,data={'session_id':sid,'frame_id':n,'captured_at':datetime.now(timezone.utc).isoformat()},files={'image':('a.jpg',jpeg(),'image/jpeg')})
            assert r.status_code==201
            r=c.post('/v1/cameras/CAM_01/observations',headers=HEADERS,json={'session_id':sid,'frame_id':n,'persons':[{'track_id':1,'foot_point':[x,300],'zone':'gowning_room','associated_ppe':classes}]})
            assert r.status_code==200,r.text
            return r.json()
        r=send(1,100,[])
        assert r['persons'][0]['zone']=='gowning_room'
        assert r['status']=='MONITORING'
        assert not c.get('/v1/devices/ESP32_01/command',headers=HEADERS).json()['green_led']
        for n in range(2,5):r=send(n,550,['hairnet','maskoff','glove'])
        assert r['status']=='VIOLATION'
        assert r['persons'][0]['zone']=='production'
        assert r['active_violations'][0]['item']=='mask'
        r=send(5,320,[])
        assert r['status']=='MONITORING'
        assert not r['active_violations']
        r=send(6,550,['hairnet','maskon','glove'])
        assert r['status']=='COMPLIANT'


def test_three_zone_missing_position_is_unknown(tmp_path):
    with TestClient(create_app(Settings(api_key=KEY,data_dir=tmp_path,cameras=('CAM_01',),zone_mode='three'))) as c:
        sid=c.post('/v1/cameras/CAM_01/sessions',headers=HEADERS).json()['session_id']
        c.post('/v1/cameras/CAM_01/frames',headers=HEADERS,data={'session_id':sid,'frame_id':0,'captured_at':datetime.now(timezone.utc).isoformat()},files={'image':('a.jpg',jpeg(),'image/jpeg')})
        r=c.post('/v1/cameras/CAM_01/observations',headers=HEADERS,json={'session_id':sid,'frame_id':0,'persons':[{'track_id':1,'associated_ppe':['hairnet','maskon','glove']}]})
        assert r.json()['status']=='UNKNOWN'


def test_action_flags_reach_events_and_recover(tmp_path):
    settings=Settings(api_key=KEY,data_dir=tmp_path,cameras=('CAM_01',),zone_mode='three',check_actions=True)
    with TestClient(create_app(settings)) as c:
        sid=c.post('/v1/cameras/CAM_01/sessions',headers=HEADERS).json()['session_id']
        for n in range(6):
            c.post('/v1/cameras/CAM_01/frames',headers=HEADERS,data={'session_id':sid,'frame_id':n,'captured_at':datetime.now(timezone.utc).isoformat()},files={'image':('a.jpg',jpeg(),'image/jpeg')})
            r=c.post('/v1/cameras/CAM_01/observations',headers=HEADERS,json={'session_id':sid,'frame_id':n,'persons':[{'track_id':1,'foot_point':[550,300],'associated_ppe':['hairnet','maskon','glove'],'face_touch':n<3,'handwash_done':n>=3}]})
            if n==2:
                assert r.json()['status']=='VIOLATION'
                assert {v['item'] for v in r.json()['active_violations']}=={'face_touch','handwash'}
        assert r.json()['status']=='COMPLIANT'
        assert len(c.get('/v1/events',headers=HEADERS).json()['events'])==4
