"""API contract tests: fresh data, isolation, persistence and safe device decisions."""
from datetime import datetime, timezone, timedelta
from io import BytesIO
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from cloud_backend.app import create_app, Settings

KEY='test-key-with-at-least-thirty-two-characters'
HEADERS={'X-API-Key': KEY}

def jpeg():
    out=BytesIO(); Image.new('RGB',(64,64),'gray').save(out,format='JPEG'); return out.getvalue()

@pytest.fixture
def setup(tmp_path):
    clock=[datetime.now(timezone.utc)]
    settings=Settings(api_key=KEY, data_dir=tmp_path, stale_seconds=15)
    with TestClient(create_app(settings, now=lambda:clock[0])) as client:
        yield client, clock, settings

def session(client,camera='CAM_01'):
    r=client.post(f'/v1/cameras/{camera}/sessions',headers=HEADERS)
    assert r.status_code==200, r.text
    return r.json()['session_id']

def upload(client,clock,sid,n,camera='CAM_01',captured=None,data=None):
    return client.post(f'/v1/cameras/{camera}/frames',headers=HEADERS,data={'session_id':sid,'frame_id':str(n),'captured_at':(captured or clock[0]).isoformat()},files={'image':('frame.jpg',data or jpeg(),'image/jpeg')})

def observation(client,sid,n,classes,camera='CAM_01',track=1):
    return client.post(f'/v1/cameras/{camera}/observations',headers=HEADERS,json={'session_id':sid,'frame_id':n,'persons':[{'track_id':track,'associated_ppe':classes}]})

def command(client):
    r=client.get('/v1/devices/ESP32_01/command',headers=HEADERS); assert r.status_code==200,r.text; return r.json()

PRESENT=['hairnet','maskon','glove']
MISSING=['hairnet','maskon','no_glove']

def test_authentication_and_unknown_defaults(setup):
    c,clock,_=setup
    assert c.get('/healthz').status_code==200
    assert c.get('/v1/events').status_code==401
    assert command(c)['status']=='UNKNOWN'
    assert command(c)['green_led'] is False
    assert c.post('/v1/cameras/UNREGISTERED/sessions',headers=HEADERS).status_code==404


def test_violation_dedup_recovery_snapshot_and_camera_isolation(setup):
    c,clock,_=setup; s=session(c); other=session(c,'CAM_02')
    for n in range(1,7):
        assert upload(c,clock,s,n).status_code==201
        result=observation(c,s,n,MISSING); assert result.status_code==200,result.text
    events=c.get('/v1/events',headers=HEADERS).json()['events']
    violations=[e for e in events if e['type']=='VIOLATION']
    assert len(violations)==1
    assert violations[0]['missing_items']==['gloves']
    snap=c.get(violations[0]['snapshot_url'],headers=HEADERS)
    assert snap.status_code==200 and snap.content==jpeg()
    cmd=command(c); assert cmd['alert'] and cmd['red_led'] and cmd['buzzer']
    assert c.post('/v1/devices/ESP32_01/ack',headers=HEADERS,json={'command_id':cmd['command_id']}).status_code==200
    assert command(c)['buzzer'] is False
    assert c.get('/v1/cameras/CAM_02',headers=HEADERS).json()['status']=='UNKNOWN'
    for n in range(7,10):
        upload(c,clock,s,n); observation(c,s,n,PRESENT)
    assert c.get('/v1/cameras/CAM_01',headers=HEADERS).json()['status']=='COMPLIANT'
    # Other camera has never been classified: cannot show overall green.
    assert command(c)['green_led'] is False
    assert not command(c)['alert']
    for n in range(1,4):
        upload(c,clock,other,n,'CAM_02'); observation(c,other,n,PRESENT,'CAM_02')
    assert command(c)['green_led'] is True


def test_unknown_does_not_become_compliance_or_recover_active_alert(setup):
    c,clock,_=setup; s=session(c)
    for n in range(1,4):
        upload(c,clock,s,n); observation(c,s,n,MISSING)
    upload(c,clock,s,4); observation(c,s,4,['hairnet','maskon'])
    assert command(c)['alert'] is True
    # Fresh session clears identity history, starts unknown.
    s=session(c); upload(c,clock,s,0); observation(c,s,0,[])
    assert c.get('/v1/cameras/CAM_01',headers=HEADERS).json()['status']=='UNKNOWN'


def test_stale_duplicate_old_session_and_late_result_rejected(setup):
    c,clock,_=setup; s=session(c)
    assert upload(c,clock,s,1).status_code==201
    assert upload(c,clock,s,1).status_code==409
    assert upload(c,clock,s,2,captured=clock[0]-timedelta(seconds=60)).status_code==422
    assert upload(c,clock,s,2).status_code==201
    assert observation(c,s,1,PRESENT).status_code==409
    assert observation(c,s,2,PRESENT).status_code==200
    assert observation(c,s,2,PRESENT).status_code==409
    clock[0]+=timedelta(seconds=16)
    assert command(c)['status']=='UNKNOWN'
    assert c.get('/v1/cameras/CAM_01',headers=HEADERS).json()['stale'] is True
    new=session(c)
    assert upload(c,clock,s,3).status_code==409
    assert upload(c,clock,new,0).status_code==201


def test_invalid_images_and_future_dates(setup):
    c,clock,_=setup; s=session(c)
    assert upload(c,clock,s,1,data=b'not-a-jpeg').status_code==422
    assert upload(c,clock,s,1,captured=clock[0]+timedelta(seconds=60)).status_code==422


def test_event_persistence_after_restart(setup):
    c,clock,settings=setup; s=session(c)
    for n in range(1,4):
        upload(c,clock,s,n); observation(c,s,n,MISSING)
    with TestClient(create_app(settings,now=lambda:clock[0])) as restarted:
        assert len(restarted.get('/v1/events',headers=HEADERS).json()['events'])==1
        assert command(restarted)['status']=='UNKNOWN'


def test_acknowledged_incident_does_not_buzz_again_when_other_camera_changes(setup):
    c,clock,_=setup; s=session(c); other=session(c,'CAM_02')
    for n in range(1,4):
        upload(c,clock,s,n); observation(c,s,n,MISSING)
    cmd=command(c)
    c.post('/v1/devices/ESP32_01/ack',headers=HEADERS,json={'command_id':cmd['command_id']})
    upload(c,clock,other,1,'CAM_02'); observation(c,other,1,PRESENT,'CAM_02')
    assert command(c)['alert'] is True
    assert command(c)['buzzer'] is False


def test_worker_claim_provides_backpressure(setup):
    c,clock,_=setup; s=session(c); upload(c,clock,s,1)
    frame=c.get('/v1/cameras/CAM_01/frame',headers={**HEADERS,'X-Claim-Frame':'true'})
    assert frame.status_code==200
    assert upload(c,clock,s,2).status_code==429
    assert observation(c,s,1,PRESENT).status_code==200
    assert upload(c,clock,s,2).status_code==201


def test_new_incident_after_recovery_can_buzz_again(setup):
    c,clock,_=setup; s=session(c)
    for n in range(1,4): upload(c,clock,s,n); observation(c,s,n,MISSING)
    cmd=command(c); c.post('/v1/devices/ESP32_01/ack',headers=HEADERS,json={'command_id':cmd['command_id']})
    for n in range(4,7): upload(c,clock,s,n); observation(c,s,n,PRESENT)
    for n in range(7,10): upload(c,clock,s,n); observation(c,s,n,MISSING)
    assert command(c)['buzzer'] is True
