from fastapi.testclient import TestClient
from cloud_backend.worker import process_once
from test_backend import KEY, HEADERS, jpeg, session, Settings, create_app
from datetime import datetime, timezone

def test_worker_connects_frame_to_inference_result(tmp_path):
    with TestClient(create_app(Settings(api_key=KEY,data_dir=tmp_path,cameras=('CAM_01',)))) as client:
        sid=session(client)
        r=client.post('/v1/cameras/CAM_01/frames',headers=HEADERS,data={'session_id':sid,'frame_id':'1','captured_at':datetime.now(timezone.utc).isoformat()},files={'image':('frame.jpg',jpeg(),'image/jpeg')})
        assert r.status_code==201
        calls=[]
        def detector(image_bytes,camera_id,session_id):
            calls.append((image_bytes,camera_id,session_id))
            return [{'track_id':1,'associated_ppe':['hairnet','maskon','glove']}]
        client.headers.update(HEADERS)
        assert process_once(client,detector)==1
        assert calls==[(jpeg(),'CAM_01',sid)]
        assert client.get('/v1/cameras/CAM_01').json()['status']=='COMPLIANT'
        assert process_once(client,detector)==0
