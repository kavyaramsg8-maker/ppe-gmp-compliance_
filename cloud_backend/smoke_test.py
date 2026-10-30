"""Explicitly synthetic integration test, not computer vision or physical hardware."""
import argparse
from datetime import datetime, timezone
from io import BytesIO
import os
import httpx
from PIL import Image, ImageDraw


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',default='http://127.0.0.1:8000')
    p.add_argument('--camera',default='CAM_01')
    p.add_argument('--confirm-test',action='store_true',help='Starts a new session on this camera; stops its current upload session')
    args=p.parse_args()
    if not args.confirm_test:
        p.error('Use --confirm-test only on your test service: this takes over the selected camera session')
    key=os.environ.get('PPE_API_KEY')
    if not key: p.error('Set PPE_API_KEY first')
    image=Image.new('RGB',(640,360),'#13202b')
    ImageDraw.Draw(image).text((30,130),'SIMULATED BACKEND TEST - NOT CV INFERENCE',fill='white')
    buffer=BytesIO();image.save(buffer,format='JPEG');data=buffer.getvalue()
    with httpx.Client(base_url=args.url.rstrip('/'),headers={'X-API-Key':key},timeout=15) as client:
        r=client.post(f'/v1/cameras/{args.camera}/sessions');r.raise_for_status();session=r.json()['session_id']
        frame_id=0
        for label,classes in [('present',['hairnet','maskon','glove']),('missing gloves',['hairnet','maskon','no_glove']),('restored',['hairnet','maskon','glove'])]:
            for _ in range(3):
                r=client.post(f'/v1/cameras/{args.camera}/frames',data={'session_id':session,'frame_id':str(frame_id),'captured_at':datetime.now(timezone.utc).isoformat()},files={'image':('test.jpg',data,'image/jpeg')});r.raise_for_status()
                r=client.post(f'/v1/cameras/{args.camera}/observations',json={'session_id':session,'frame_id':frame_id,'persons':[{'track_id':1,'associated_ppe':classes}]});r.raise_for_status()
                frame_id+=1
            print(label,'->',r.json()['status'])
            cmd=client.get('/v1/devices/ESP32_01/command');cmd.raise_for_status()
            if label=='missing gloves':
                assert cmd.json()['alert'] is True
                ack=client.post('/v1/devices/ESP32_01/ack',json={'command_id':cmd.json()['command_id']});ack.raise_for_status()
                assert client.get('/v1/devices/ESP32_01/command').json()['buzzer'] is False
            if label=='restored': assert cmd.json()['alert'] is False
        result=client.get('/v1/events',params={'camera_id':args.camera,'limit':200});result.raise_for_status()
        events=[e for e in result.json()['events'] if e['session_id']==session]
        assert sum(e['type']=='VIOLATION' for e in events)==1
        assert sum(e['type']=='RECOVERY' for e in events)==1
        snapshot=client.get(events[0]['snapshot_url']);snapshot.raise_for_status();assert snapshot.content==data
        print('PASS: saved violation, recovery, authenticated snapshot and simulated ESP32 acknowledgement')
        print('This test used synthetic labels. Actual model inference and physical ESP32 remain separate integration checks.')

if __name__=='__main__': main()
