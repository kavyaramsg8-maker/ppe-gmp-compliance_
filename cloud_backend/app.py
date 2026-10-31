"""Run with: uvicorn cloud_backend.app:create_app --factory --workers 1."""
from __future__ import annotations

from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from threading import RLock
from typing import Annotated
import hmac
import json
import os
import sqlite3
import uuid

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import Response, HTMLResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field, field_validator

from ml_adapter import MLAdapter
from demo_policy import zone_at


@dataclass(frozen=True)
class Settings:
    api_key: str
    data_dir: Path
    cameras: tuple[str, ...] = ('CAM_01', 'CAM_02')
    required_ppe: tuple[str, ...] = ('hairnet', 'mask', 'gloves')
    stale_seconds: int = 15
    voting_k: int = 3
    voting_n: int = 5
    recovery_frames: int = 3
    max_image_bytes: int = 2 * 1024 * 1024
    zone_mode: str = 'production'
    check_actions: bool = False

    def __post_init__(self):
        if self.zone_mode not in ('production', 'three'):
            raise ValueError('PPE_ZONE_MODE must be production or three')
        if len(self.api_key) < 32:
            raise ValueError('PPE_API_KEY must contain at least 32 characters')
        if not self.cameras or len(set(self.cameras)) != len(self.cameras):
            raise ValueError('Configure unique camera IDs')
        if not 1 <= self.voting_k <= self.voting_n or self.recovery_frames < 1 or self.stale_seconds < 1:
            raise ValueError('Invalid temporal configuration')

    @classmethod
    def from_env(cls):
        return cls(api_key=os.environ.get('PPE_API_KEY', ''),
                   data_dir=Path(os.environ.get('PPE_DATA_DIR', 'cloud_data')),
                   cameras=tuple(c.strip() for c in os.environ.get('PPE_CAMERA_IDS', 'CAM_01,CAM_02').split(',') if c.strip()),
                   stale_seconds=int(os.environ.get('PPE_STALE_SECONDS', '15')),
                   zone_mode=os.environ.get('PPE_ZONE_MODE', 'production'),
                   check_actions=os.environ.get('PPE_CHECK_ACTIONS', 'false').lower() == 'true')


class Detection(BaseModel):
    class_name: str = Field(alias='class', max_length=80)
    confidence: float = Field(ge=0, le=1)


class Person(BaseModel):
    track_id: int = Field(ge=0)
    foot_point: tuple[float, float] | None = None
    face_touch: bool | None = None
    handwash_done: bool | None = None
    associated_ppe: list[str | Detection] = Field(default_factory=list, max_length=50)

    @field_validator('associated_ppe')
    @classmethod
    def bounded_names(cls, value):
        if any(isinstance(v, str) and len(v) > 80 for v in value):
            raise ValueError('Class names must be at most 80 characters')
        return value


class Observation(BaseModel):
    session_id: uuid.UUID
    frame_id: int = Field(ge=0)
    persons: list[Person] = Field(default_factory=list, max_length=50)

    @field_validator('persons')
    @classmethod
    def unique_tracks(cls, value):
        if len({p.track_id for p in value}) != len(value):
            raise ValueError('track_id must be unique within this frame')
        return value


class Ack(BaseModel):
    command_id: str = Field(min_length=1, max_length=64)


def create_app(settings: Settings | None = None, now=None) -> FastAPI:
    settings = settings or Settings.from_env()
    now = now or (lambda: datetime.now(timezone.utc))
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(settings.data_dir / 'ppe_cloud.db', check_same_thread=False)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('''CREATE TABLE IF NOT EXISTS events (
        event_id TEXT PRIMARY KEY, type TEXT NOT NULL, camera_id TEXT NOT NULL,
        session_id TEXT NOT NULL, frame_id INTEGER NOT NULL, captured_at TEXT NOT NULL,
        recorded_at TEXT NOT NULL, track_id INTEGER NOT NULL, item TEXT NOT NULL,
        snapshot BLOB NOT NULL)''')
    db.execute('CREATE INDEX IF NOT EXISTS events_recorded ON events(recorded_at DESC)')
    db.execute('''CREATE TABLE IF NOT EXISTS device_state (
        device_id TEXT PRIMARY KEY, last_seen TEXT NOT NULL, acknowledged_command TEXT)''')
    db.execute('CREATE TABLE IF NOT EXISTS acknowledged_incidents (device_id TEXT NOT NULL, event_id TEXT NOT NULL, PRIMARY KEY(device_id,event_id))')
    db.commit()
    lock = RLock()
    cameras = {c: {} for c in settings.cameras}

    @asynccontextmanager
    async def lifespan(app):
        yield
        db.close()

    app = FastAPI(title='PPE Cloud Backend', version='1.0.0', lifespan=lifespan,
                  description='Phone JPEG upload → cloud detector result → persistent events → ESP32 polling. Unknown is never compliant.')

    def auth(x_api_key: Annotated[str | None, Header()] = None):
        if x_api_key is None or not hmac.compare_digest(x_api_key, settings.api_key):
            raise HTTPException(401, 'Valid X-API-Key required')

    secured = [Depends(auth)]

    def camera(camera_id):
        if camera_id not in cameras:
            raise HTTPException(404, 'Camera is not configured')
        return cameras[camera_id]

    def age(timestamp):
        return (now() - timestamp).total_seconds()

    def state(camera_id):
        c = camera(camera_id)
        frame = c.get('frame')
        decision = c.get('decision')
        stale = (not decision or age(decision['captured_dt']) > settings.stale_seconds
                 or age(decision['received_dt']) > settings.stale_seconds)
        return dict(camera_id=camera_id, session_id=c.get('session_id'),
                    status='UNKNOWN' if stale else decision['status'], stale=stale,
                    latest_frame_id=frame['frame_id'] if frame else None,
                    processed_frame_id=decision['frame_id'] if decision else None,
                    captured_at=decision['captured_dt'].isoformat() if decision else None,
                    persons=[] if stale else decision['persons'],
                    active_violations=[] if stale else decision['active_violations'],
                    decision_age_seconds=round(max(0, age(decision['captured_dt'])), 3) if decision else None)

    def record(c, camera_id, track_id, item, kind):
        f = c['frame']; event_id = str(uuid.uuid4())
        db.execute('INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?)',
                   (event_id, kind, camera_id, c['session_id'], f['frame_id'], f['captured_dt'].isoformat(), now().isoformat(), track_id, item, f['jpeg']))
        return event_id

    @app.get('/camera', response_class=HTMLResponse, include_in_schema=False)
    def camera_page():
        return HTMLResponse((Path(__file__).parent / 'camera.html').read_text(), headers={'Cache-Control': 'no-store', 'Content-Security-Policy': "frame-ancestors 'none'", 'Referrer-Policy': 'no-referrer'})

    @app.get('/camera-config', include_in_schema=False)
    def camera_config():
        return {'cameras': settings.cameras, 'zone_mode': settings.zone_mode, 'check_actions': settings.check_actions}

    @app.get('/healthz')
    def health():
        with lock:
            db.execute('SELECT 1').fetchone()
        return {'status': 'ok', 'service': 'ppe-cloud', 'inference': 'external-worker', 'time': now().isoformat()}

    @app.post('/v1/cameras/{camera_id}/sessions', dependencies=secured)
    def start_session(camera_id: str):
        with lock:
            c = camera(camera_id)
            c.clear()
            c.update(session_id=str(uuid.uuid4()), adapter=MLAdapter(hold_frames=0), tracks={})
            return {'camera_id': camera_id, 'session_id': c['session_id'], 'next_frame_id': 0}

    @app.post('/v1/cameras/{camera_id}/frames', dependencies=secured, status_code=201)
    async def frame_upload(camera_id: str, session_id: Annotated[uuid.UUID, Form()],
                           frame_id: Annotated[int, Form(ge=0)], captured_at: Annotated[datetime, Form()],
                           image: Annotated[UploadFile, File()]):
        if captured_at.tzinfo is None:
            raise HTTPException(422, 'captured_at must include UTC offset, for example +00:00')
        data = await image.read(settings.max_image_bytes + 1)
        await image.close()
        if len(data) > settings.max_image_bytes:
            raise HTTPException(413, 'JPEG exceeds 2 MiB limit')
        try:
            with Image.open(BytesIO(data)) as im:
                if im.format != 'JPEG' or im.width * im.height > 1920 * 1080:
                    raise HTTPException(422, 'Use a JPEG of at most 1920 × 1080 pixels')
                im.verify()
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
            raise HTTPException(422, 'Invalid JPEG')
        with lock:
            c = camera(camera_id)
            if str(session_id) != c.get('session_id'):
                raise HTTPException(409, 'Session expired; start a new camera session')
            if not -5 <= age(captured_at) <= settings.stale_seconds:
                raise HTTPException(422, 'Frame timestamp is stale or too far in the future; sync camera clock')
            if frame_id <= c.get('frame', {}).get('frame_id', -1):
                raise HTTPException(409, 'Duplicate or out-of-order frame')
            claimed = c.get('claimed_at')
            if claimed and age(claimed) < min(10, settings.stale_seconds):
                raise HTTPException(429, 'Detector is processing this camera; drop this frame and retry later', headers={'Retry-After': '1'})
            # One latest frame per camera: no unbounded queue or old-frame backlog.
            c['frame'] = dict(frame_id=frame_id, captured_dt=captured_at, received_dt=now(), jpeg=data)
            return {'camera_id': camera_id, 'session_id': str(session_id), 'frame_id': frame_id,
                    'status': 'AWAITING_INFERENCE'}

    @app.get('/v1/cameras', dependencies=secured)
    def list_cameras():
        with lock:
            return {'cameras': [state(c) for c in cameras], 'required_ppe': settings.required_ppe,
                    'voting': {'k': settings.voting_k, 'n': settings.voting_n, 'recovery_frames': settings.recovery_frames}}

    @app.get('/v1/cameras/{camera_id}', dependencies=secured)
    def camera_status(camera_id: str):
        with lock:
            return state(camera_id)

    @app.get('/v1/cameras/{camera_id}/frame', dependencies=secured)
    def latest_frame(camera_id: str, x_claim_frame: Annotated[bool, Header()] = False):
        with lock:
            c = camera(camera_id); f = c.get('frame')
            if not f:
                raise HTTPException(404, 'No frame available')
            if age(f['captured_dt']) > settings.stale_seconds:
                raise HTTPException(410, 'Latest frame has expired')
            if x_claim_frame:
                c['claimed_at'] = now()
            return Response(f['jpeg'], media_type='image/jpeg', headers={
                'X-Frame-ID': str(f['frame_id']), 'X-Session-ID': c['session_id'],
                'X-Captured-At': f['captured_dt'].isoformat(), 'Cache-Control': 'no-store'})

    @app.post('/v1/cameras/{camera_id}/observations', dependencies=secured)
    def observations(camera_id: str, observation: Observation):
        with lock:
            c = camera(camera_id); f = c.get('frame')
            if not f or str(observation.session_id) != c.get('session_id') or observation.frame_id != f['frame_id']:
                raise HTTPException(409, 'Result must match the latest camera session and frame')
            if observation.frame_id <= c.get('decision', {}).get('frame_id', -1):
                raise HTTPException(409, 'This frame result was already processed')
            if age(f['captured_dt']) > settings.stale_seconds:
                raise HTTPException(409, 'Inference result expired')
            adapter = c['adapter']; tracks = c['tracks']; rows = []
            # An expired identity is forgotten; stale history must not vote on a new person.
            for tid in list(tracks):
                if age(tracks[tid]['last_seen']) > settings.stale_seconds:
                    del tracks[tid]
                    adapter.last_state.pop(tid, None); adapter.missing_frames.pop(tid, None); adapter.last_seen.pop(tid, None)
            for person in observation.persons:
                detections = [d if isinstance(d, str) else {'class': d.class_name}
                              for d in person.associated_ppe if isinstance(d, str) or d.confidence >= .5]
                ppe = adapter.update(person.track_id, detections, observation.frame_id)
                t = tracks.setdefault(person.track_id, {'last_seen': now(), 'items': {}})
                t['last_seen'] = now()
                zone = 'production' if settings.zone_mode == 'production' else zone_at(*person.foot_point) if person.foot_point else 'outside'
                # Clear location-specific incidents on exit; do not call this PPE recovery.
                if t.get('zone') != zone:
                    for old_item, old in t['items'].items():
                        if old['active']: record(c, camera_id, person.track_id, old_item, 'ZONE_EXIT')
                    t['items'] = {}
                t['zone'] = zone
                evaluated = {i: ppe[i] for i in settings.required_ppe} if zone == 'production' else {}
                if settings.check_actions and zone == 'production':
                    evaluated['face_touch'] = 'not_detected' if person.face_touch is None else 'missing' if person.face_touch else 'present'
                    # Handwashing can only be carried within this camera/session/track.
                    evaluated['handwash'] = 'not_detected' if person.handwash_done is None else 'present' if person.handwash_done else 'missing'
                for item, v in evaluated.items():
                    item_state = t['items'].setdefault(item, {'votes': deque(maxlen=settings.voting_n), 'clear': 0, 'active': None})
                    item_state['votes'].append(v == 'missing' if v != 'not_detected' else None)
                    item_state['clear'] = item_state['clear'] + 1 if v == 'present' else 0
                    # Only a currently explicit absence can confirm a violation.
                    if v == 'missing' and sum(x is True for x in item_state['votes']) >= settings.voting_k and not item_state['active']:
                        item_state['active'] = record(c, camera_id, person.track_id, item, 'VIOLATION')
                    if item_state['active'] and item_state['clear'] >= settings.recovery_frames:
                        record(c, camera_id, person.track_id, item, 'RECOVERY')
                        item_state['active'] = None
                        item_state['votes'].clear()
                rows.append({'track_id': person.track_id, 'zone': zone, 'ppe': {i: ppe[i] for i in settings.required_ppe}, 'evaluated': evaluated, 'handwash_done': person.handwash_done, 'face_touch': person.face_touch})
            active = [{'track_id': tid, 'item': item, 'event_id': v['active']}
                      for tid, t in tracks.items() for item, v in t['items'].items() if v['active']]
            if active:
                status = 'VIOLATION'
            elif not rows:
                status = 'NO_PERSON'
            elif any(p['zone'] == 'outside' for p in rows):
                status = 'UNKNOWN'
            elif not any(p['evaluated'] for p in rows):
                status = 'MONITORING'
            elif any(v == 'not_detected' for p in rows for v in p['evaluated'].values()):
                status = 'UNKNOWN'
            elif any(v == 'missing' for p in rows for v in p['evaluated'].values()):
                status = 'PENDING'
            else:
                status = 'COMPLIANT'
            c['decision'] = dict(status=status, frame_id=observation.frame_id, captured_dt=f['captured_dt'],
                                 received_dt=now(), persons=rows, active_violations=active)
            c.pop('claimed_at', None)
            db.commit()
            return state(camera_id)

    def device_command(device_id):
        if device_id != 'ESP32_01':
            raise HTTPException(404, 'Device is not configured')
        states = [state(c) for c in cameras]
        alert = any(c['status'] == 'VIOLATION' for c in states)
        compliant = all(c['status'] == 'COMPLIANT' for c in states)
        overall = 'VIOLATION' if alert else 'COMPLIANT' if compliant else 'UNKNOWN'
        # Include incident IDs so a later, identical-looking violation is a new command.
        identity = [(c['camera_id'], c['session_id'], c['status'], c['active_violations']) for c in states]
        command_id = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:32]
        row = db.execute('SELECT acknowledged_command FROM device_state WHERE device_id=?', (device_id,)).fetchone()
        acknowledged = {r[0] for r in db.execute('SELECT event_id FROM acknowledged_incidents WHERE device_id=?', (device_id,))}
        active_ids = {v['event_id'] for c in states for v in c['active_violations']}
        return dict(command_id=command_id, device_id=device_id, status=overall, alert=alert,
                    red_led=alert, green_led=compliant, amber_led=not alert and not compliant,
                    buzzer=alert and bool(active_ids - acknowledged), buzzer_duration_ms=1000,
                    valid_for_seconds=settings.stale_seconds, cameras=states, issued_at=now().isoformat())

    @app.get('/v1/devices/{device_id}/command', dependencies=secured)
    def poll_command(device_id: str):
        with lock:
            cmd = device_command(device_id)
            db.execute('INSERT INTO device_state(device_id,last_seen) VALUES (?,?) ON CONFLICT(device_id) DO UPDATE SET last_seen=excluded.last_seen', (device_id, now().isoformat()))
            db.commit()
            return cmd

    @app.post('/v1/devices/{device_id}/ack', dependencies=secured)
    def acknowledge(device_id: str, ack: Ack):
        with lock:
            if device_command(device_id)['command_id'] != ack.command_id:
                raise HTTPException(409, 'Command changed; poll again')
            for c in cameras:
                for incident in state(c)['active_violations']:
                    db.execute('INSERT OR IGNORE INTO acknowledged_incidents VALUES (?,?)', (device_id, incident['event_id']))
            db.execute('INSERT INTO device_state VALUES (?,?,?) ON CONFLICT(device_id) DO UPDATE SET last_seen=excluded.last_seen, acknowledged_command=excluded.acknowledged_command',
                       (device_id, now().isoformat(), ack.command_id))
            db.commit()
            return {'acknowledged': ack.command_id}

    @app.get('/v1/devices/{device_id}', dependencies=secured)
    def device_status(device_id: str):
        with lock:
            if device_id != 'ESP32_01':
                raise HTTPException(404, 'Device is not configured')
            row = db.execute('SELECT * FROM device_state WHERE device_id=?', (device_id,)).fetchone()
            last_seen = datetime.fromisoformat(row['last_seen']) if row else None
            return {'device_id': device_id, 'online': bool(last_seen and age(last_seen) <= settings.stale_seconds),
                    'last_seen': row['last_seen'] if row else None}

    @app.get('/v1/events', dependencies=secured)
    def events(camera_id: str | None = None, limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
        with lock:
            if camera_id is not None:
                camera(camera_id)
            sql = 'SELECT event_id,type,camera_id,session_id,frame_id,captured_at,recorded_at,track_id,item FROM events'
            args = []
            if camera_id:
                sql += ' WHERE camera_id=?'; args.append(camera_id)
            sql += ' ORDER BY recorded_at DESC, rowid DESC LIMIT ? OFFSET ?'
            result = []
            for row in db.execute(sql, (*args, limit, offset)).fetchall():
                value = dict(row); value['missing_items'] = [value.pop('item')]
                value['snapshot_url'] = f"/v1/events/{value['event_id']}/snapshot"
                result.append(value)
            return {'events': result, 'limit': limit, 'offset': offset}

    @app.get('/v1/events/{event_id}/snapshot', dependencies=secured)
    def snapshot(event_id: uuid.UUID):
        with lock:
            row = db.execute('SELECT snapshot FROM events WHERE event_id=?', (str(event_id),)).fetchone()
            if not row:
                raise HTTPException(404, 'Event not found')
            return Response(row['snapshot'], media_type='image/jpeg', headers={'Cache-Control': 'no-store'})

    return app

