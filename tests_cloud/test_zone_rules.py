"""Zone geometry and per-zone rules end to end through the backend API."""
from datetime import datetime, timezone, timedelta
import json
import pytest
from fastapi.testclient import TestClient
from cloud_backend.app import Settings, create_app
from cloud_backend.zone_policy import HandwashTimer, ZoneConfig, load_zone_config, normalize_point, point_in_polygon, select_ppe
from test_backend import KEY, HEADERS, jpeg

PRESENT = [{'class': 'hairnet', 'confidence': .8}, {'class': 'maskon', 'confidence': .8}, {'class': 'glove', 'confidence': .8}]
NO_MASK = [{'class': 'hairnet', 'confidence': .8}, {'class': 'maskoff', 'confidence': .8}, {'class': 'glove', 'confidence': .8}]
CORRIDOR, SINK = [0.2, 0.9], [0.8, 0.9]


class Rig:
    """Real default layout (CAM_01 corridor + handwash region, CAM_02 production)."""

    def __init__(self, tmp_path, **kw):
        self.clock = [datetime.now(timezone.utc)]
        self.settings = Settings(api_key=KEY, data_dir=tmp_path, **kw)
        self.client = TestClient(create_app(self.settings, now=lambda: self.clock[0]))
        self.client.__enter__()
        self.sid = {}; self.n = {}

    def start(self, cam):
        self.sid[cam] = self.client.post(f'/v1/cameras/{cam}/sessions', headers=HEADERS).json()['session_id']
        self.n[cam] = 0
        return self.sid[cam]

    def send(self, cam, persons, size=(64, 64), content_box=None, expect=200):
        if cam not in self.sid:
            self.start(cam)
        self.n[cam] += 1
        data = {'session_id': self.sid[cam], 'frame_id': self.n[cam], 'captured_at': self.clock[0].isoformat()}
        if content_box:
            data['content_box'] = content_box
        r = self.client.post(f'/v1/cameras/{cam}/frames', headers=HEADERS, data=data, files={'image': ('a.jpg', jpeg('gray', size), 'image/jpeg')})
        assert r.status_code == 201, r.text
        r = self.client.post(f'/v1/cameras/{cam}/observations', headers=HEADERS,
                             json={'session_id': self.sid[cam], 'frame_id': self.n[cam], 'persons': persons})
        assert r.status_code == expect, r.text
        return r.json()

    def events(self):
        return self.client.get('/v1/events', headers=HEADERS).json()['events']

    def command(self):
        return self.client.get('/v1/devices/ESP32_01/command', headers=HEADERS).json()


@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)
    yield r
    r.client.__exit__(None, None, None)


def person(point=None, ppe=(), tid=1, **extra):
    return {'track_id': tid, 'foot_point_norm': point, 'associated_ppe': list(ppe), **extra}


def test_geometry_is_per_camera_not_thirds():
    z = load_zone_config()
    assert z.zone_for('CAM_01', (0.2, 0.9)) == 'corridor'
    assert z.zone_for('CAM_01', (0.8, 0.9)) == 'handwash'
    # CAM_02 is one production view regardless of position, even with no foot point.
    assert {z.zone_for('CAM_02', p) for p in [(0.05, .5), (0.5, .5), (0.95, .5), None]} == {'production'}
    assert z.zone_for('CAM_01', None) == 'outside'
    assert z.zone_for('CAM_03', (0.5, 0.5)) == 'outside'
    assert point_in_polygon((0.5, 0.5), [(0.5, 0), (1, 0), (1, 1), (0.5, 1)])  # on edge counts
    assert not point_in_polygon((0.3, 0.5), [(0.4, 0.2), (0.9, 0.2), (0.6, 0.9)])


def test_letterbox_content_box_maps_to_real_picture():
    # Portrait phone letterboxed into 640x360: picture occupies x 219..421.
    box = (219, 0, 421, 360)
    assert normalize_point(320, 360, 640, 360, box) == pytest.approx((0.5, 1.0), abs=.01)
    assert normalize_point(100, 200, 640, 360, box) is None  # on a black bar
    # The old fixed x>320 rule would call x=330 "sink"; normalised it is 55% across the picture.
    assert normalize_point(330, 300, 640, 360, box)[0] == pytest.approx(0.55, abs=.01)


def test_config_validation_and_legacy_roi(monkeypatch):
    with pytest.raises(ValueError, match='unknown PPE'):
        ZoneConfig.from_dict({'cameras': {'C': {'default_zone': 'p'}}, 'zones': {'p': {'required_ppe': ['goggles']}}})
    with pytest.raises(ValueError, match='polygon'):
        ZoneConfig.from_dict({'cameras': {'C': {'default_zone': 'p', 'regions': [{'zone': 'p', 'polygon': [[0, 0], [2, 0], [0, 1]]}]}}, 'zones': {'p': {}}})
    with pytest.raises(ValueError, match='No zone layout'):
        Settings(api_key=KEY, data_dir='.', cameras=('CAM_09',))
    monkeypatch.setenv('PPE_HANDWASH_ROI', json.dumps([0.6, 0.1, 0.9, 0.8]))
    z = load_zone_config()
    assert z.zone_for('CAM_01', (0.55, 0.5)) == 'corridor' and z.zone_for('CAM_01', (0.7, 0.5)) == 'handwash'


def test_cam01_never_needs_ppe_and_cam02_does(rig):
    r = rig.send('CAM_01', [person(CORRIDOR)])
    assert r['status'] == 'MONITORING' and r['persons'][0]['zone'] == 'corridor'
    assert r['persons'][0]['required'] == []
    for _ in range(3):
        r = rig.send('CAM_02', [person(None, NO_MASK)])
    assert r['status'] == 'VIOLATION' and r['active_violations'][0]['item'] == 'mask'
    assert r['persons'][0]['zone'] == 'production'


def test_legacy_pixel_foot_point_uses_frame_size_and_letterbox(rig):
    # 640x360 letterboxed upload; foot at x=330px is the corridor (55% across) under
    # the default layout's 50% split only if mapped against the picture, not the canvas.
    r = rig.send('CAM_01', [{'track_id': 1, 'foot_point': [300, 350], 'associated_ppe': []}],
                 size=(640, 360), content_box='219,0,421,360')
    assert r['persons'][0]['zone'] == 'corridor'  # 40% across the picture
    r = rig.send('CAM_01', [{'track_id': 2, 'foot_point': [100, 350], 'associated_ppe': []}],
                 size=(640, 360), content_box='219,0,421,360')
    assert r['persons'][0]['zone'] == 'outside' and r['status'] == 'UNKNOWN'


def test_low_confidence_or_conflicting_ppe_is_unknown_not_missing(rig):
    weak = [{'class': 'hairnet', 'confidence': .8}, {'class': 'maskoff', 'confidence': .3}, {'class': 'glove', 'confidence': .8}]
    for _ in range(4):
        r = rig.send('CAM_02', [person(None, weak)])
    assert r['status'] == 'UNKNOWN' and not r['active_violations']
    clash = PRESENT + [{'class': 'maskoff', 'confidence': .76}]
    assert select_ppe(clash) == [{'class': 'hairnet', 'confidence': .8}, {'class': 'glove', 'confidence': .8}]
    for _ in range(4):
        r = rig.send('CAM_02', [person(None, clash)])
    assert r['status'] == 'UNKNOWN' and not r['active_violations']


def test_unreliable_person_is_unknown_even_with_positive_boxes(rig):
    r = rig.send('CAM_02', [person(None, PRESENT, ppe_reliable=False)])
    assert r['status'] == 'UNKNOWN'


def test_zone_flicker_does_not_reset_or_close_incident(rig, tmp_path):
    settings_rig = Rig(tmp_path / 'b', check_actions=True)
    try:
        for _ in range(3):
            r = settings_rig.send('CAM_01', [person(CORRIDOR, handwash_status='incomplete')])
        assert r['status'] == 'VIOLATION'
        r = settings_rig.send('CAM_01', [person(SINK, handwash_status='incomplete')])  # one jittery frame
        assert r['status'] == 'VIOLATION' and r['persons'][0]['zone'] == 'corridor'
        assert [e['type'] for e in settings_rig.events()] == ['VIOLATION']
    finally:
        settings_rig.client.__exit__(None, None, None)


def test_track_loss_and_session_reset_close_incidents_without_recovery(rig):
    for _ in range(3):
        rig.send('CAM_02', [person(None, NO_MASK)])
    rig.clock[0] += timedelta(seconds=14)
    r = rig.send('CAM_02', [person(None, PRESENT, tid=9)])  # track 1 unseen 14 s: still open
    assert r['active_violations']
    rig.clock[0] += timedelta(seconds=2)
    r = rig.send('CAM_02', [person(None, PRESENT, tid=9)])
    assert not r['active_violations']
    for _ in range(3):
        rig.send('CAM_02', [person(None, NO_MASK, tid=9)])
    rig.start('CAM_02')
    types = [e['type'] for e in reversed(rig.events())]
    assert types == ['VIOLATION', 'TRACK_LOST', 'VIOLATION', 'SESSION_RESET']
    assert 'RECOVERY' not in types
    assert all(e['snapshot_url'] is None for e in rig.events() if e['type'] in ('TRACK_LOST', 'SESSION_RESET'))


def test_action_fields_are_consumed_when_enforced(tmp_path):
    r = Rig(tmp_path, check_actions=True)
    try:
        for _ in range(3):
            out = r.send('CAM_02', [person(None, PRESENT, face_touch=True)])
        assert out['status'] == 'VIOLATION' and out['active_violations'][0]['item'] == 'face_touch'
        for _ in range(3):
            out = r.send('CAM_02', [person(None, PRESENT, face_touch=False)])
        assert out['status'] == 'COMPLIANT'
        # Unknown face-touch evidence neither violates nor blocks PPE compliance.
        assert r.send('CAM_02', [person(None, PRESENT, face_touch=None)])['status'] == 'COMPLIANT'
        for _ in range(3):
            out = r.send('CAM_01', [person(CORRIDOR, handwash_status='incomplete', handwash_seconds=6)])
        assert out['active_violations'][0]['item'] == 'handwash'
        for _ in range(3):
            out = r.send('CAM_01', [person(CORRIDOR, handwash_status='completed', handwash_seconds=20)])
        assert out['status'] == 'MONITORING'
        assert [e['type'] for e in r.events()][:2] == ['RECOVERY', 'VIOLATION']
        # Handwash is never evaluated on the production camera: no cross-camera identity.
        out = r.send('CAM_02', [person(None, PRESENT, tid=1, handwash_status='incomplete')])
        assert 'handwash' not in out['persons'][0]['evaluated']
    finally:
        r.client.__exit__(None, None, None)


def test_action_fields_reported_but_not_alerted_when_not_enforced(rig):
    for _ in range(3):
        out = rig.send('CAM_02', [person(None, PRESENT, face_touch=True)])
    assert out['status'] == 'COMPLIANT' and out['actions_enforced'] is False
    assert out['persons'][0]['face_touch'] is True


def test_esp32_needs_fresh_compliance_and_never_greens_unknown(rig):
    rig.send('CAM_01', [])
    for _ in range(3):
        rig.send('CAM_02', [person(None, PRESENT)])
    cmd = rig.command()
    assert cmd['green_led'] and cmd['status'] == 'COMPLIANT'  # empty corridor + compliant production
    rig.send('CAM_01', [person(None)])  # person but no foot point: cannot be placed
    assert rig.command()['amber_led']
    rig.send('CAM_01', [person(CORRIDOR)])
    assert rig.command()['green_led']
    rig.clock[0] += timedelta(seconds=16)  # both cameras silent: stale
    cmd = rig.command()
    assert cmd['status'] == 'UNKNOWN' and not cmd['green_led'] and not cmd['red_led']


def test_long_poll_claim_returns_newer_frame_or_204(rig):
    c = rig.client; sid = rig.start('CAM_01')
    r = c.get('/v1/cameras/CAM_01/frame', params={'after': -1, 'wait': .2, 'session': sid}, headers=HEADERS)
    assert r.status_code == 204
    c.post('/v1/cameras/CAM_01/frames', headers=HEADERS, data={'session_id': sid, 'frame_id': 4, 'captured_at': rig.clock[0].isoformat()},
           files={'image': ('a.jpg', jpeg('gray', (48, 64)), 'image/jpeg')})
    r = c.get('/v1/cameras/CAM_01/frame', params={'after': -1, 'wait': 2, 'session': sid}, headers={**HEADERS, 'X-Claim-Frame': 'true'})
    assert r.status_code == 200 and r.headers['X-Frame-ID'] == '4' and r.headers['X-Content-Box'] == '0,0,48,64'
    assert c.get('/v1/cameras/CAM_01/frame', params={'after': 4, 'wait': .1, 'session': sid}, headers=HEADERS).status_code == 204
    # A worker still holding an old session's frame number gets the new session's frame.
    new = rig.start('CAM_01')
    c.post('/v1/cameras/CAM_01/frames', headers=HEADERS, data={'session_id': new, 'frame_id': 0, 'captured_at': rig.clock[0].isoformat()},
           files={'image': ('a.jpg', jpeg(), 'image/jpeg')})
    r = c.get('/v1/cameras/CAM_01/frame', params={'after': 4, 'wait': .5, 'session': sid}, headers=HEADERS)
    assert r.status_code == 200 and r.headers['X-Session-ID'] == new


def test_handwash_timer_counts_only_observed_active_intervals():
    t = HandwashTimer(required_seconds=20, max_gap=1.0)
    for i in range(11):
        t.update(100 + i * 0.5, True)       # 5 s of continuous evidence at 2 FPS
    assert t.elapsed == pytest.approx(5.0)
    t.update(110, True)                      # 5 s gap (tracking loss / dropped frames): not counted
    t.update(110.5, False)                   # pause
    t.update(111, True)
    t.update(110.9, True)                    # out-of-order (stale) sample ignored
    assert t.elapsed == pytest.approx(5.0)
    assert t.gaps == 1


@pytest.mark.parametrize('fps', [2, 4, 8])
def test_handwash_duration_is_independent_of_processing_rate(fps):
    t = HandwashTimer(required_seconds=20, max_gap=1.0)
    ts = 0.0
    while ts <= 19.0:
        t.update(ts, True); ts += 1 / fps
    assert not t.done
    while ts <= 20.6:
        t.update(ts, True); ts += 1 / fps
    assert t.done


def test_existing_event_database_is_migrated_not_lost(tmp_path):
    import sqlite3
    db = sqlite3.connect(tmp_path / 'ppe_cloud.db')
    db.execute('''CREATE TABLE events (event_id TEXT PRIMARY KEY, type TEXT NOT NULL, camera_id TEXT NOT NULL,
        session_id TEXT NOT NULL, frame_id INTEGER NOT NULL, captured_at TEXT NOT NULL, recorded_at TEXT NOT NULL,
        track_id INTEGER NOT NULL, item TEXT NOT NULL, snapshot BLOB NOT NULL)''')
    db.execute("INSERT INTO events VALUES ('5f0c6a3e-0b7e-4c1e-9d1a-3f2b8c9d0e11','VIOLATION','CAM_01','s',3,'2026-10-30T00:00:00+00:00','2026-10-30T00:00:01+00:00',1,'gloves',?)", (jpeg(),))
    db.commit(); db.close()
    r = Rig(tmp_path)
    try:
        events = r.events()
        assert events[0]['event_id'] == '5f0c6a3e-0b7e-4c1e-9d1a-3f2b8c9d0e11' and events[0]['zone'] is None
        assert r.client.get(events[0]['snapshot_url'], headers=HEADERS).content == jpeg()
        for _ in range(3):
            r.send('CAM_02', [person(None, NO_MASK)])
        assert [e['zone'] for e in r.events()][:2] == ['production', None]
    finally:
        r.client.__exit__(None, None, None)
