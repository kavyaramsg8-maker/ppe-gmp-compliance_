"""Detector logic and a smoke test with the real weights. Run in the CV environment (.venv)."""
from collections import deque
from pathlib import Path

import pytest

np = pytest.importorskip('numpy')
cv2 = pytest.importorskip('cv2')
pytest.importorskip('ultralytics')
pytest.importorskip('mediapipe')
import ai_detector as d  # noqa: E402
from cloud_backend.zone_policy import HandwashTimer, load_zone_config  # noqa: E402

L = d.L
STATION = [[(0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0)]]


def to_norm(x, y):
    return (x / 640, y / 360) if 0 <= x <= 640 and 0 <= y <= 360 else None


def landmarks(lw, rw, vis=0.9, nose=(500, 60)):
    lms = {i: (np.array([0.0, 0.0]), 0.0) for i in range(33)}
    lms[L.LEFT_SHOULDER] = (np.array([470.0, 120.0]), vis)
    lms[L.RIGHT_SHOULDER] = (np.array([530.0, 120.0]), vis)
    lms[L.NOSE] = (np.array(nose, float), vis)
    lms[L.LEFT_WRIST] = (np.array(lw, float), vis)
    lms[L.RIGHT_WRIST] = (np.array(rw, float), vis)
    return lms


def test_ppe_goes_to_one_plausible_person():
    a = {'box': [100, 0, 300, 360], 'candidates': []}
    b = {'box': [200, 0, 400, 360], 'candidates': []}
    cands = [{'box': [210, 10, 260, 50], 'class': 'hairnet', 'confidence': .8},     # head, inside both boxes
             {'box': [210, 250, 260, 290], 'class': 'hairnet', 'confidence': .7},   # "hairnet" at hand height
             {'box': [210, 40, 260, 80], 'class': 'no_glove', 'confidence': .7},    # bare hand at the head is possible
             {'box': [600, 10, 630, 40], 'class': 'maskon', 'confidence': .9}]      # nobody there
    stats = d.assign_ppe(cands, [a, b])
    owners = [p for p in (a, b) if any(c['class'] == 'hairnet' for c in p['candidates'])]
    assert len(owners) == 1  # never duplicated onto every overlapping person
    assert sum(c['box'][1] == 250 for p in (a, b) for c in p['candidates']) == 0
    assert any(c['class'] == 'no_glove' for p in (a, b) for c in p['candidates'])
    assert stats == {'unassigned': 1, 'implausible': 1}


def test_face_touch_is_unknown_without_landmarks():
    assert d.check_face_touch(None) is None
    assert d.check_face_touch(landmarks((505, 70), (560, 250), nose=(500, 60))) is True
    assert d.check_face_touch(landmarks((420, 250), (560, 250))) is False
    assert d.check_face_touch(landmarks((505, 70), (560, 250), vis=0.2)) is None


def test_rubbing_motion_evidence_needs_both_wrists_in_station_and_reversal():
    h = deque()
    xs = [0, 12, 0, 12, 0]
    out = [d.handwash_evidence(landmarks((480 + x, 250), (520 + x, 250)), h, t * 0.25, STATION, to_norm) for t, x in enumerate(xs)]
    assert out[-1] is True
    h = deque()
    still = [d.handwash_evidence(landmarks((480, 250), (520, 250)), h, t * 0.25, STATION, to_norm) for t in range(5)]
    assert still[-1] is False  # hands together at the sink but not moving
    assert d.handwash_evidence(landmarks((200, 250), (520, 250)), deque(), 0, STATION, to_norm) is False  # one wrist outside
    h = deque()  # same rubbing motion with hands raised to the head (e.g. removing a hairnet)
    raised = [d.handwash_evidence(landmarks((480 + x, 70), (520 + x, 70)), h, t * 0.25, STATION, to_norm) for t, x in enumerate(xs)]
    assert not any(raised)
    assert d.handwash_evidence(landmarks((480, 250), (520, 250), vis=0.3), deque(), 0, STATION, to_norm) is None


def test_handwash_states_pause_leave_early_and_complete():
    st = {'timer': HandwashTimer(10, 1.0)}
    assert d.update_handwash(st, True, None, 0.0) == 'unknown'
    assert d.update_handwash(st, True, False, 0.5) == 'waiting'
    for i in range(2, 10):
        status = d.update_handwash(st, True, True, i * 0.5)
    assert status == 'washing' and st['timer'].elapsed == pytest.approx(3.5)
    assert d.update_handwash(st, True, False, 5.0) == 'paused'
    # One frame of foot-point jitter outside the sink is not leaving.
    assert d.update_handwash(st, False, None, 5.5) == 'not_at_station'
    assert d.update_handwash(st, True, True, 6.0) == 'washing'
    assert st['timer'].elapsed == pytest.approx(3.5)  # the gap out of the station added nothing
    d.update_handwash(st, False, None, 6.5)
    assert d.update_handwash(st, False, None, 7.0) == 'incomplete'
    for i in range(16, 40):
        status = d.update_handwash(st, True, True, i * 0.5)
    assert status == 'completed'
    assert d.update_handwash(st, False, None, 21.0) == 'completed'


@pytest.fixture(scope='module')
def frames():
    cap = cv2.VideoCapture(str(Path(d.ROOT) / 'review_video.mp4'))
    out = []
    for i in range(0, 60, 6):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, f = cap.read()
        if ok:
            out.append(f)
    if not out:
        pytest.skip('review_video.mp4 not available')
    return out


def enc(img):
    return cv2.imencode('.jpg', img)[1].tobytes()


def test_real_model_output_contract_both_cameras(frames, tmp_path, monkeypatch):
    monkeypatch.setenv('PPE_MONITOR_DIR', str(tmp_path))
    d.configure(load_zone_config())
    seen = []
    for i, f in enumerate(frames):
        for cam in ('CAM_01', 'CAM_02'):
            people = d.detect(enc(f), cam, f'test-{cam}', frame_id=i, captured_at=1000 + i * 0.25)
            for p in people:
                seen.append(p)
                assert 0 <= p['foot_point_norm'][0] <= 1 and 0 <= p['foot_point_norm'][1] <= 1
                items = [d.PPE_CLASSES[x['class']][0] for x in p['associated_ppe']]
                assert len(items) == len(set(items))  # one decision per item
                assert all(x['confidence'] >= d.PPE_CONFIDENCE for x in p['associated_ppe'])
                if cam == 'CAM_02':
                    assert p['zone'] == 'production' and p['handwash_status'] is None
                else:
                    assert p['zone'] in ('corridor', 'handwash') and p['handwash_status'] is not None
    assert seen, 'person detector found nobody in review_video.mp4'
    assert d._sessions['CAM_01']['person_model'] is not d._sessions['CAM_02']['person_model']
    meta = (tmp_path / 'live_CAM_02.frame.json').read_text()
    assert '"frame_id": %d' % (len(frames) - 1) in meta


def test_letterboxed_and_portrait_inputs_map_to_the_same_zone(frames, tmp_path, monkeypatch):
    monkeypatch.setenv('PPE_MONITOR_DIR', str(tmp_path))
    f = frames[1]
    plain = d.detect(enc(f), 'CAM_01', 'lb-a', frame_id=0)
    # Same picture letterboxed into a wider canvas, with its content box declared.
    canvas = np.zeros((360, 1000, 3), np.uint8); canvas[:, 180:820] = f
    boxed = d.detect(enc(canvas), 'CAM_01', 'lb-b', frame_id=0, content_box=(180, 0, 820, 360))
    assert [p['zone'] for p in plain] == [p['zone'] for p in boxed]
    for a, b in zip(plain, boxed):
        assert a['foot_point_norm'] == pytest.approx(b['foot_point_norm'], abs=0.03)
    # Portrait upload is processed at its own aspect ratio (no distortion, no crash).
    portrait = cv2.rotate(f, cv2.ROTATE_90_CLOCKWISE)
    d.detect(enc(portrait), 'CAM_01', 'pt', frame_id=0)
    assert d.get_metrics('CAM_01')['input'] == [360, 640]
