"""Headless PPE / person / pose detector called by cloud_backend.worker.

Produces model EVIDENCE per tracked person; the backend applies zone rules and
temporal confirmation to produce verdicts. Unknown evidence is reported as unknown.
"""
from collections import deque
from hashlib import sha256
import json
import os
from pathlib import Path
from threading import RLock
import time

import cv2
import mediapipe as mp
import numpy as np
import torch
from ultralytics import YOLO

from cloud_backend.zone_policy import (PPE_CLASSES, PPE_ITEMS, HandwashTimer, ZoneConfig, denormalize_polygon,
                                       load_zone_config, normalize_point, point_in_polygon, ppe_states, select_ppe)
from performance_tools import atomic_bytes, rss_mib

ROOT = Path(__file__).resolve().parent
_DEVICE = os.environ.get('PPE_DEVICE', 'auto')
if _DEVICE == 'auto':
    _DEVICE = 'cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
PPE_IMGSZ = int(os.environ.get('PPE_IMGSZ', '640'))
PERSON_IMGSZ = int(os.environ.get('PPE_PERSON_IMGSZ', '416'))
POSE_COMPLEXITY = int(os.environ.get('PPE_POSE_COMPLEXITY', '1'))
# The model call keeps weaker candidates so contradictions are visible; decisions use PPE_CONFIDENCE.
PPE_CANDIDATE_CONF = float(os.environ.get('PPE_CANDIDATE_CONF', '0.25'))
PPE_CONFIDENCE = float(os.environ.get('PPE_CONFIDENCE', '0.50'))
PPE_CONFLICT_MARGIN = float(os.environ.get('PPE_CONFLICT_MARGIN', '0.10'))
PERSON_CONF = float(os.environ.get('PPE_PERSON_CONF', '0.40'))
# Persons shorter than this (pixels in the received image) give no usable PPE evidence.
MIN_PERSON_PX = int(os.environ.get('PPE_MIN_PERSON_PX', '120'))
if PPE_IMGSZ < 160 or PERSON_IMGSZ < 160:
    raise ValueError('Image sizes must be >=160')
if not 0 < PPE_CANDIDATE_CONF <= PPE_CONFIDENCE <= 1:
    raise ValueError('Need 0 < PPE_CANDIDATE_CONF <= PPE_CONFIDENCE <= 1')

# Vertical band (fraction of person-box height, 0=top) where an item can plausibly be.
# Measured on review/demo clips: hairnet centres p90 <= 0.13, mask centres p90 <= 0.29.
HEAD_BANDS = {'hairnet': 0.35, 'mask': 0.45}
MIN_CONTAINMENT = 0.6

_detector_lock = RLock()
_sessions = {}
_last_metrics = {}
_zones = load_zone_config()

ppe_model = YOLO(str(ROOT / 'best.pt'))
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=POSE_COMPLEXITY)
L = mp_pose.PoseLandmark


def configure(zone_config):
    """Install the backend's zone layout (dict from /camera-config)."""
    global _zones
    with _detector_lock:
        _zones = zone_config if isinstance(zone_config, ZoneConfig) else ZoneConfig.from_dict(zone_config)


def _sync():
    if _DEVICE.startswith('cuda'):
        torch.cuda.synchronize()
    elif _DEVICE == 'mps':
        torch.mps.synchronize()


def _session(camera_id, session_id):
    """A new session resets this camera's tracker IDs, wash timers and motion history."""
    old = _sessions.get(camera_id)
    if old is None or old['session_id'] != session_id:
        if old is None and len(_sessions) >= 8:
            raise ValueError('Too many cameras')
        _sessions[camera_id] = {'session_id': session_id, 'person_model': YOLO(str(ROOT / 'yolov8n.pt')), 'people': {}}
    return _sessions[camera_id]


def get_metrics(camera_id):
    return dict(_last_metrics.get(camera_id, {}))


# ---------------------------------------------------------------- evidence helpers

def _landmarks(result, crop_shape, offset):
    """Pose landmarks as full-image pixel coordinates with visibility, or None."""
    if result is None or not result.pose_landmarks:
        return None
    h, w = crop_shape[:2]
    ox, oy = offset
    return {idx: (np.array([lm.x * w + ox, lm.y * h + oy]), lm.visibility)
            for idx, lm in enumerate(result.pose_landmarks.landmark)}


def check_face_touch(lms, tau=0.4):
    """True/False only when nose, both wrists and both shoulders are visible; else None."""
    needed = (L.NOSE, L.LEFT_WRIST, L.RIGHT_WRIST, L.LEFT_SHOULDER, L.RIGHT_SHOULDER)
    if lms is None or any(lms[i][1] < 0.5 for i in needed):
        return None
    shoulder = np.linalg.norm(lms[L.LEFT_SHOULDER][0] - lms[L.RIGHT_SHOULDER][0])
    if shoulder < 5:
        return None
    nose = lms[L.NOSE][0]
    return bool(min(np.linalg.norm(nose - lms[L.LEFT_WRIST][0]), np.linalg.norm(nose - lms[L.RIGHT_WRIST][0])) / shoulder < tau)


def handwash_evidence(lms, history, timestamp, stations, to_norm):
    """Rubbing-motion evidence: True, False, or None when it cannot be judged.

    Requires both wrists visible inside a configured station polygon and below the
    shoulders (hands raised to the head, e.g. adjusting a hairnet, are not washing),
    hands within 1.2 shoulder-widths, and back-and-forth motion within the last 2 s
    (>= 3 samples). This is motion evidence at a sink, not verification of soap,
    water or technique.
    """
    if lms is None:
        return None
    lw, rw = lms[L.LEFT_WRIST], lms[L.RIGHT_WRIST]
    if min(lw[1], rw[1], lms[L.LEFT_SHOULDER][1], lms[L.RIGHT_SHOULDER][1]) < 0.5:
        return None
    shoulder = float(np.linalg.norm(lms[L.LEFT_SHOULDER][0] - lms[L.RIGHT_SHOULDER][0]))
    if shoulder < 5:
        return None
    shoulder_y = max(lms[L.LEFT_SHOULDER][0][1], lms[L.RIGHT_SHOULDER][0][1])
    points = [to_norm(*wrist[0]) for wrist in (lw, rw)]
    if (any(p is None for p in points) or not all(any(point_in_polygon(p, s) for s in stations) for p in points)
            or min(lw[0][1], rw[0][1]) <= shoulder_y):
        history.clear()
        return False
    left, right = lw[0], rw[0]
    history.append((timestamp, float((left[0] + right[0]) / 2 / shoulder), float((left[0] - right[0]) / shoulder)))
    while history and timestamp - history[0][0] > 2.0:
        history.popleft()
    if np.linalg.norm(left - right) / shoulder >= 1.2 or len(history) < 3:
        return False
    for series in (1, 2):
        delta = [history[i][series] - history[i - 1][series] for i in range(1, len(history))]
        if any(abs(d) > 0.02 for d in delta) and any(a * b < 0 for a, b in zip(delta, delta[1:])):
            return True
    return False


def update_handwash(state, station, evidence, timestamp):
    """Per-track handwash status driven by capture timestamps (see HandwashTimer)."""
    timer = state['timer']
    if station:
        state['away'] = 0
        state['visited'] = True
        timer.update(timestamp, bool(evidence))
        if timer.done:
            state['incomplete'] = False
            return 'completed'
        if evidence:
            return 'washing'
        if evidence is None and timer.elapsed == 0:
            return 'unknown'
        return 'paused' if timer.elapsed > 0 else 'waiting'
    timer.update(timestamp, False)  # leaving breaks continuity: nothing bridges the absence
    state['away'] = state.get('away', 0) + 1
    # Two consecutive frames away from the sink, so boundary jitter is not "leaving".
    if state.get('visited') and state['away'] >= 2 and 0 < timer.elapsed < timer.required:
        state['incomplete'] = True
    if timer.done:
        return 'completed'
    return 'incomplete' if state.get('incomplete') else 'not_at_station'


def assign_ppe(candidates, people):
    """Give each PPE box to at most one plausible person (never to every overlapping box)."""
    stats = {'unassigned': 0, 'implausible': 0}
    for d in candidates:
        b = d['box']
        area = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        if not area:
            continue
        item = PPE_CLASSES[d['class']][0]
        matches = []
        implausible = False
        for p in people:
            x1, y1, x2, y2 = p['box']
            inter = max(0, min(b[2], x2) - max(b[0], x1)) * max(0, min(b[3], y2) - max(b[1], y1))
            if inter / area < MIN_CONTAINMENT:
                continue
            centre = ((b[1] + b[3]) / 2 - y1) / max(1, y2 - y1)
            if item in HEAD_BANDS and centre > HEAD_BANDS[item]:
                implausible = True
                continue
            matches.append((inter / area, -(x2 - x1) * (y2 - y1), p))
        if matches:
            max(matches, key=lambda m: m[:2])[2]['candidates'].append(d)
        else:
            stats['implausible' if implausible else 'unassigned'] += 1
    return stats


# ---------------------------------------------------------------- drawing

COLORS = {'present': (70, 190, 70), 'missing': (60, 60, 230), 'unknown': (0, 190, 240), 'monitor': (200, 150, 60)}


def _draw(frame, camera_id, frame_id, people, candidates, content_box):
    h, w = frame.shape[:2]
    layout = _zones.cameras.get(camera_id)
    if layout:
        for region in layout.regions:
            pts = np.array(denormalize_polygon(region.polygon, w, h, content_box), np.int32)
            cv2.polylines(frame, [pts], True, (220, 160, 60), 2)
            cv2.putText(frame, region.zone.upper(), tuple(int(v) for v in pts.min(axis=0) + [6, 40]),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (230, 180, 80), 2)
        l, t, r, b = map(int, content_box)
        if (l, t, r, b) != (0, 0, w, h):
            cv2.rectangle(frame, (l, t), (r - 1, b - 1), (120, 120, 120), 1)
        cv2.putText(frame, f'elsewhere: {layout.default_zone.upper()}', (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, .45, (230, 230, 230), 1)
    for d in candidates:
        x1, y1, x2, y2 = map(int, d['box'])
        colour = COLORS['present'] if PPE_CLASSES[d['class']][1] == 'present' else COLORS['missing']
        if d['confidence'] < PPE_CONFIDENCE:
            colour = (150, 150, 150)  # below the decision threshold: shown, not used
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 1)
        cv2.putText(frame, f"{d['class']} {d['confidence']:.2f}", (x1, max(10, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, .38, colour, 1)
    mark = {'present': 'ok', 'missing': 'NO', 'unknown': '?'}
    for p in people:
        x1, y1, x2, y2 = p['box']
        required = _zones.rules(p['zone']).required_ppe
        states = ppe_states(p['associated_ppe'])
        values = [states[i] for i in required]
        state = ('monitor' if not required else 'unknown' if not p['ppe_reliable'] or 'unknown' in values
                 else 'missing' if 'missing' in values else 'present')
        colour = COLORS[state]
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        lines = [f"ID {p['track_id']} {p['zone'].upper()}",
                 ' '.join(f'{i[0].upper()}:{mark[states[i]]}' for i in PPE_ITEMS) if p['ppe_reliable'] else 'PPE: too small to judge']
        if p['handwash_status'] not in (None, 'not_at_station'):
            lines.append(f"wash {p['handwash_seconds']:.1f}/{p['handwash_required_seconds']:.0f}s {p['handwash_status']}")
        if p['face_touch']:
            lines.append('FACE TOUCH')
        for i, text in enumerate(lines):
            y = max(34 + 15 * i, y1 - 6 - 15 * (len(lines) - 1 - i))
            cv2.putText(frame, text, (x1 + 2, y), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 3)
            cv2.putText(frame, text, (x1 + 2, y), cv2.FONT_HERSHEY_SIMPLEX, .45, colour, 1)
    cv2.rectangle(frame, (0, 0), (w, 20), (25, 25, 25), -1)
    cv2.putText(frame, f'{camera_id} frame {frame_id} | model evidence, not backend verdict', (6, 14),
                cv2.FONT_HERSHEY_SIMPLEX, .42, (235, 235, 235), 1)


# ---------------------------------------------------------------- main entry

def detect(frame_content, camera_id, session_id, frame_id=None, captured_at=None, content_box=None):
    # Shared PPE/pose models execute serially; network work may overlap.
    with _detector_lock:
        return _detect(frame_content, camera_id, session_id, frame_id, captured_at, content_box)


def _detect(frame_content, camera_id, session_id, frame_id, captured_at, content_box):
    started = time.perf_counter()
    # Capture time (phone clock) drives handwash timing, so processing delay is not wash time.
    timestamp = float(captured_at) if captured_at is not None else time.time()
    state = _session(camera_id, session_id)
    frame = cv2.imdecode(np.frombuffer(frame_content, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError('Invalid JPEG; no inference result submitted')
    h, w = frame.shape[:2]
    box = tuple(content_box) if content_box else (0, 0, w, h)
    if not (0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h):
        box = (0, 0, w, h)

    def to_norm(x, y):
        return normalize_point(x, y, w, h, box)

    stations = _zones.station_polygons(camera_id)

    _sync(); t = time.perf_counter()
    ppe_results = ppe_model(frame, conf=PPE_CANDIDATE_CONF, imgsz=PPE_IMGSZ, device=_DEVICE, verbose=False)[0]
    _sync(); ppe_ms = (time.perf_counter() - t) * 1000; t = time.perf_counter()
    person_results = state['person_model'].track(frame, classes=[0], conf=PERSON_CONF, imgsz=PERSON_IMGSZ, device=_DEVICE,
                                                 persist=True, tracker='bytetrack.yaml', verbose=False)[0]
    _sync(); person_ms = (time.perf_counter() - t) * 1000
    pose_ms = 0.0; pose_calls = 0

    candidates = []
    if ppe_results.boxes is not None:
        for b, c, s in zip(ppe_results.boxes.xyxy.cpu().numpy(), ppe_results.boxes.cls.cpu().numpy().astype(int),
                           ppe_results.boxes.conf.cpu().numpy()):
            name = ppe_model.names[int(c)]
            if name in PPE_CLASSES:
                candidates.append({'box': [float(v) for v in b], 'class': name, 'confidence': round(float(s), 3)})

    people = []
    if person_results.boxes is not None and person_results.boxes.id is not None:
        for b, tid in zip(person_results.boxes.xyxy.cpu().numpy(), person_results.boxes.id.cpu().numpy().astype(int)):
            x1, y1, x2, y2 = int(max(0, b[0])), int(max(0, b[1])), int(min(w, b[2])), int(min(h, b[3]))
            if x2 <= x1 or y2 <= y1:
                continue
            foot = to_norm((x1 + x2) / 2, y2)
            people.append({'track_id': int(tid), 'box': [x1, y1, x2, y2], 'foot_point': [(x1 + x2) / 2, float(y2)],
                           'foot_point_norm': list(foot) if foot else None, 'zone': _zones.zone_for(camera_id, foot),
                           'candidates': [], 'ppe_reliable': (y2 - y1) >= MIN_PERSON_PX})
    assignment = assign_ppe(candidates, people)

    required = _zones.handwash_seconds
    for p in people:
        chosen = select_ppe(p.pop('candidates'), PPE_CONFIDENCE, PPE_CONFLICT_MARGIN)
        p['associated_ppe'] = chosen if p['ppe_reliable'] else []
        x1, y1, x2, y2 = p['box']
        crop = frame[y1:y2, x1:x2]
        t = time.perf_counter()
        result = pose.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)) if crop.size else None
        pose_ms += (time.perf_counter() - t) * 1000; pose_calls += int(bool(crop.size))
        lms = _landmarks(result, crop.shape, (x1, y1))
        st = state['people'].get(p['track_id'])
        if st is None or timestamp - st['seen'] > _zones.forget_after_seconds:
            st = state['people'][p['track_id']] = {'timer': HandwashTimer(required, _zones.max_gap_seconds),
                                                   'history': deque(maxlen=16), 'seen': timestamp}
        st['seen'] = timestamp
        if stations:
            evidence = handwash_evidence(lms, st['history'], timestamp, stations, to_norm)
            status = update_handwash(st, _zones.rules(p['zone']).handwash_station, evidence, timestamp)
        else:
            evidence = status = None
        p.update(face_touch=check_face_touch(lms), handwash_status=status, handwash_evidence=evidence,
                 handwash_seconds=round(st['timer'].elapsed, 2) if stations else None, handwash_required_seconds=required)
    state['people'] = {k: v for k, v in state['people'].items() if timestamp - v['seen'] <= _zones.forget_after_seconds}

    _draw(frame, camera_id, frame_id, people, candidates, box)
    folder = Path(os.environ.get('PPE_MONITOR_DIR', str(ROOT))); folder.mkdir(parents=True, exist_ok=True)
    ok, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise RuntimeError('Preview JPEG encoding failed')
    data = jpeg.tobytes()
    atomic_bytes(folder / f'live_{camera_id}.jpg', data)
    # Lets the monitor tie this exact image to its frame and capture time.
    atomic_bytes(folder / f'live_{camera_id}.frame.json', json.dumps({
        'camera_id': camera_id, 'session_id': session_id, 'frame_id': frame_id, 'captured_at': captured_at,
        'processed_at': time.time(), 'jpeg_sha256': sha256(data).hexdigest()}).encode())
    metrics = {'detector_ms': (time.perf_counter() - started) * 1000, 'ppe_ms': ppe_ms, 'person_ms': person_ms, 'pose_ms': pose_ms,
               'pose_calls': pose_calls, 'people': len(people), 'ppe_candidates': len(candidates),
               'ppe_unassigned': assignment['unassigned'], 'ppe_implausible': assignment['implausible'],
               'device': _DEVICE, 'ppe_imgsz': PPE_IMGSZ, 'person_imgsz': PERSON_IMGSZ, 'input': [w, h], 'rss_mib': rss_mib()}
    _last_metrics[camera_id] = metrics
    atomic_bytes(folder / f'live_{camera_id}.metrics.json', json.dumps(metrics).encode())
    if os.environ.get('PPE_DEBUG') == '1':
        for p in people:
            print(f"DEBUG {camera_id} f{frame_id} ID {p['track_id']} {p['zone']} {p['associated_ppe']} "
                  f"wash={p['handwash_status']} face={p['face_touch']}")
    return people
