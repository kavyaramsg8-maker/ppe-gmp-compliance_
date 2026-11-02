"""Summarise a recorded live session folder into report.md (actual measurements only).

  python performance/session_report.py performance_run/session_YYYYmmdd_HHMMSS
"""
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from performance_tools import camera_summary  # noqa: E402

SETTLE = 5.0  # seconds after a spoken instruction before judging the step


def ts(s):
    return datetime.fromisoformat(s).timestamp()


def jsonl(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def upload_rates(status):
    """Per camera: list of (time, uploads/s over the last ~10 s) from backend counters."""
    rates = {}
    hist = {}
    for row in status:
        t = ts(row['t'])
        for c in row.get('cameras', []):
            pipe = c.get('pipeline') or {}
            h = hist.setdefault(c['camera_id'], [])
            h.append((t, c.get('session_id'), pipe.get('uploads', 0)))
            past = [x for x in h if t - x[0] <= 10 and x[1] == c.get('session_id')]
            rate = (past[-1][2] - past[0][2]) / (past[-1][0] - past[0][0]) if len(past) > 1 and past[-1][0] > past[0][0] else 0
            rates.setdefault(c['camera_id'], []).append((t, rate))
    return rates


def rate_class(rates, t):
    near = min(rates, key=lambda x: abs(x[0] - t)) if rates else (t, 0)
    r = near[1]
    return '3 FPS' if 2.0 <= r < 4.0 else '5 FPS' if r >= 4.0 else None


def fmt(d, k='p50'):
    v = d.get(k) if d else None
    return '-' if v is None else f'{v:.0f}'


def judge(step, cam, rows):
    truth = step['truth'][cam]
    statuses = Counter(r['status'] for r in rows)
    persons = [p for r in rows for p in r.get('persons', [])]
    items = {}
    for p in persons:
        for k, v in (p.get('ppe') or {}).items():
            items.setdefault(k, Counter())[v] += 1
    wash = [p.get('handwash_seconds') or 0 for p in persons]
    wstat = Counter(p.get('handwash_status') for p in persons)
    obs = f"status {dict(statuses)}"
    ok = None
    if isinstance(truth, dict):
        obs += ' | ' + ', '.join(f"{k}: {dict(v)}" for k, v in sorted(items.items()))
        want_violation = 'missing' in truth.values()
        ok = statuses.get('VIOLATION', 0) > 0 if want_violation else statuses.get('COMPLIANT', 0) > 0
    elif truth == 'empty':
        ok = statuses.get('NO_PERSON', 0) >= 0.6 * max(1, len(rows))
    elif truth == 'unclear':
        ok = statuses.get('COMPLIANT', 0) == 0
    elif truth == 'corridor':
        ok = statuses.get('MONITORING', 0) > 0 and statuses.get('VIOLATION', 0) == 0
    elif truth in ('washing', 'paused', 'incomplete', 'washing_to_complete'):
        obs += f" | wash status {dict(wstat)} | timer {min(wash, default=0):.1f}->{max(wash, default=0):.1f}s"
        growth = (max(wash) - min(wash)) if wash else 0
        ok = {'washing': growth > 1.0, 'paused': growth <= 0.6, 'incomplete': wstat.get('incomplete', 0) > 0,
              'washing_to_complete': wstat.get('completed', 0) > 0}[truth]
    elif truth == 'disconnected':
        ok = any(r.get('stale') for r in rows) and statuses.get('COMPLIANT', 0) == 0
    elif truth == 'reconnected':
        ok = any(not r.get('stale') for r in rows)
    return truth, obs, ok


def main():
    d = Path(sys.argv[1])
    rows = jsonl(d / 'live.jsonl'); status = jsonl(d / 'status.jsonl'); markers = jsonl(d / 'markers.jsonl')
    rates = upload_rates(status)
    out = [f'# Live session report: {d.name}', '',
           'Real phones → Render backend → Mac worker. All numbers measured in this session. '
           'Phone timestamps are corrected to server time by the phone page; capture→decision uses the Mac clock.', '']
    out += ['## Throughput, steadiness and delay (per camera, per phone upload rate)', '',
            '| camera | phone rate | uploads/s | processed FPS | interval p50/p95 ms | interval CV | capture→decision p50/p95 ms | frame age at fetch p50 ms | detector p50 ms | frames | not processed |',
            '|---|---|---|---|---|---|---|---|---|---|---|']
    for cam in sorted({r.get('camera_id') for r in rows if r.get('camera_id')}):
        for cls in ('3 FPS', '5 FPS'):
            sub = [r for r in rows if r.get('camera_id') == cam and r.get('recorded_at') and rate_class(rates.get(cam, []), ts(r['recorded_at'])) == cls]
            if len(sub) < 5:
                continue
            s = camera_summary(sub)
            up = [r for t, r in rates.get(cam, []) if (('3' in cls and 2 <= r < 4) or ('5' in cls and r >= 4))]
            out.append(f"| {cam} | {cls} | {sum(up)/len(up):.2f} | {s['accepted_fps'] or 0:.2f} | {fmt(s['interval_ms'])}/{fmt(s['interval_ms'],'p95')} | "
                       f"{(s['interval_cv'] or 0):.2f} | {fmt(s['capture_to_decision_ms'])}/{fmt(s['capture_to_decision_ms'],'p95')} | "
                       f"{fmt(s['age_at_fetch_ms'])} | {fmt(s['detector_ms'])} | {s['accepted_frames']} | {s['uploaded_frames_skipped']} |")
    mon = d / 'monitor_summary.json'
    if mon.exists():
        out += ['', 'Monitor (whole session): ' + '; '.join(f"{c}: shown-interval p50 {fmt(v['shown_interval_ms'])} ms, CV {(v['shown_interval_ms'].get('cv') or 0):.2f}, smoothing +{v['smoothing_extra_ms']:.0f} ms"
                                                       for c, v in json.loads(mon.read_text()).items())]
    # Stale handling
    out += ['', '## Stale / disconnected handling', '']
    for cam in ('CAM_01', 'CAM_02'):
        spans, cur = [], None
        for row in status:
            c = next((x for x in row.get('cameras', []) if x['camera_id'] == cam), None)
            if c is None:
                continue
            if c['stale'] and cur is None:
                cur = [row['t'], row['t'], c['status'], row.get('command', {}).get('green_led')]
            elif c['stale']:
                cur[1] = row['t']; cur[3] = cur[3] or row.get('command', {}).get('green_led')
            elif cur:
                spans.append(cur); cur = None
        if cur:
            spans.append(cur)
        for a, b, st, green in spans:
            out.append(f'- {cam}: stale {a[11:19]}–{b[11:19]} UTC ({ts(b)-ts(a):.0f} s), status {st}, ESP32 green during stale: {bool(green)}')
    # Guided steps
    if markers:
        out += ['', '## Guided test: expected vs observed (judged from +5 s after each instruction)', '',
                '| step | instruction | camera | expected | observed | result |', '|---|---|---|---|---|---|']
        for m in markers:
            a, b = ts(m['start']) + SETTLE, ts(m['end'])
            for cam in m['truth']:
                sel = [c for row in status if a <= ts(row['t']) <= b for c in row.get('cameras', []) if c['camera_id'] == cam]
                truth, obs, ok = judge(m, cam, sel)
                res = 'no data' if not sel else 'OK' if ok else 'NOT MET'
                out.append(f"| {m['step']} | {m['text'][:70]} | {cam} | {json.dumps(truth) if isinstance(truth, dict) else truth} | {obs[:260]} | {res} |")
        # PPE evidence vs label, per item (person-seconds, after settle)
        conf = {}
        for m in markers:
            for cam, truth in m['truth'].items():
                if not isinstance(truth, dict):
                    continue
                a, b = ts(m['start']) + SETTLE, ts(m['end'])
                for row in status:
                    if a <= ts(row['t']) <= b:
                        for c in row.get('cameras', []):
                            if c['camera_id'] == cam:
                                for p in c.get('persons', []):
                                    for item, want in truth.items():
                                        conf.setdefault(item, Counter())[(want, (p.get('ppe') or {}).get(item, 'not_detected'))] += 1
        out += ['', '## PPE model evidence vs scripted label (person-seconds on CAM_02)', '',
                '| item | label | present | missing | not_detected (unknown) |', '|---|---|---|---|---|']
        for item, cnt in sorted(conf.items()):
            for want in ('present', 'missing'):
                out.append(f"| {item} | {want} | {cnt[(want,'present')]} | {cnt[(want,'missing')]} | {cnt[(want,'not_detected')]} |")
    # Events and command
    ev_path = d / 'events' / 'events.json'
    if ev_path.exists():
        ev = json.loads(ev_path.read_text())
        snaps = len(list((d / 'events' / 'snapshots').glob('*.jpg')))
        out += ['', f'## Events ({len(ev)}; {snaps} snapshots saved in events/snapshots)', '',
                'By type: ' + ', '.join(f'{k} {v}' for k, v in Counter(e['type'] for e in ev).items()), '']
        for e in sorted(ev, key=lambda e: e['recorded_at'])[-40:]:
            out.append(f"- {e['recorded_at'][11:19]} {e['type']} {e['camera_id']} track {e['track_id']} {e['missing_items'][0]} zone {e.get('zone')}")
    changes, last = [], None
    for row in status:
        cmd = row.get('command')
        key = cmd and (cmd['status'], cmd['red_led'], cmd['green_led'], cmd['amber_led'], cmd['buzzer'])
        if key and key != last:
            changes.append(f"- {row['t'][11:19]} status {cmd['status']} red {cmd['red_led']} green {cmd['green_led']} amber {cmd['amber_led']} buzzer {cmd['buzzer']}")
            last = key
    out += ['', f'## ESP32 command responses (backend side only; {len(changes)} changes)', ''] + changes[:60]
    (d / 'report.md').write_text('\n'.join(out) + '\n')
    print('\n'.join(out[:40]))
    print(f'\nWrote {d / "report.md"}')


if __name__ == '__main__':
    main()
