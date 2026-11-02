"""Back up every event and its snapshot from a backend (old or new API) to a folder.

  PPE_API_KEY=... python performance/export_events.py --url https://… --out backups/render_events_X
"""
import argparse
import json
import os
from pathlib import Path

import httpx


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--url', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    key = os.environ.get('PPE_API_KEY') or ap.error('Set PPE_API_KEY')
    out = Path(args.out); (out / 'snapshots').mkdir(parents=True, exist_ok=True)
    events, offset = [], 0
    with httpx.Client(base_url=args.url.rstrip('/'), headers={'X-API-Key': key}, timeout=60) as c:
        while True:
            r = c.get('/v1/events', params={'limit': 200, 'offset': offset}); r.raise_for_status()
            page = r.json()['events']
            events += page; offset += len(page)
            if len(page) < 200:
                break
        saved = 0
        for e in events:
            if e.get('snapshot_url'):
                s = c.get(e['snapshot_url'])
                if s.status_code == 200:
                    (out / 'snapshots' / f"{e['event_id']}.jpg").write_bytes(s.content); saved += 1
    (out / 'events.json').write_text(json.dumps(events, indent=1))
    print(f'Saved {len(events)} events and {saved} snapshots to {out}')


if __name__ == '__main__':
    main()
