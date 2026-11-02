"""Log backend camera state and the ESP32 command once per second (no images or key written).

Uses the read-only `observer=true` command query, so the ESP32's online status is not affected.
"""
import argparse
from datetime import datetime, timezone
import json
import os
import signal
import time

import httpx


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--url', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    key = os.environ.get('PPE_API_KEY') or ap.error('Set PPE_API_KEY')
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    with httpx.Client(base_url=args.url.rstrip('/'), headers={'X-API-Key': key}, timeout=10) as c, open(args.out, 'a') as f:
        while not stop:
            t0 = time.monotonic(); row = {'t': datetime.now(timezone.utc).isoformat()}
            try:
                row['cameras'] = c.get('/v1/cameras').json()['cameras']
                cmd = c.get('/v1/devices/ESP32_01/command', params={'observer': 'true'}).json()
                row['command'] = {k: cmd.get(k) for k in ('command_id', 'status', 'alert', 'red_led', 'green_led', 'amber_led', 'buzzer')}
            except (httpx.HTTPError, ValueError, KeyError) as e:
                row['error'] = type(e).__name__
            f.write(json.dumps(row) + '\n'); f.flush()
            time.sleep(max(0, 1 - (time.monotonic() - t0)))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
