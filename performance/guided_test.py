"""Spoken, timed test script for the two-phone demo; writes ground-truth markers.

Each step is announced aloud (macOS `say`) and on screen with a countdown, and its
start/end time and the expected truth are written to markers.jsonl. The session
report compares backend decisions and model evidence with these labels.
"""
import argparse
from datetime import datetime, timezone
import json
import subprocess
import time

FULL = {'hairnet': 'present', 'mask': 'present', 'gloves': 'present'}
# (seconds, spoken instruction, expected truth)
STEPS = [
    (15, 'Both phones at 3 frames per second. Everyone leave both camera views.', {'CAM_01': 'empty', 'CAM_02': 'empty'}),
    (25, 'Stand in the production camera view, CAM 2, wearing hairnet, mask and gloves. Face the camera.', {'CAM_02': dict(FULL)}),
    (20, 'Remove your mask. Stay facing camera 2.', {'CAM_02': dict(FULL, mask='missing')}),
    (20, 'Put your mask back on.', {'CAM_02': dict(FULL)}),
    (20, 'Remove your gloves. Keep hands visible.', {'CAM_02': dict(FULL, gloves='missing')}),
    (15, 'Put your gloves back on.', {'CAM_02': dict(FULL)}),
    (20, 'Remove your hairnet.', {'CAM_02': dict(FULL, hairnet='missing')}),
    (15, 'Put your hairnet back on.', {'CAM_02': dict(FULL)}),
    (15, 'Turn your back to camera 2 and cover your face with your hands.', {'CAM_02': 'unclear'}),
    (15, 'Walk out of camera 2 into the corridor view of camera 1. Stay away from the sink.', {'CAM_01': 'corridor', 'CAM_02': 'empty'}),
    (8, 'Go to the sink and rub your hands together, below your shoulders.', {'CAM_01': 'washing'}),
    (6, 'Stop rubbing. Keep your hands still at the sink.', {'CAM_01': 'paused'}),
    (6, 'Rub your hands again.', {'CAM_01': 'washing'}),
    (12, 'Leave the sink now and stand in the corridor.', {'CAM_01': 'incomplete'}),
    (28, 'Go back to the sink and rub your hands continuously until told to stop.', {'CAM_01': 'washing_to_complete'}),
    (12, 'Stop. Leave both camera views.', {'CAM_01': 'empty', 'CAM_02': 'empty'}),
    (25, 'Change the upload rate on both phones to 5 frames per second, then stand in camera 2 with full P P E.', {'CAM_02': dict(FULL)}),
    (20, 'Remove your mask.', {'CAM_02': dict(FULL, mask='missing')}),
    (20, 'Put your mask back on.', {'CAM_02': dict(FULL)}),
    (25, 'Press Stop on the camera 2 phone and wait.', {'CAM_02': 'disconnected'}),
    (20, 'Press Start on the camera 2 phone again.', {'CAM_02': 'reconnected'}),
]


def say(text):
    print('\n>>> ' + text, flush=True)
    try:
        subprocess.Popen(['say', text])
    except OSError:
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    ap.add_argument('--lead', type=int, default=10, help='Seconds before the first step')
    args = ap.parse_args()
    total = sum(s for s, _, _ in STEPS)
    say(f'Guided test starting in {args.lead} seconds. It takes about {round(total / 60)} minutes.')
    time.sleep(args.lead)
    with open(args.out, 'a') as f:
        for i, (seconds, text, truth) in enumerate(STEPS, 1):
            say(f'Step {i}. {text}')
            start = datetime.now(timezone.utc)
            for left in range(seconds, 0, -1):
                print(f'\r   {left:3d}s ', end='', flush=True); time.sleep(1)
            f.write(json.dumps({'step': i, 'text': text, 'truth': truth, 'start': start.isoformat(),
                                'end': datetime.now(timezone.utc).isoformat()}) + '\n'); f.flush()
    say('Guided test finished. Thank you.')


if __name__ == '__main__':
    main()
