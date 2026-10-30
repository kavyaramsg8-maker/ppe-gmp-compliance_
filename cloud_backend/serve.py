"""Single-instance cloud launcher; optionally supervise an inference worker too."""
import os
import signal
import subprocess
import sys
import time


def main():
    # Validate settings before starting either process.
    from cloud_backend.app import Settings
    Settings.from_env()
    port=os.environ.get('PORT','8000')
    if not port.isdigit(): raise ValueError('PORT must be numeric')
    processes=[]
    def stop(*_):
        for process in processes:
            if process.poll() is None: process.terminate()
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    processes.append(subprocess.Popen([sys.executable,'-m','uvicorn','cloud_backend.app:create_app','--factory','--host','0.0.0.0','--port',port,'--workers','1']))
    if os.environ.get('PPE_DETECTOR'):
        env=dict(os.environ,PPE_BACKEND_URL=f'http://127.0.0.1:{port}')
        processes.append(subprocess.Popen([sys.executable,'-m','cloud_backend.worker'],env=env))
    try:
        while all(p.poll() is None for p in processes): time.sleep(.25)
    finally:
        stop()
        for process in processes:
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired: process.kill();process.wait()
    # An unexpected child exit must be visible to the hosting supervisor.
    return next((p.returncode for p in processes if p.returncode not in (0,-15,-2)),0)

if __name__=='__main__': sys.exit(main())
