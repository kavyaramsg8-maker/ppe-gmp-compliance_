# Kavya — cloud backend starter

This package ADDS files to ppe-gmp-compliance. It does not replace your local
model, detect_and_track_live_FINAL.py, payloads or existing rules. Source
interfaces were read from GitHub commit 08b4705fc8aca19e1181dd04e855de61b8dacd2f.
Unpushed laptop changes were not accessible and were not overwritten.

## 1. Copy the addon into the project
Extract this ZIP. Copy cloud_backend/, tests_cloud/, docs/CLOUD_API_CONTRACT.md,
Dockerfile.cloud, requirements-cloud.txt, requirements-cloud-test.txt,
.env.cloud.example and KAVYA_START_HERE.md into your current project root
(the same folder that contains ml_adapter.py). Merge the docs folder; do not
replace it. Keep your original files. Do not run git reset or git clean.

Add these entries to your EXISTING .gitignore (do not replace its contents):

```
.env
.env.cloud
cloud_data/
.venv-cloud/
__pycache__/
```

## 2. Local API setup (macOS VS Code terminal)
Your screenshot already shows an active .venv. For backend dependencies isolated
from the computer-vision environment, create a second environment:

```bash
python3 -m venv .venv-cloud
source .venv-cloud/bin/activate
python -m pip install -r requirements-cloud-test.txt
export PPE_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export PPE_CAMERA_IDS=CAM_01
export PPE_DATA_DIR="$PWD/cloud_data"
python -m cloud_backend.serve
```

Keep this terminal running. The secret key exists only in this shell unless you
save it privately. Do not paste it into chat, screenshots, GitHub or a public URL.
For another terminal on your laptop, set the same key privately there; alternatively
use Swagger's header parameter in http://localhost:8000/docs.

- Health: http://localhost:8000/healthz
- API explorer: http://localhost:8000/docs
- Laptop webcam upload test: http://localhost:8000/camera
- Phone upload later: https://YOUR-CLOUD-HOST/camera

Localhost on your phone refers to your PHONE, not the laptop. Browser camera
access normally requires HTTPS or localhost, so use the cloud HTTPS page for the
phone test. An ordinary LAN HTTP URL is not a reliable mobile camera test.
Starting camera upload alone should say UNKNOWN/AWAITING_INFERENCE. That is
expected: Aravind's detector must submit results before compliance can be claimed.

## 3. Test your backend without waiting for inference
Stop live camera upload first. In a second terminal with the SAME PPE_API_KEY:

```bash
source .venv-cloud/bin/activate
python -m cloud_backend.smoke_test --confirm-test
```

This explicitly starts a new CAM_01 session and submits SYNTHETIC test labels.
It checks a compliant → missing gloves → restored sequence, event storage,
authenticated snapshot retrieval and a simulated ESP32 acknowledgement.
It is not a CV or physical hardware test. Restart the camera page afterwards.
Do not run this synthetic test on a live deployment while a real worker is running.

Run the automated backend and existing small rules tests:

```bash
python -m pytest tests_cloud test_smoothing_basic.py test_zones_basic.py -q
```

## 4. Deploy the backend
See docs/CLOUD_DEPLOYMENT.md. Hosting account access and persistent storage must
be configured by you. No cloud account was accessed or deployment created by
this package. The Docker build is prepared but hasn't been built on a cloud host.
The network API runs without downloading models. Aravind adds the real detector
and its dependencies to the same cloud host/image using the integration contract.

## 5. Share the contract
Give docs/CLOUD_API_CONTRACT.md to Aravind, Ujju and Sana. Your backend is useful
before the detector: each person can implement against these endpoints.
First milestone: one phone → cloud API + REAL worker → confirmed decision → ESP32.
Then set PPE_CAMERA_IDS=CAM_01,CAM_02 and connect the second phone.

## Scope and limits
SQLite stores event metadata and JPEG snapshots transactionally in the configured
data directory. Unprocessed frames are bounded in memory, not archived. Use one
API process and one worker/instance; never --workers 2 or autoscale this MVP.
A server restart preserves saved events/acks on a persistent disk but resets
live camera/tracker state to UNKNOWN; phones must start new sessions.
Auth is a single shared team key, not per-device accounts. Rotate it if exposed.
Event storage grows over time: monitor disk usage and archive/delete old demo
records before long unattended operation. No retention job is included.
Current network rules cover PPE; original face-touch/handwash/gowning features
are not automatically cloud-enabled by this addon. Those remain Aravind's scope.
