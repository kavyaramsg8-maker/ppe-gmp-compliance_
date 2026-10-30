# Cloud deployment — API first, real cloud inference next

## Render option
Render's FastAPI deployment guide:
https://render.com/docs/deploy-fastapi
Web services and environment settings:
https://render.com/docs/web-services
Persistent filesystem requirements:
https://render.com/docs/disks

Create a Web Service connected to your repository/working branch after you add
and test these files. Use the existing project root; do not deploy the old GUI
webcam script as the web server.

| Setting | API-only value |
|---|---|
| Runtime | Python |
| Build command | pip install -r requirements-cloud.txt |
| Start command | python -m cloud_backend.serve |
| Health check | /healthz |
| PPE_API_KEY | Your private random key, at least 32 characters |
| PPE_CAMERA_IDS | CAM_01 initially, CAM_01,CAM_02 after one-camera test |
| PPE_DATA_DIR | /var/data |
| Persistent disk mount | /var/data |
| Instance count | 1 |

Bind address and Render's PORT are handled by the launcher. Render normally
provides HTTPS, which the mobile browser camera page needs.

A persistent disk is available on a paid service. Without one, local SQLite data
can be lost on restart/redeploy. A free ephemeral service can test requests but
does not meet durable event/snapshot storage. Do not enter payment details or
create paid resources until you choose that option. Existing university/cloud VM
with persistent disk is another option; use the Docker/VM approach below.

## Real inference on the cloud
Aravind supplies the module:function implementation, model weights and dependencies.
Install them in the SAME cloud service image/environment and set
PPE_DETECTOR=aravind_inference:infer. The launcher then supervises both API and worker.
For Render's Python runtime, change the build command to install his additional
requirements too, using the exact dependency file your team agrees.
Backend-only requirements intentionally do not install Torch/OpenCV/Ultralytics.
Do not enable PPE_DETECTOR until its module can be imported headlessly.

Measure actual model runtime and memory on the chosen compute. No GPU or
real-time two-camera throughput is guaranteed by a hosting entry tier.

## Docker / university VM
From the existing project root:

```bash
docker build -f Dockerfile.cloud -t ppe-cloud .
docker run --rm -p 8000:8000 --env PPE_API_KEY --env PPE_CAMERA_IDS \
  -v ppe-cloud-data:/var/data ppe-cloud
```

The Dockerfile includes only the backend and existing MLAdapter. Aravind must
extend it with his detector module/weights/dependencies for real inference.
On a public VM place an HTTPS reverse proxy/load balancer in front of port 8000,
use a persistent volume, and expose the HTTPS hostname to phones/ESP32.
The Dockerfile was prepared but not built in this environment.

## Deployment acceptance checklist
1. HTTPS /healthz responds; /v1/events rejects requests without the key.
2. Open HTTPS /camera on one phone; upload CAM_01 frames.
3. Until a detector is connected, status remains UNKNOWN, not COMPLIANT.
4. With real cloud inference connected, test present / missing / unclear / restored.
5. Poll and acknowledge commands from the real ESP32, including offline watchdog.
6. Confirm the event image endpoint returns the correct camera/session JPEG.
7. Restart/redeploy service, confirm stored events persist and live state is UNKNOWN.
8. Restart phone session; connect CAM_02 and repeat with overlapping track IDs.

If the deployment is unresponsive, check service logs without exposing the API
key. If CPU inference exceeds the freshness limit, lower input size/model workload
or choose suitable compute; do not claim outdated results are live.
