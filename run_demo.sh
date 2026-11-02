#!/usr/bin/env bash
# Live demo control. Needs PPE_API_KEY exported in this terminal (never printed or saved).
#   ./run_demo.sh backup   export all events + snapshots from the backend
#   ./run_demo.sh start    worker (MPS) + monitor (records demo.mp4) + state logger
#   ./run_demo.sh test     spoken guided test (run while a session is started)
#   ./run_demo.sh status   short health view of the running session
#   ./run_demo.sh stop     stop everything gracefully, export events, write report.md
set -euo pipefail
cd "$(dirname "$0")"
URL="${PPE_BACKEND_URL:-https://ppe-gmp-compliance.onrender.com}"
PY=.venv/bin/python
EXPECTED_API=2026-11-02.latest-frame-zones
CUR=performance_run/current_session
need_key() { [ -n "${PPE_API_KEY:-}" ] || { echo 'First run: export PPE_API_KEY="$(pbpaste)"   (copy the Render key first)'; exit 1; }; }
stop_pid() {  # graceful SIGTERM, then wait up to 20 s
  local f=$1; [ -f "$f" ] || return 0; local pid; pid=$(cat "$f")
  kill -TERM "$pid" 2>/dev/null || return 0
  for _ in $(seq 40); do kill -0 "$pid" 2>/dev/null || return 0; sleep 0.5; done
  echo "warning: $f did not stop; killing"; kill -KILL "$pid" 2>/dev/null || true
}
case "${1:-}" in
  backup)
    need_key; $PY performance/export_events.py --url "$URL" --out "backups/render_events_$(date +%Y%m%d_%H%M%S)" ;;
  start)
    need_key
    api=$(curl -s -m 60 "$URL/healthz" | $PY -c 'import json,sys;print(json.load(sys.stdin).get("api_version",""))' || true)
    [ "$api" = "$EXPECTED_API" ] || { echo "Backend reports '$api', expected $EXPECTED_API. Deploy fix/live-pipeline-zones on Render first."; exit 1; }
    [ ! -f "$CUR" ] || { echo "Session $(cat $CUR) is running; ./run_demo.sh stop first."; exit 1; }
    d="performance_run/session_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$d"; echo "$d" > "$CUR"
    PPE_DEVICE=mps nohup $PY -m cloud_backend.worker --url "$URL" --detector ai_detector:detect --metrics "$d/live.jsonl" > "$d/worker.log" 2>&1 &
    echo $! > "$d/worker.pid"
    nohup $PY performance/live_check.py --url "$URL" --out "$d/status.jsonl" > "$d/live_check.log" 2>&1 & echo $! > "$d/check.pid"
    nohup $PY monitor.py --cameras CAM_01,CAM_02 --record "$d/demo.mp4" --summary "$d/monitor_summary.json" > "$d/monitor.log" 2>&1 &
    echo $! > "$d/monitor.pid"
    sleep 8; tail -n 3 "$d/worker.log"; echo "Session started: $d (monitor window opens shortly)" ;;
  test)
    [ -f "$CUR" ] || { echo 'Start a session first.'; exit 1; }
    $PY performance/guided_test.py --out "$(cat $CUR)/markers.jsonl" ;;
  status)
    [ -f "$CUR" ] || { echo 'No session running.'; exit 0; }
    d=$(cat "$CUR"); tail -n 2 "$d/worker.log"
    tail -n 1 "$d/status.jsonl" | $PY -c 'import json,sys;r=json.loads(sys.stdin.read());[print(c["camera_id"],c["status"],"stale" if c["stale"] else "fresh","processed",c["processed_frame_id"],"uploads",(c.get("pipeline") or {}).get("uploads")) for c in r.get("cameras",[])];print("ESP32 command:",r.get("command",{}).get("status"))' ;;
  stop)
    need_key; [ -f "$CUR" ] || { echo 'No session running.'; exit 0; }
    d=$(cat "$CUR")
    stop_pid "$d/monitor.pid"; stop_pid "$d/check.pid"; stop_pid "$d/worker.pid"
    $PY performance/export_events.py --url "$URL" --out "$d/events" || echo 'event export failed (backend unreachable?)'
    $PY performance/session_report.py "$d" > /dev/null && echo "Report: $d/report.md   Video: $d/demo.mp4"
    rm -f "$CUR" ;;
  *) sed -n 2,8p "$0"; exit 1 ;;
esac
