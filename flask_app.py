import eventlet
eventlet.monkey_patch()  # MUST be first

import os
import sqlite3
import threading
import time
from typing import Dict, Optional, Tuple, List

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory
from flask_socketio import SocketIO

print("[flask_app] module import starting...")

# --------------------- CONFIG ---------------------
DB_PATH = os.getenv("SC_DB", "smart_crosswalk.db")
JPEG_QUALITY = int(os.getenv("SC_JPEG_QUALITY", "70"))  # lower = smaller, faster
STREAM_INTERVAL_S = float(os.getenv("SC_STREAM_INTERVAL", "0.03"))
DB_TAIL_INTERVAL_S = float(os.getenv("SC_DB_TAIL_INTERVAL", "0.5"))

# --------------------- SHARED STATE ---------------------
latest_frames: Dict[str, Optional[np.ndarray]] = {"ped": None, "veh": None, "tl": None}
latest_jpegs:  Dict[str, Optional[bytes]]     = {"ped": None, "veh": None, "tl": None}

# Current board & scenario state (controlled by /api/set_scenario)
board_state = {
    "board_veh": "OFF",      # "ON" or "OFF"
    "board_ped_l": "OFF",
    "board_ped_r": "OFF",
    "scenario": "baseline"   # baseline | scenario_1_night_ped | scenario_2_rush_hold | scenario_3_emergency
}

latest_status = {
    "ts": 0.0,
    "ped_count": 0,
    "veh_count": 0,
    "tl_color": "unknown",
    "nearest_vehicle_distance_m": 0.0,
    "avg_vehicle_speed_mps": 0.0,
    "action": "OFF",
    "scenario": "baseline",
    # still tracked, but not shown as badges anymore
    "board_veh": "OFF",
    "board_ped_l": "OFF",
    "board_ped_r": "OFF",
}

_state_lock = threading.Lock()

# --------------------- APP / SOCKET ---------------------
app = Flask(__name__, static_folder="static", static_url_path="")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

@app.get("/")
def root_index():
    return send_from_directory(app.static_folder, "index.html")

# Optional favicon (avoid 404)
@app.get("/favicon.ico")
def favicon():
    path = os.path.join(app.static_folder, "favicon.ico")
    if os.path.exists(path):
        return send_from_directory(app.static_folder, "favicon.ico", mimetype="image/vnd.microsoft.icon")
    png = os.path.join(app.static_folder, "images", "Adamson.png")
    if os.path.exists(png):
        return send_from_directory(os.path.dirname(png), "Adamson.png", mimetype="image/png")
    return ("", 204)

@app.get("/api/health")
def api_health():
    try:
        conn = sqlite3.connect(DB_PATH, timeout=1.0)
        conn.execute("PRAGMA user_version;")
        conn.close()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "msg": repr(e)}), 500

# --------------------- SCENARIO ---------------------
def decide_scenario(now_ts: float, ped_count: int, veh_count: int,
                    tl_color: str, flags: Dict[str, bool]) -> Tuple[str, str]:
    # emergency flag always overrides
    if flags.get("ambulance", False):
        return ("STOP", "scenario_3_emergency")
    # simple demo heuristics
    if flags.get("night", False) and ped_count >= 30 and veh_count <= 2 and tl_color == "green":
        return ("STOP", "scenario_1_night_ped")
    if flags.get("rush", False) and (5 <= ped_count <= 10) and veh_count >= 20 and tl_color == "red":
        return ("OFF", "scenario_2_rush_hold")
    # baseline
    action = "OFF"
    if tl_color == "red":
        action = "STOP"
    elif tl_color == "green":
        if ped_count > 0 and veh_count > 0:
            action = "STOP"
        elif ped_count > 0:
            action = "GO"
    elif tl_color == "yellow":
        action = "STOP" if ped_count > 0 else "OFF"
    else:
        if ped_count > 0 and veh_count > 0:
            action = "STOP"
        elif ped_count > 0:
            action = "GO"
    return (action, "baseline")

# --------------------- MJPEG STREAMING (uses cached JPEGs) ---------------------
def mjpeg_stream(key: str):
    boundary = b"--frame"
    header   = b"Content-Type: image/jpeg\r\nCache-Control: no-cache\r\n\r\n"
    while True:
        with _state_lock:
            jpg = latest_jpegs.get(key)
        if jpg is None:
            socketio.sleep(0.03)
            continue
        yield boundary + b"\r\n" + header + jpg + b"\r\n"
        socketio.sleep(STREAM_INTERVAL_S)

@app.get("/stream/<cam>")
def stream_cam(cam: str):
    if cam not in ("ped", "veh", "tl"):
        return "unknown cam", 404
    return Response(mjpeg_stream(cam), mimetype="multipart/x-mixed-replace; boundary=frame")

# --------------------- DB HELPERS ---------------------
def _db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=3.0, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=3000;")
    return conn

def _safe_query_rows(sql: str, args=()):
    try:
        conn = _db_connect()
        cur = conn.cursor()
        cur.execute(sql, args)
        rows = cur.fetchall()
        conn.close()
        return rows
    except sqlite3.OperationalError as e:
        if "no such table" in str(e).lower():
            return []
        raise

# --------------------- REST APIs ---------------------
@app.get("/api/status")
def api_status():
    with _state_lock:
        return jsonify(latest_status)

@app.get("/api/status_now")
def api_status_now():
    """Simple poll endpoint the dashboard can use if Socket.IO fails."""
    with _state_lock:
        return jsonify(latest_status)

@app.get("/api/logs")
def api_logs():
    limit = int(request.args.get("limit", 100))
    rows = _safe_query_rows(
        """
        SELECT id, ts, ped_count, veh_count, tl_color,
               nearest_vehicle_distance_m, avg_vehicle_speed_mps, action
        FROM events
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )
    with _state_lock:
        bveh = latest_status.get("board_veh", "OFF")
        pl   = latest_status.get("board_ped_l", "OFF")
        pr   = latest_status.get("board_ped_r", "OFF")
    return jsonify([
        {
            "id": int(r[0]),
            "ts": float(r[1]),
            "ped_count": int(r[2]),
            "veh_count": int(r[3]),
            "tl_color": str(r[4]),
            "nearest_vehicle_distance_m": float(r[5]),
            "avg_vehicle_speed_mps": float(r[6]),
            "action": str(r[7]),
            "board_veh": bveh,
            "board_ped_l": pl,
            "board_ped_r": pr,
        }
        for r in rows
    ])

@app.get("/api/analytics")
def api_analytics():
    # minute-level aggregates for last 60 minutes
    rows = _safe_query_rows(
        """
        SELECT strftime('%Y-%m-%d %H:%M', ts, 'unixepoch') AS minute,
               AVG(ped_count), AVG(veh_count),
               SUM(action='GO'), SUM(action='STOP'), SUM(action='OFF')
        FROM events
        WHERE ts >= strftime('%s','now','-60 minutes')
        GROUP BY minute
        ORDER BY minute ASC
        """
    )
    return jsonify([
        {
            "minute": m,
            "avg_ped": float(p or 0),
            "avg_veh": float(v or 0),
            "go": int(go or 0),
            "stop": int(st or 0),
            "off": int(off or 0),
        }
        for (m, p, v, go, st, off) in rows
    ])

@app.post("/api/set_scenario")
def api_set_scenario():
    """Set current scenario name and/or board LED states from the Scenarios tab."""
    data = request.get_json(silent=True) or {}
    name = data.get("scenario")     # optional string
    veh  = data.get("board_veh")    # "ON"/"OFF" optional
    pl   = data.get("board_ped_l")  # "ON"/"OFF" optional
    pr   = data.get("board_ped_r")  # "ON"/"OFF" optional
    with _state_lock:
        if isinstance(name, str) and name:
            board_state["scenario"] = name
            latest_status["scenario"] = name
        for k, v in (("board_veh",veh), ("board_ped_l",pl), ("board_ped_r",pr)):
            if v in ("ON","OFF"):
                board_state[k] = v
                latest_status[k] = v
    socketio.emit("status", latest_status, namespace="/realtime")
    return jsonify({"ok": True, "state": {**board_state}})

# ---- Archive management ----
@app.post("/api/logs/delete")
def api_logs_delete():
    data = request.get_json(silent=True) or {}
    ids: List[int] = data.get("ids") or []
    if not ids:
        return jsonify({"ok": False, "msg": "no ids"}), 400
    placeholders = ",".join("?" for _ in ids)
    conn = _db_connect()
    cur = conn.cursor()
    cur.execute(f"DELETE FROM events WHERE id IN ({placeholders})", ids)
    deleted = cur.rowcount
    conn.commit()
    conn.close()
    return jsonify({"ok": True, "deleted": int(deleted)})

@app.post("/api/logs/clear")
def api_logs_clear():
    data = request.get_json(silent=True) or {}
    all_flag = bool(data.get("all"))
    older_than = data.get("older_than_seconds")  # optional
    conn = _db_connect()
    cur = conn.cursor()
    if all_flag:
        cur.execute("DELETE FROM events")
    elif older_than is not None:
        cur.execute("DELETE FROM events WHERE ts < strftime('%s','now') - ?", (int(older_than),))
    else:
        return jsonify({"ok": False, "msg": "provide {all:true} or older_than_seconds"}), 400
    deleted = cur.rowcount
    conn.commit()
    conn.close()
    return jsonify({"ok": True, "deleted": int(deleted)})

# --------------------- realtime emits ---------------------
def tail_db_emit():
    print("[flask_app] DB tailer started")
    last_id = 0
    while True:
        try:
            rows = _safe_query_rows(
                """
                SELECT id, ts, ped_count, veh_count, tl_color,
                       nearest_vehicle_distance_m, avg_vehicle_speed_mps, action
                FROM events
                ORDER BY id DESC LIMIT 1
                """
            )
            if rows:
                row = rows[0]
                if row[0] != last_id:
                    last_id = row[0]
                    payload = {
                        "ts": float(row[1]),
                        "ped_count": int(row[2]),
                        "veh_count": int(row[3]),
                        "tl_color": str(row[4]),
                        "nearest_vehicle_distance_m": float(row[5]),
                        "avg_vehicle_speed_mps": float(row[6]),
                        "action": str(row[7]),
                        "scenario": latest_status.get("scenario","baseline"),
                        "board_veh": latest_status.get("board_veh","OFF"),
                        "board_ped_l": latest_status.get("board_ped_l","OFF"),
                        "board_ped_r": latest_status.get("board_ped_r","OFF"),
                    }
                    socketio.emit("status", payload, namespace="/realtime")
                    socketio.emit("log_insert", payload, namespace="/realtime")
        except Exception as e:
            print("[flask_app] tailer error:", repr(e))
        socketio.sleep(DB_TAIL_INTERVAL_S)

# --------------------- HOOKS (called by app.py) ---------------------
def publish_frame(cam_key: str, frame: np.ndarray):
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    with _state_lock:
        latest_frames[cam_key] = frame
        latest_jpegs[cam_key]  = buf.tobytes() if ok else None

def publish_status_from_loop(now_ts: float, ped_count: int, veh_count: int,
                             tl_color: str, nearest_m: float, avg_mps: float,
                             flags: Dict[str, bool], extra: Dict[str, bool]):
    action, scenario = decide_scenario(now_ts, ped_count, veh_count, tl_color, {
        "night": flags.get("night", False),
        "rush": flags.get("rush", False),
        "ambulance": extra.get("ambulance", False),
    })
    scenario_to_show = board_state.get("scenario", "baseline") or "baseline"
    with _state_lock:
        latest_status.update({
            "ts": float(now_ts),
            "ped_count": int(ped_count),
            "veh_count": int(veh_count),
            "tl_color": str(tl_color),
            "nearest_vehicle_distance_m": float(nearest_m),
            "avg_vehicle_speed_mps": float(avg_mps),
            "action": action,
            "scenario": scenario_to_show if scenario_to_show != "baseline" else scenario,
            "board_veh": board_state["board_veh"],
            "board_ped_l": board_state["board_ped_l"],
            "board_ped_r": board_state["board_ped_r"],
        })
    socketio.emit("status", latest_status, namespace="/realtime")

# --------------------- MAIN ---------------------
def start_http_server(host="0.0.0.0", port=5000):
    print(f"[flask_app] Starting Flask-SocketIO on http://{host}:{port} ...")
    socketio.start_background_task(tail_db_emit)
    socketio.run(app, host=host, port=port, debug=False)

if __name__ == "__main__":
    start_http_server()
