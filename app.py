# app.py
# Import the web app FIRST so eventlet is monkey-patched before anything else.
from flask_app import publish_frame, publish_status_from_loop, start_http_server

import os
import re
import cv2
import time
import math
import sqlite3
import threading
from queue import Queue, Empty, Full
from collections import deque
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Union

import numpy as np
from ultralytics import YOLO

# Keep OpenCV predictable/light on Pi
cv2.setNumThreads(1)
try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

SHOW_WINDOWS = False
PRINT_DEBUG = True

# ---------- CONFIG ----------
# Robustness / perf toggles
FALLBACK_SYNC_AFTER_S = float(os.getenv("SC_FALLBACK_SYNC_AFTER_S", "0.6"))
DISABLE_CLASS_FILTER  = os.getenv("SC_DISABLE_CLASS_FILTER", "0") == "1"

# YOLO
YOLO_MODEL  = os.getenv("SC_YOLO_MODEL", "yolov8n.pt")
YOLO_CONF   = float(os.getenv("SC_YOLO_CONF", "0.20"))
YOLO_IMG_SZ = int(os.getenv("SC_YOLO_IMG", "416"))

# Fast classes (COCO indices)
COCO_PED = [0]                    # person
COCO_VEH = [1, 2, 3, 5, 7]        # bicycle, car, motorcycle, bus, truck

# Drawing classes (labels)
PEDESTRIAN_CLASSES = {"person"}
VEHICLE_CLASSES    = {"bicycle", "car", "motorbike", "bus", "truck"}

# Frames
FRAME_W = int(os.getenv("SC_FRAME_W", "640"))
FRAME_H = int(os.getenv("SC_FRAME_H", "360"))
FPS_TARGET = int(os.getenv("SC_FPS", "15"))
FRAME_TIME = 1.0 / max(1, FPS_TARGET)
SKIP_FRAMES = int(os.getenv("SC_SKIP", "1"))

# Throttle JPEG encodes only (keeps inference at SKIP=1 smooth)
PUBLISH_HZ = float(os.getenv("SC_PUBLISH_HZ", "10"))
_last_pub = {"ped": 0.0, "veh": 0.0, "tl": 0.0}

# Vehicle speed/distance calibration
PIXELS_PER_METER_VEH = float(os.getenv("SC_VEH_PPM", "40.0"))
MIN_TRACK_HITS = 3
SPEED_AVG_WINDOW = 5
PEDESTRIAN_LANE_Y = int(os.getenv("SC_LANE_Y", "250"))
VEHICLE_CLOSE_THRESH_M = float(os.getenv("SC_VEH_CLOSE_M", "6.0"))

# Traffic light ROI (x,y,w,h)
TRAFFIC_LIGHT_ROI = tuple(map(int, os.getenv("SC_TL_ROI", "100,60,120,160").split(",")))
HSV_RED_1 = ((0,120,120),(10,255,255))
HSV_RED_2 = ((170,120,120),(180,255,255))
HSV_YELLOW= ((15,120,120),(35,255,255))
HSV_GREEN = ((40,70,70),(90,255,255))

# DB + status throttling
DB_PATH = os.getenv("SC_DB", "smart_crosswalk.db")
LOG_EVERY_SEC = int(os.getenv("SC_LOG_SEC", "30"))
STATUS_MIN_PERIOD = float(os.getenv("SC_STATUS_PERIOD", "0.25"))

# Colors
COLOR_GREEN  = (0,255,0)
COLOR_RED    = (0,0,255)
COLOR_YELLOW = (0,255,255)
COLOR_WHITE  = (255,255,255)
COLOR_BLUE   = (255,0,0)

# ---------- LED ----------
def _init_led():
    try:
        from luma.core.interface.serial import spi, noop
        from luma.led_matrix.device import max7219
        from luma.core.render import canvas
        from PIL import ImageFont

        serial = spi(port=0, device=0, gpio=noop())
        device = max7219(serial, cascaded=int(os.getenv("SC_LED_CASCADE","4")),
                         block_orientation=int(os.getenv("SC_LED_ORIENTATION","-90")), rotate=0)
        font = ImageFont.load_default()

        def show_led(msg: str):
            with canvas(device) as draw:
                draw.text((1, -2), msg, fill="white", font=font)
        if PRINT_DEBUG: print("[LED] MAX7219 initialized")
        return show_led
    except Exception as e:
        if PRINT_DEBUG: print("[LED] Fallback console:", repr(e))
        return lambda msg: print("[LED]", msg)

show_led = _init_led()

# ---------- UTIL ----------
def _normalize_cam(value: Union[str,int,None]):
    if value is None: return None
    if isinstance(value,int): return value
    s = str(value).strip()
    if s.isdigit(): return int(s)
    try:
        if s.startswith("/dev/"):
            real = os.path.realpath(s)
            m = re.match(r"^/dev/video(\d+)$", real)
            if m: return int(m.group(1))
            if os.path.exists(real): return real
    except Exception: pass
    m = re.match(r"^/dev/video(\d+)$", s)
    return int(m.group(1)) if m else s

def q_replace_latest(q: Queue, item):
    try:
        while True:
            try: q.get_nowait()
            except Empty: break
        q.put_nowait(item)
    except Full:
        pass

def publish_frame_throttled(key: str, frame: np.ndarray):
    now = time.time()
    if now - _last_pub.get(key,0.0) >= (1.0/max(1.0,PUBLISH_HZ)):
        publish_frame(key, frame)
        _last_pub[key] = now

# ---------- CAMERA ----------
class CameraStream:
    def __init__(self, index, width, height, fps):
        self.index, self.width, self.height, self.fps = index, width, height, fps
        self.lock = threading.Lock()
        self.frame = None
        self.ret = False
        self.stopped = False
        self._open_camera()
        threading.Thread(target=self._update, daemon=True).start()

    def _open_camera(self):
        idx = _normalize_cam(self.index)
        if isinstance(idx,int):
            self.cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if not self.cap.isOpened(): self.cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
            if not self.cap.isOpened(): raise RuntimeError(f"Could not open camera index {idx}")
            pretty = f"/dev/video{idx}"
        else:
            self.cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if not self.cap.isOpened(): self.cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
            if not self.cap.isOpened(): raise RuntimeError(f"Could not open camera path {idx}")
            pretty = str(idx)

        try: self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception: pass

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, min(self.fps, 20))

        def set_fourcc(code):
            f = cv2.VideoWriter_fourcc(*code)
            self.cap.set(cv2.CAP_PROP_FOURCC, f)
            return int(self.cap.get(cv2.CAP_PROP_FOURCC)) == f
        if not set_fourcc('MJPG'):
            set_fourcc('YUYV') or set_fourcc('YUY2')

        self.cap.read()

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f = self.cap.get(cv2.CAP_PROP_FPS)
        four = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        four_s = "".join([chr((four >> 8*i) & 0xFF) for i in range(4)])
        print(f"[OPEN] {pretty} -> {w}x{h}@{f:.1f} FOURCC={four_s}")

    def _reopen_once(self):
        try: self.cap.release()
        except Exception: pass
        time.sleep(0.2)
        self._open_camera()

    def _update(self):
        no_frame = 0
        frame_interval = 1.0 / max(1,self.fps)
        last_t = 0.0
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                no_frame += 1
                if no_frame == 20:
                    print("[INFO] Reopening camera due to stalled framesâ€¦")
                    self._reopen_once()
                    no_frame = 0
                time.sleep(0.02)
                continue
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width,self.height), interpolation=cv2.INTER_AREA)
            with self.lock:
                self.ret, self.frame = True, frame
            now = time.time()
            sleep_left = frame_interval - (now - last_t)
            if sleep_left > 0: time.sleep(sleep_left)
            last_t = now

    def read(self):
        with self.lock:
            return self.ret, None if self.frame is None else self.frame.copy()

    def stop(self):
        self.stopped = True
        try: self.cap.release()
        except Exception: pass

# ---------- TRACKER ----------
@dataclass
class Track:
    id: int
    cls: str
    history: deque
    hits: int = 0

class CentroidTracker:
    def __init__(self, max_dist_px=120.0, max_age_s=1.0):
        self.next_id = 1
        self.tracks: Dict[int, Track] = {}
        self.max_dist = max_dist_px
        self.max_age = max_age_s

    def update(self, detections, now):
        det_centroids = [(lab, ((x1+x2)//2, (y1+y2))) for lab,(x1,y1,x2,y2) in detections]
        unmatched = list(range(len(det_centroids)))

        for tid, tr in list(self.tracks.items()):
            best_j, best_d = None, 1e9
            _, (lt, lx, ly) = tr.cls, tr.history[-1]
            for j in unmatched:
                lab,(cx,cy) = det_centroids[j]
                if lab != tr.cls: continue
                d = math.hypot(cx-lx, cy-ly)
                if d < best_d: best_d, best_j = d, j
            if best_j is not None and best_d <= self.max_dist:
                lab,(cx,cy) = det_centroids[best_j]
                tr.history.append((now,cx,cy))
                tr.hits += 1
                if len(tr.history)>30: tr.history.popleft()
                unmatched.remove(best_j)

        for j in unmatched:
            lab,(cx,cy) = det_centroids[j]
            tr = Track(id=self.next_id, cls=lab, history=deque(maxlen=30))
            tr.history.append((now,cx,cy))
            tr.hits = 1
            self.tracks[tr.id] = tr
            self.next_id += 1

        to_del = []
        for tid,tr in self.tracks.items():
            last_t,_,_ = tr.history[-1]
            if now - last_t > self.max_age: to_del.append(tid)
        for tid in to_del: self.tracks.pop(tid,None)

    def speeds_mps(self, ppm):
        out = {}
        for tid,tr in self.tracks.items():
            if tr.hits < MIN_TRACK_HITS or len(tr.history)<2: continue
            pts = list(tr.history)[-SPEED_AVG_WINDOW:]
            ds,dt = 0.0,0.0
            for (t1,x1,y1),(t2,x2,y2) in zip(pts,pts[1:]):
                ds += math.hypot(x2-x1,y2-y1)
                dt += (t2-t1)
            if dt<=0: continue
            out[tid] = (ds/ppm)/dt
        return out

# ---------- YOLO ----------
model = YOLO(YOLO_MODEL)
try: model.fuse()
except Exception: pass

def yolo_detect(frame, conf, img_size, classes=None):
    res = model.predict(
        frame,
        conf=conf,
        imgsz=img_size,
        verbose=False,
        max_det=50,
        classes=None if DISABLE_CLASS_FILTER else classes
    )
    boxes=[]
    r0 = res[0]
    names = r0.names
    for b in r0.boxes:
        cls_id = int(b.cls[0])
        label = names[cls_id].lower()
        x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
        boxes.append((label,(x1,y1,x2,y2), float(b.conf[0])))
    return boxes

# ---------- DB ----------
def init_db(path):
    con = sqlite3.connect(path); cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            ped_count INTEGER,
            veh_count INTEGER,
            tl_color TEXT,
            nearest_vehicle_distance_m REAL,
            avg_vehicle_speed_mps REAL,
            action TEXT
        );
    """)
    con.commit(); con.close()

def log_event(path, ts, ped_count, veh_count, tl_color, nearest_m, avg_mps, action):
    con = sqlite3.connect(path); cur = con.cursor()
    cur.execute(
        "INSERT INTO events (ts,ped_count,veh_count,tl_color,nearest_vehicle_distance_m,avg_vehicle_speed_mps,action) VALUES (?,?,?,?,?,?,?)",
        (ts, ped_count, veh_count, tl_color, nearest_m, avg_mps, action)
    )
    con.commit(); con.close()

# ---------- TL DETECTION ----------
def detect_traffic_light_color(frame):
    x,y,w,h = TRAFFIC_LIGHT_ROI
    roi = frame[y:y+h, x:x+w]
    if roi.size==0: return "unknown"
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    def mask(hsv_img, lo, hi):
        return cv2.inRange(hsv_img, np.array(lo,np.uint8), np.array(hi,np.uint8))

    mask_red = cv2.bitwise_or(mask(hsv,*HSV_RED_1), mask(hsv,*HSV_RED_2))
    mask_y   = mask(hsv,*HSV_YELLOW)
    mask_g   = mask(hsv,*HSV_GREEN)

    r = int(np.sum(mask_red>0))
    yv= int(np.sum(mask_y>0))
    g = int(np.sum(mask_g>0))
    vals={"red":r,"yellow":yv,"green":g}
    best=max(vals,key=vals.get)
    return best if vals[best]>=50 else "unknown"

# ---------- CAM PICK ----------
def safe_camera(idx):
    x = _normalize_cam(idx)
    try:
        cap = cv2.VideoCapture(x, cv2.CAP_V4L2)
        if not cap.isOpened(): cap = cv2.VideoCapture(x, cv2.CAP_ANY)
        if not cap.isOpened(): return None
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        cap.set(cv2.CAP_PROP_FPS,6)
        ok,_ = cap.read()
        cap.release()
        return x if ok else None
    except Exception:
        return None

def pick_cameras():
    ped_env, veh_env, tl_env = os.getenv("SC_CAM_PED"), os.getenv("SC_CAM_VEH"), os.getenv("SC_CAM_TL")
    if ped_env and veh_env and tl_env:
        ped,veh,tl = _normalize_cam(ped_env), _normalize_cam(veh_env), _normalize_cam(tl_env)
        ok = [safe_camera(ped), safe_camera(veh), safe_camera(tl)]
        if all(o is not None for o in ok): return ped,veh,tl
        raise RuntimeError(f"SC_CAM_* failed: {ok}")

    found=[]
    for i in range(10):
        ok = safe_camera(i)
        if ok is not None: found.append(ok)
        if len(found)>=3: break
    if len(found)<3: raise RuntimeError(f"Found only {len(found)} working cams: {found}")
    return found[0], found[1], found[2]

# ---------- YOLO WORKERS ----------
def yolo_worker(frame_q: Queue, result_q: Queue, classes: List[int]):
    while True:
        frame = frame_q.get()
        if frame is None: break
        try:
            boxes = yolo_detect(frame, YOLO_CONF, YOLO_IMG_SZ, classes=classes)
        except Exception as e:
            boxes = []
            if PRINT_DEBUG: print("[YOLO worker err]", repr(e))
        while not result_q.empty():
            try: result_q.get_nowait()
            except Exception: break
        result_q.put(boxes)

# ---------- PIPELINE ----------
def run_pipeline():
    init_db(DB_PATH)

    i_ped, i_veh, i_tl = pick_cameras()
    cam_ped = CameraStream(i_ped, FRAME_W, FRAME_H, FPS_TARGET)
    cam_veh = CameraStream(i_veh, FRAME_W, FRAME_H, FPS_TARGET)
    cam_tl  = CameraStream(i_tl,  480,     270,     max(8, FPS_TARGET//2))
    print(f"[Pedestrian Cam] {i_ped}")
    print(f"[Vehicle Cam]    {i_veh}")
    print(f"[Traffic Light]  {i_tl}")

    # Wait first frame (non-fatal)
    def wait_first(c,n, t=3.0):
        t0=time.time()
        while time.time()-t0<t:
            ok,fr=c.read()
            if ok and fr is not None: return True
            time.sleep(0.05)
        print(f"[WARN] {n} first frame timeout"); return False
    wait_first(cam_ped,"Ped")
    wait_first(cam_veh,"Veh")
    wait_first(cam_tl,"TL")

    # Queues/workers
    ped_frames, ped_results = Queue(maxsize=1), Queue(maxsize=1)
    veh_frames, veh_results = Queue(maxsize=1), Queue(maxsize=1)
    threading.Thread(target=yolo_worker, args=(ped_frames,ped_results,COCO_PED), daemon=True).start()
    threading.Thread(target=yolo_worker, args=(veh_frames,veh_results,COCO_VEH), daemon=True).start()

    ped_last_ts = 0.0
    veh_last_ts = 0.0
    veh_tracker = CentroidTracker()

    frame_idx = 0
    last_log_ts = 0.0
    last_status_ts = 0.0

    try:
        while True:
            loop_start = time.time()
            frame_idx += 1

            rp, fp = cam_ped.read()
            rv, fv = cam_veh.read()
            rt, ft = cam_tl.read()

            ok_ped = bool(rp and fp is not None)
            ok_veh = bool(rv and fv is not None)
            ok_tl  = bool(rt and ft is not None)
            online = ok_ped or ok_veh or ok_tl

            blank_640x360 = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
            blank_480x270 = np.zeros((270, 480, 3), dtype=np.uint8)
            fp_s = fp if ok_ped else blank_640x360
            fv_s = fv if ok_veh else blank_640x360
            ft_s = ft if ok_tl  else blank_480x270

            # enqueue latest frames at SKIP cadence
            if frame_idx % max(1,SKIP_FRAMES)==0:
                if ok_ped: q_replace_latest(ped_frames, fp_s.copy())
                if ok_veh: q_replace_latest(veh_frames, fv_s.copy())

            # pull results if available
            det_p = ped_results.get_nowait() if not ped_results.empty() else []
            if det_p: ped_last_ts = time.time()
            det_v = veh_results.get_nowait() if not veh_results.empty() else []
            if det_v: veh_last_ts = time.time()

            # fallback sync detect if worker stale
            now_chk = time.time()
            if ok_ped and (now_chk - ped_last_ts) > FALLBACK_SYNC_AFTER_S:
                det_p = yolo_detect(fp_s, YOLO_CONF, YOLO_IMG_SZ, classes=None if DISABLE_CLASS_FILTER else COCO_PED)
                ped_last_ts = now_chk
            if ok_veh and (now_chk - veh_last_ts) > FALLBACK_SYNC_AFTER_S:
                det_v = yolo_detect(fv_s, YOLO_CONF, YOLO_IMG_SZ, classes=None if DISABLE_CLASS_FILTER else COCO_VEH)
                veh_last_ts = now_chk

            # ---- Pedestrians ----
            ped_count = 0
            if ok_ped and det_p:
                det_p = [(lab,b,cf) for (lab,b,cf) in det_p if (lab in PEDESTRIAN_CLASSES) or DISABLE_CLASS_FILTER]
                ped_count = len(det_p)
                for lab,(x1,y1,x2,y2),cf in det_p:
                    cv2.rectangle(fp_s,(x1,y1),(x2,y2),COLOR_GREEN,2)
                    cv2.putText(fp_s,f"{lab} {cf:.2f}",(x1,y1-6),cv2.FONT_HERSHEY_SIMPLEX,0.45,COLOR_GREEN,1)
            cv2.putText(fp_s,f"Pedestrians: {ped_count}",(8,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,COLOR_WHITE,2)

            # ---- Vehicles ----
            veh_count = 0
            nearest_vehicle_distance_m = float('inf')
            avg_speed_mps = 0.0

            if ok_veh and det_v:
                det_v = [(lab,b,cf) for (lab,b,cf) in det_v if (lab in VEHICLE_CLASSES) or DISABLE_CLASS_FILTER]
                veh_count = len(det_v)

                now_ts = time.time()
                veh_tracker.update([(lab,b) for lab,b,_ in det_v], now_ts)
                speed_by_tid = veh_tracker.speeds_mps(PIXELS_PER_METER_VEH)

                for lab,(x1,y1,x2,y2),cf in det_v:
                    cv2.rectangle(fv_s,(x1,y1),(x2,y2),COLOR_RED,2)
                    bx,by = (x1+x2)//2, (y1+y2)//2
                    best_tid,best_d = None,1e9
                    for tid,tr in veh_tracker.tracks.items():
                        if tr.cls != lab or not tr.history: continue
                        _,cx,cy = tr.history[-1]
                        d = math.hypot(bx-cx, by-cy)
                        if d<best_d: best_d,best_tid = d,tid
                    speed_mps = speed_by_tid.get(best_tid,0.0) if best_tid is not None else 0.0

                    cx,cy = (x1+x2)//2, y2
                    dy_px  = max(0, PEDESTRIAN_LANE_Y - cy)
                    dist_m = abs(dy_px)/PIXELS_PER_METER_VEH
                    nearest_vehicle_distance_m = min(nearest_vehicle_distance_m, dist_m)
                    cv2.line(fv_s,(cx,cy),(cx,PEDESTRIAN_LANE_Y),COLOR_YELLOW,1)

                    label = f"{lab} {cf:.2f} | {speed_mps*3.6:.0f} km/h | {dist_m:.1f} m"
                    cv2.putText(fv_s,label,(x1,max(12,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,0.45,COLOR_RED,1)

                if speed_by_tid: avg_speed_mps = float(np.mean(list(speed_by_tid.values())))

                for tid,tr in veh_tracker.tracks.items():
                    if not tr.history: continue
                    _,cx,cy = tr.history[-1]
                    s = speed_by_tid.get(tid,0.0)
                    cv2.circle(fv_s,(cx,cy),3,COLOR_BLUE,-1)
                    cv2.putText(fv_s,f"ID {tid} {s:.1f} m/s",(cx+6,cy-6),cv2.FONT_HERSHEY_SIMPLEX,0.45,COLOR_BLUE,1)

            cv2.putText(fv_s,"Vehicles: {}".format(veh_count),(8,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,COLOR_WHITE,2)
            cv2.line(fv_s,(0,PEDESTRIAN_LANE_Y),(FRAME_W,PEDESTRIAN_LANE_Y),COLOR_YELLOW,2)

            # ---- TL color ----
            tl_color = detect_traffic_light_color(ft_s) if ok_tl else "unknown"
            cv2.putText(ft_s,f"TL: {tl_color.upper()}",(8,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,COLOR_WHITE,2)
            x,y,w,h = TRAFFIC_LIGHT_ROI
            cv2.rectangle(ft_s,(x,y),(x+w,y+h),COLOR_WHITE,2)

            # Publish frames throttled
            publish_frame_throttled("ped", fp_s)
            publish_frame_throttled("veh", fv_s)
            publish_frame_throttled("tl",  ft_s)

            # Status (throttled)
            now = time.time()
            nearest_m = 0.0 if nearest_vehicle_distance_m==float('inf') else nearest_vehicle_distance_m
            if now - last_status_ts >= STATUS_MIN_PERIOD:
                flags = {"night": time.localtime(now).tm_hour >= 21, "rush": time.localtime(now).tm_hour == 7}
                extra = {"ambulance": False}
                publish_status_from_loop(
                    now_ts=now,
                    ped_count=ped_count,
                    veh_count=veh_count,
                    tl_color=tl_color,
                    nearest_m=nearest_m,
                    avg_mps=avg_speed_mps,
                    flags=flags,
                    extra=extra,
                )
                last_status_ts = now

            # LED action
            action = "OFF"
            veh_close = (nearest_vehicle_distance_m < VEHICLE_CLOSE_THRESH_M) if ok_veh else False
            if tl_color == "red":
                action = "STOP"
            elif tl_color == "green":
                if ped_count > 0 and veh_close: action = "STOP"
                elif ped_count > 0: action = "GO"
            elif tl_color == "yellow":
                action = "STOP" if ped_count>0 else "OFF"
            else:
                if ped_count>0 and (veh_close or veh_count>0): action="STOP"
                elif ped_count>0: action="GO"
            show_led(action)

            # Overlay quick stats
            cv2.putText(fv_s,f"Nearest dist: {nearest_m:.1f} m",(8,44),cv2.FONT_HERSHEY_SIMPLEX,0.6,COLOR_WHITE,2)
            cv2.putText(fv_s,f"Avg speed: {avg_speed_mps:.1f} m/s",(8,68),cv2.FONT_HERSHEY_SIMPLEX,0.6,COLOR_WHITE,2)

            # DB logging (time-based)
            if now - last_log_ts >= LOG_EVERY_SEC:
                log_event(DB_PATH, now, int(ped_count), int(veh_count), tl_color, float(nearest_m), float(avg_speed_mps), action)
                last_log_ts = now

            if SHOW_WINDOWS:
                cv2.imshow("Ped", fp_s); cv2.imshow("Veh", fv_s); cv2.imshow("TL", ft_s)
                if (cv2.waitKey(1)&0xFF)==27: break

            elapsed = time.time() - loop_start
            if elapsed < FRAME_TIME: time.sleep(FRAME_TIME - elapsed)

    finally:
        for c in (cam_ped, cam_veh, cam_tl):
            try: c.stop()
            except Exception: pass
        try: ped_frames.put(None); veh_frames.put(None)
        except Exception: pass
        if SHOW_WINDOWS: cv2.destroyAllWindows()

if __name__ == "__main__":
    threading.Thread(target=run_pipeline, daemon=True).start()
    start_http_server(host="0.0.0.0", port=5000)
