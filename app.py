import os
import time
import json
import sqlite3
import threading
import urllib.request
import urllib.parse
from datetime import datetime
from contextlib import contextmanager
from dataclasses import dataclass, field

import cv2
import numpy as np
import pymysql
from ultralytics import YOLO
from flask import Flask, Response, render_template_string, jsonify, request, session, redirect, abort

# ==============================================================================
# KONFIGURASI SISTEM ITS NASIONAL & MARIADB (DBEAVER)
# ==============================================================================
DEVICE = "cpu"
print("🚀 Memproses YOLOv8 & MariaDB Integration untuk DBeaver")

DB_CONFIG = {
    "host": "localhost",
    "user": "root",       
    "password": "",       
    "database": "its_traffic_db",
    "charset": "utf8mb4"
}

VEHICLE_CLASSES = {
    1: "Sepeda Motor", 
    3: "Sepeda Motor", 
    2: "Mobil Penumpang", 
    5: "Bus Besar", 
    7: "Truk Barang"
}

CLASS_COLORS = {
    "Sepeda Motor": (0, 215, 255),      
    "Mobil Penumpang": (255, 128, 0),    
    "Kendaraan Sedang": (0, 140, 255),   
    "Bus Besar": (211, 0, 148),          
    "Truk Barang": (0, 204, 50)          
}

DEFAULT_SOURCE_FPS = 25.0

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

CAMERAS_JSON_PATH = os.path.join(DATA_DIR, "cameras_config.json")
USERS_JSON_PATH = os.path.join(DATA_DIR, "users_config.json")

def load_users_db():
    default_users = {
        "admin": {
            "password": "admin123",
            "name": "Administrator ITS",
            "role": "admin",
            "last_active": 0
        }
    }
    if os.path.exists(USERS_JSON_PATH):
        try:
            with open(USERS_JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and len(data) > 0:
                    for u_key in data:
                        if "last_active" not in data[u_key]:
                            data[u_key]["last_active"] = 0
                    return data
        except Exception:
            pass
    save_users_db(default_users)
    return default_users

def save_users_db(users_dict):
    try:
        with open(USERS_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(users_dict, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

USERS_DB = load_users_db()

@dataclass
class CameraConfig:
    cam_id: str
    name: str
    ruas_name: str
    lat: float
    lng: float
    stream_url: str
    output_dir: str
    model_path: str = "yolov8n.pt"         
    confidence_threshold: float = 0.01     
    iou_threshold: float = 0.40
    infer_imgsz: int = 640                 
    max_stream_height: int = 360           
    prefer_highest_quality_stream: bool = False

    zones_config_path: str = field(init=False, default="")
    tracker_config_path: str = field(init=False, default="")

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        self.zones_config_path = os.path.join(self.output_dir, "zones_config.json")
        self.tracker_config_path = os.path.join(self.output_dir, "bytetrack_custom.yaml")
        ensure_custom_tracker_config(self.tracker_config_path)

def ensure_custom_tracker_config(path: str):
    content = (
        "tracker_type: bytetrack\n"
        "track_high_thresh: 0.04\n"
        "track_low_thresh: 0.01\n"
        "new_track_thresh: 0.04\n"
        "track_buffer: 50\n"
        "match_thresh: 0.75\n"
        "fuse_score: True\n"
    )
    with open(path, "w", encoding="utf-8") as f: 
        f.write(content)

def load_cameras_registry():
    default_cams = [
        {
            "cam_id": "gunung_pasir",
            "name": "CCTV Simpang Gunung Pasir",
            "ruas_name": "GUNUNG PASIR - BALIKPAPAN",
            "lat": -1.2675, "lng": 116.8312,
            "stream_url": "https://cctv.balikpapan.go.id/sp_gunung_pasir_hd/main_stream.m3u8"
        },
        {
            "cam_id": "lapangan_foni",
            "name": "CCTV Lapangan Foni KB Sayur",
            "ruas_name": "LAPANGAN FONI - BALIKPAPAN",
            "lat": -1.2750, "lng": 116.8280,
            "stream_url": "https://cctv.balikpapan.go.id/sp_gunung_pasir_hd/main_stream.m3u8"
        },
        {
            "cam_id": "pasar_buton",
            "name": "CCTV Pasar Buton Fix HD",
            "ruas_name": "PASAR BUTON - BALIKPAPAN",
            "lat": -1.2550, "lng": 116.8500,
            "stream_url": "https://cctv.balikpapan.go.id/sp_gunung_pasir_hd/main_stream.m3u8"
        },
        {
            "cam_id": "gunung_malang",
            "name": "CCTV Simpang Gunung Malang",
            "ruas_name": "GUNUNG MALANG - BALIKPAPAN",
            "lat": -1.2450, "lng": 116.8350,
            "stream_url": "https://cctv.balikpapan.go.id/sp_gunung_pasir_hd/main_stream.m3u8"
        },
        {
            "cam_id": "gunung_agung_bali",
            "name": "CCTV Simpang Gunung Agung Denpasar",
            "ruas_name": "SIMPANG GUNUNG AGUNG - KOTA DENPASAR, BALI",
            "lat": -8.6500, "lng": 115.2167,
            "stream_url": "https://atcs.denpasarkota.go.id/stream/A001GUNUNGAGUNGPTZ/stream.m3u8"
        },
        {
            "cam_id": "kapten_sujana_bali",
            "name": "CCTV Simpang Kapten Sujana Denpasar",
            "ruas_name": "SIMPANG KAPTEN SUJANA - KOTA DENPASAR, BALI",
            "lat": -8.6525, "lng": 115.2200,
            "stream_url": "https://atcs.denpasarkota.go.id/stream/A002KAPTENSUJANAPTZ/stream.m3u8"
        },
        {
            "cam_id": "gunung_salak_bali",
            "name": "CCTV Simpang Gunung Salak Denpasar",
            "ruas_name": "SIMPANG GUNUNG SALAK - KOTA DENPASAR, BALI",
            "lat": -8.6700, "lng": 115.1950,
            "stream_url": "https://atcs.denpasarkota.go.id/stream/A003GUNUNGSALAKPTZ/stream.m3u8"
        },
        {
            "cam_id": "cctv_sidakarya_bedugul",
            "name": "CCTV SIDAKARYA - BEDUGUL",
            "ruas_name": "JALAN SIDAKARYA - BEDUGUL",
            "lat": -8.6500, "lng": 115.2167,
            "stream_url": "https://atcs.denpasarkota.go.id/stream/A001GUNUNGAGUNGPTZ/stream.m3u8"
        }
    ]
    save_cameras_registry(default_cams)
    return default_cams

def save_cameras_registry(cams_list):
    try:
        with open(CAMERAS_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(cams_list, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

def load_json_config(path: str):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f: 
                data = json.load(f)
            if isinstance(data, list) and len(data) > 0:
                return [{"label": z.get("label", "Garis Hitung"), "points": [tuple(p) for p in z["points"]], "color": tuple(z.get("color", (0, 255, 127)))} for z in data]
        except Exception: 
            pass
    return []

def save_json_config(path: str, zones: list):
    try:
        serializable = [{"label": z["label"], "points": [list(p) for p in z["points"]], "color": list(z["color"])} for z in zones]
        with open(path, "w", encoding="utf-8") as f: 
            json.dump(serializable, f, indent=2, ensure_ascii=False)
    except Exception: 
        pass

def resolve_best_hls_variant(master_url: str, max_height: int, prefer_highest: bool, timeout: float = 5.0) -> str:
    try:
        req = urllib.request.Request(master_url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urllib.request.urlopen(req, timeout=timeout) as resp: 
            content = resp.read().decode("utf-8", errors="ignore")
        if "#EXT-X-STREAM-INF" not in content: 
            return master_url
        lines = content.splitlines()
        candidates = []
        for i, line in enumerate(lines):
            if not line.startswith("#EXT-X-STREAM-INF"): 
                continue
            bw, height, res = None, None, ""
            for attr in line.split(","):
                attr = attr.strip()
                if attr.startswith("BANDWIDTH="):
                    try: 
                        bw = int(attr.split("=", 1)[1])
                    except ValueError: 
                        pass
                elif attr.startswith("RESOLUTION="):
                    res = attr.split("=", 1)[1]
                    try: 
                        height = int(res.lower().split("x")[1])
                    except Exception: 
                        pass
            if bw is not None and i + 1 < len(lines):
                uri_line = lines[i + 1].strip()
                if uri_line and not uri_line.startswith("#"): 
                    candidates.append((bw, height, uri_line, res))
        if not candidates: 
            return master_url
        capped = [c for c in candidates if c[1] is not None and c[1] <= max_height]
        best_uri = min(capped, key=lambda c: c[0])[2] if capped else min(candidates, key=lambda c: c[0])[2]
        return urllib.parse.urljoin(master_url, best_uri)
    except Exception: 
        return master_url

def denormalize_points(points, width, height): 
    return [(int(x * width), int(y * height)) for x, y in points]

GLOBAL_YOLO_MODEL = None

class SpatialFilteredAIEngine:
    def __init__(self, cfg: CameraConfig):
        global GLOBAL_YOLO_MODEL
        self.cfg = cfg
        if GLOBAL_YOLO_MODEL is None:
            GLOBAL_YOLO_MODEL = YOLO(cfg.model_path)
        self.model = GLOBAL_YOLO_MODEL

    def process_frame(self, frame):
        try:
            results = self.model.track(
                frame, persist=True, tracker=self.cfg.tracker_config_path,
                conf=self.cfg.confidence_threshold, iou=self.cfg.iou_threshold,
                classes=list(VEHICLE_CLASSES.keys()), verbose=False, device=DEVICE,
                imgsz=self.cfg.infer_imgsz, half=False
            )[0]
            boxes = []
            if results.boxes is not None:
                ids = results.boxes.id
                for i, box in enumerate(results.boxes):
                    cls_id = int(box.cls[0])
                    raw_label = VEHICLE_CLASSES.get(cls_id)
                    if not raw_label: 
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    track_id = int(ids[i]) if ids is not None else i
                    boxes.append((x1, y1, x2, y2, raw_label, conf, track_id))
            return boxes
        except Exception:
            return []

class SpatialFilteredStreamWorker:
    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        self.src = cfg.stream_url
        self._resolved_src = None
        self.cap = None
        self.width, self.height, self.source_fps = 640, 360, DEFAULT_SOURCE_FPS
        self.stopped = True 
        self.ai_engine = None
        self._latest_jpeg = None
        self.last_access_time = 0 
        
        self.counted_track_ids = set()
        self.track_history = {}
        self.zone_counts = {}

        self._latest_boxes = []
        self._boxes_lock = threading.Lock()
        self._is_ai_processing = False

        self._frame_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._stats = {
            "cam_id": cfg.cam_id, "name": cfg.name, "ruas_name": cfg.ruas_name, "lat": cfg.lat, "lng": cfg.lng, "connected": False,
            "total_frame": 0, "total_kumulatif": 0,
            "counts_frame": {"Sepeda Motor": 0, "Mobil Penumpang": 0, "Kendaraan Sedang": 0, "Bus Besar": 0, "Truk Barang": 0},
            "counts_opposite": {"Sepeda Motor": 0, "Mobil Penumpang": 0, "Kendaraan Sedang": 0, "Bus Besar": 0, "Truk Barang": 0},
            "status": "Lancar", "updated_at": None, "zone_counts": {}
        }
        self.zones_lock = threading.Lock()
        self.zones = load_json_config(cfg.zones_config_path)
        self.thread = None

    def start_if_needed(self):
        self.last_access_time = time.time()
        if self.stopped:
            self.stopped = False
            if self.ai_engine is None:
                self.ai_engine = SpatialFilteredAIEngine(self.cfg)
            self.thread = threading.Thread(target=self._stream_loop, daemon=True)
            self.thread.start()

    def _connect(self):
        if self.cap is not None: 
            self.cap.release()
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "timeout;1000000|reconnect;1|reconnect_streamed;1"
        if self._resolved_src is None:
            if isinstance(self.src, str) and self.src.lower().split("?")[0].endswith(".m3u8"):
                self._resolved_src = resolve_best_hls_variant(self.src, self.cfg.max_stream_height, self.cfg.prefer_highest_quality_stream)
            else: 
                self._resolved_src = self.src
                
        self.cap = cv2.VideoCapture(self._resolved_src, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if self.cap.isOpened():
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 360

    def _async_ai_worker(self, frame_copy):
        boxes = self.ai_engine.process_frame(frame_copy)
        with self._boxes_lock:
            self._latest_boxes = boxes
        self._is_ai_processing = False

    def _stream_loop(self):
        self._connect()
        frame_counter = 0

        while not self.stopped:
            if time.time() - self.last_access_time > 10.0:
                self.stopped = True
                break

            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.5)
                self._connect()
                continue
            
            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.02)
                self._connect()
                continue

            frame_counter += 1

            if frame_counter % 3 == 0 and not self._is_ai_processing:
                self._is_ai_processing = True
                t = threading.Thread(target=self._async_ai_worker, args=(frame.copy(),), daemon=True)
                t.start()

            with self._boxes_lock:
                current_boxes = list(self._latest_boxes)

            with self.zones_lock: 
                current_zones = list(self.zones)

            frame_w, frame_h = self.width, self.height

            line_segments = []
            for zone in current_zones:
                pts = denormalize_points(zone["points"], frame_w, frame_h)
                lbl = zone.get("label", "Garis Hitung")
                count_val = self.zone_counts.get(lbl, 0)

                for i in range(len(pts) - 1): 
                    cv2.line(frame, pts[i], pts[i + 1], zone["color"], 2, cv2.LINE_AA)
                    line_segments.append((pts[i], pts[i + 1], lbl))

            normal_counts = {"Sepeda Motor": 0, "Mobil Penumpang": 0, "Kendaraan Sedang": 0, "Bus Besar": 0, "Truk Barang": 0}
            opposite_counts = {"Sepeda Motor": 0, "Mobil Penumpang": 0, "Kendaraan Sedang": 0, "Bus Besar": 0, "Truk Barang": 0}

            active_ids_in_frame = set()

            for (x1, y1, x2, y2, label, conf, track_id) in current_boxes:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                active_ids_in_frame.add(track_id)
                prev_y = self.track_history.get(track_id, cy)
                self.track_history[track_id] = cy

                color = CLASS_COLORS.get(label, (0, 255, 0))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                tag = f"{label} #{track_id}"
                
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(frame, (x1, max(y1 - 16, 0)), (x1 + tw + 4, max(y1, 16)), color, -1)
                cv2.putText(frame, tag, (x1 + 2, max(y1 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

                if track_id not in self.counted_track_ids and line_segments:
                    for line_data in line_segments:
                        p1, p2, zone_lbl = line_data[0], line_data[1], line_data[2]
                        min_x = min(p1[0], p2[0]) - 35
                        max_x = max(p1[0], p2[0]) + 35
                        line_y_ref = (p1[1] + p2[1]) // 2
                        
                        if min_x <= cx <= max_x:
                            if (prev_y <= line_y_ref <= cy) or (prev_y >= line_y_ref >= cy) or (abs(cy - line_y_ref) <= 18):
                                self.counted_track_ids.add(track_id)
                                self.zone_counts[zone_lbl] = self.zone_counts.get(zone_lbl, 0) + 1
                                
                                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                is_motor = 1 if label == "Sepeda Motor" else 0
                                is_mobil = 1 if label == "Mobil Penumpang" else 0
                                is_truk = 1 if label == "Truk Barang" else 0
                                is_bus = 1 if label == "Bus Besar" else 0
                                
                                try:
                                    with get_connection() as conn:
                                        with conn.cursor() as cursor:
                                            cursor.execute(
                                                """
                                                INSERT INTO crossing_log 
                                                (timestamp, stream_name, zone_label, mobil, motor, truk, bus, total)
                                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                                                """,
                                                (now_str, self.cfg.name, zone_lbl, is_mobil, is_motor, is_truk, is_bus, 1)
                                            )
                                        conn.commit()
                                except Exception:
                                    pass
                                break

                if cy > (frame_h // 2):
                    normal_counts[label] += 1
                else:
                    opposite_counts[label] += 1

            stale_ids = [tid for tid in self.track_history if tid not in active_ids_in_frame]
            for tid in stale_ids:
                self.track_history.pop(tid, None)

            total_frame = sum(normal_counts.values()) + sum(opposite_counts.values())
            status = "MACET" if total_frame >= 20 else ("PADAT MERAYAP" if total_frame >= 15 else "Lancar")

            ok_enc, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 45])
            if ok_enc:
                with self._frame_lock: 
                    self._latest_jpeg = buf.tobytes()

            with self._stats_lock:
                self._stats.update({
                    "connected": True, 
                    "total_frame": total_frame, 
                    "total_kumulatif": len(self.counted_track_ids),
                    "counts_frame": normal_counts, 
                    "counts_opposite": opposite_counts,
                    "status": status, 
                    "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "zone_counts": dict(self.zone_counts)
                })

            time.sleep(0.01)

        if self.cap is not None:
            self.cap.release()

    def get_jpeg(self):
        self.last_access_time = time.time()
        with self._frame_lock: 
            return self._latest_jpeg

    def get_stats(self):
        self.last_access_time = time.time()
        with self._stats_lock: 
            return dict(self._stats)

    def update_zones(self, new_zones):
        cleaned = []
        for idx, z in enumerate(new_zones):
            lbl = z.get("label", f"Garis_{idx+1}").strip()
            if not lbl:
                lbl = f"Garis_{idx+1}"
            pts = [tuple(p) for p in z.get("points", [])]
            clr = tuple(z.get("color", (0, 255, 127)))
            if len(pts) >= 2:
                cleaned.append({"label": lbl, "points": pts, "color": clr})
        with self.zones_lock:
            self.zones = cleaned
            save_json_config(self.cfg.zones_config_path, cleaned)

@contextmanager
def get_connection():
    conn = pymysql.connect(
        host=DB_CONFIG["host"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        database=DB_CONFIG["database"],
        charset=DB_CONFIG["charset"]
    )
    try: 
        yield conn
    finally: 
        conn.close()

def init_db():
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS traffic_log (
                    id INT AUTO_INCREMENT PRIMARY KEY, 
                    timestamp VARCHAR(50), 
                    stream_name VARCHAR(100), 
                    mobil_frame INT, 
                    motor_frame INT, 
                    truk_frame INT, 
                    bus_frame INT, 
                    total_frame INT, 
                    status VARCHAR(50), 
                    total_kumulatif INT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS crossing_log (
                    id INT AUTO_INCREMENT PRIMARY KEY, 
                    timestamp VARCHAR(50), 
                    stream_name VARCHAR(100), 
                    zone_label VARCHAR(100),
                    mobil INT, 
                    motor INT, 
                    truk INT, 
                    bus INT, 
                    total INT
                )
            """)
        conn.commit()

def start_database_logger(interval_seconds=60):
    def _logger_loop():
        while True:
            time.sleep(interval_seconds)
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            for cam_id, worker in WORKERS.items():
                if not worker.stopped:
                    stats = worker.get_stats()
                    counts = stats.get("counts_frame", {})
                    
                    motor = counts.get("Sepeda Motor", 0)
                    mobil = counts.get("Mobil Penumpang", 0)
                    truk = counts.get("Truk Barang", 0)
                    bus = counts.get("Bus Besar", 0)
                    
                    total_frame = stats.get("total_frame", 0)
                    total_kumulatif = stats.get("total_kumulatif", 0)
                    status = stats.get("status", "Lancar")
                    
                    try:
                        with get_connection() as conn:
                            with conn.cursor() as cursor:
                                cursor.execute(
                                    """
                                    INSERT INTO traffic_log 
                                    (timestamp, stream_name, mobil_frame, motor_frame, truk_frame, bus_frame, total_frame, status, total_kumulatif)
                                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                                    """,
                                    (current_time, worker.cfg.name, mobil, motor, truk, bus, total_frame, status, total_kumulatif)
                                )
                            conn.commit()
                    except Exception:
                        pass

    thread = threading.Thread(target=_logger_loop, daemon=True)
    thread.start()

WORKERS = {}

def load_all_workers():
    raw_cams = load_cameras_registry()
    configs_list = []
    for c in raw_cams:
        cam_id = c["cam_id"]
        cfg = CameraConfig(
            cam_id=cam_id,
            name=c["name"],
            ruas_name=c["ruas_name"],
            lat=float(c["lat"]),
            lng=float(c["lng"]),
            stream_url=c["stream_url"],
            output_dir=os.path.join(DATA_DIR, cam_id)
        )
        configs_list.append(cfg)
        if cam_id in WORKERS:
            WORKERS[cam_id].cfg = cfg
        else:
            WORKERS[cam_id] = SpatialFilteredStreamWorker(cfg)
    return configs_list

init_db()
CAMERA_CONFIGS = load_all_workers()

app = Flask(__name__)
app.secret_key = "balikpapan_its_secret_key_2026"

def get_city_group(cam):
    ruas = (cam.get("ruas_name") or "").upper()
    name = (cam.get("name") or "").upper()
    combined = ruas + " " + name
    if "BALIKPAPAN" in combined:
        return "Balikpapan"
    elif "DENPASAR" in combined or "BALI" in combined or "BEDUGUL" in combined or "SIDAKARYA" in combined:
        return "Denpasar, Bali"
    else:
        return "Wilayah Lainnya"

json_lock = threading.Lock()

def update_user_activity_helper(username):
    if username:
        with json_lock:
            db = load_users_db()
            if username in db:
                db[username]["last_active"] = time.time()
                save_users_db(db)

@app.before_request
def update_user_activity():
    if "user" in session:
        update_user_activity_helper(session["user"].get("username"))

PUBLIC_TEMPLATE = """
<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CCTV KENDARAAN NON TOL NASIONAL - PUBLIK</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
    :root { 
        --bahau-red: #8b0000; --bahau-yellow: #ffd700; --bahau-black: #0d0d0d; 
        --bpn-blue: #0b2545; --bg-gray: #f4f6f9; --border-color: #d1d5db;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', Arial, sans-serif; }
    body { background: var(--bg-gray); color: #1f2937; }
    .dayak-bahau-pattern-bar {
        height: 14px; background-color: var(--bahau-black);
        background-image: radial-gradient(circle at 50% 50%, var(--bahau-yellow) 20%, transparent 22%), linear-gradient(45deg, var(--bahau-black) 25%, var(--bahau-yellow) 25%, var(--bahau-yellow) 50%, var(--bahau-red) 50%, var(--bahau-red) 75%, var(--bahau-black) 75%);
        background-size: 38px 14px; border-bottom: 2px solid var(--bahau-yellow);
    }
    .top-bar { background: var(--bahau-black); color: var(--bahau-yellow); padding: 8px 30px; font-size: 12px; display: flex; justify-content: space-between; font-weight: bold; align-items: center; }
    header { background: linear-gradient(135deg, var(--bpn-blue), #061527); color: #fff; padding: 14px 30px; display: flex; align-items: center; justify-content: space-between; border-bottom: 4px solid var(--bahau-red); }
    .logo-container { display: flex; align-items: center; gap: 16px; }
    .manuntung-logo { width: 48px; height: 48px; background: #ffffff; border-radius: 50%; display: flex; align-items: center; justify-content: center; border: 2px solid var(--bahau-yellow); }
    .manuntung-logo svg { width: 100%; height: 100%; }
    .logo-text h2 { font-size: 18px; color: var(--bahau-yellow); font-weight: 800; }
    .logo-text p { font-size: 11px; color: #9ca3af; }
    .auth-group { display: flex; gap: 8px; align-items: center; }
    .auth-btn { background: var(--bahau-yellow); color: #000; padding: 6px 14px; border-radius: 4px; border: none; font-weight: bold; cursor: pointer; font-size: 12px; }
    .register-btn { background: #10b981; color: #fff; padding: 6px 14px; border-radius: 4px; border: none; font-weight: bold; cursor: pointer; font-size: 12px; }
    .hero-banner { background: linear-gradient(rgba(11, 37, 69, 0.9), rgba(13, 13, 13, 0.92)); padding: 25px 20px; text-align: center; color: #fff; border-bottom: 4px solid var(--bahau-yellow); }
    .hero-banner h1 { font-size: 24px; font-weight: 900; color: var(--bahau-yellow); margin-bottom: 4px; }
    
    .filter-section { max-width: 1100px; margin: 25px auto; background: #fff; padding: 18px; border-radius: 10px; box-shadow: 0 6px 20px rgba(0,0,0,0.12); display: flex; gap: 15px; align-items: center; border: 2px solid var(--bahau-yellow); }
    .filter-section label { font-size: 13px; font-weight: bold; color: var(--bpn-blue); }
    .filter-section input { padding: 10px; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 13px; flex: 1; }
    .btn-search { background: linear-gradient(to right, var(--bahau-red), #e74c3c); color: #fff; border: none; padding: 10px 28px; border-radius: 6px; cursor: pointer; font-weight: bold; }

    .container { max-width: 1280px; margin: 25px auto; padding: 0 15px; }
    #map-container { height: 320px; width: 100%; border-radius: 10px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.15); margin-bottom: 25px; border: 3px solid var(--bpn-blue); }
    
    .city-section-title { font-size: 16px; font-weight: 800; color: var(--bpn-blue); margin: 25px 0 12px; border-left: 5px solid var(--bahau-red); padding-left: 10px; text-transform: uppercase; }
    .cards-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 18px; margin-bottom: 20px; }
    .ruas-card { background: #fff; border-radius: 10px; padding: 20px; text-align: center; border-top: 5px solid var(--bahau-yellow); box-shadow: 0 4px 12px rgba(0,0,0,0.05); }
    .ruas-card h4 { font-size: 11px; text-transform: uppercase; color: #6b7280; font-weight: bold; }
    .ruas-card h3 { font-size: 14px; color: var(--bpn-blue); margin: 5px 0 14px; font-weight: 800; }
    .btn-detail { display: inline-block; background: var(--bpn-blue); color: var(--bahau-yellow); border: none; padding: 8px 20px; border-radius: 20px; font-weight: bold; cursor: pointer; font-size: 12px; }
    
    .analytics-panel { background: #fff; border-radius: 12px; padding: 22px; box-shadow: 0 6px 25px rgba(0,0,0,0.1); display: none; border-top: 5px solid var(--bahau-red); margin-top: 25px; }
    .analytics-panel.active { display: block; }
    .video-box { background: #000; border-radius: 8px; overflow: hidden; aspect-ratio: 16/9; position: relative; border: 2px solid var(--bpn-blue); margin-bottom: 12px; }
    .video-box img { width: 100%; height: 100%; object-fit: contain; }

    .modal { display: none; position: fixed; z-index: 2000; left: 0; top: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.6); align-items: center; justify-content: center; }
    .modal-content { background: #fff; padding: 25px; border-radius: 10px; width: 350px; text-align: center; border-top: 6px solid var(--bpn-blue); position: relative; z-index: 2001; }
    .modal-content input { width: 100%; padding: 10px; margin: 8px 0; border: 1px solid #ccc; border-radius: 5px; }
    .modal-content button { background: var(--bpn-blue); color: var(--bahau-yellow); border: none; padding: 10px 20px; font-weight: bold; border-radius: 5px; cursor: pointer; width: 100%; margin-top: 6px; }
    
    .lock-notice { background: #fffbeb; border: 2px dashed #f59e0b; padding: 15px; text-align: center; border-radius: 8px; margin-top: 15px; font-size: 13px; color: #78350f; }
    footer { background: var(--bahau-black); color: var(--bahau-yellow); text-align: center; padding: 18px; font-size: 12px; margin-top: 35px; border-top: 4px solid var(--bahau-red); font-weight: bold; }
</style>
</head>
<body>
<div class="dayak-bahau-pattern-bar"></div>
<div class="top-bar">
    <span>SISTEM PEMANTAUAN KENDARAAN NON TOL NASIONAL</span>
    <div class="auth-group">
        <button class="auth-btn" onclick="openLoginModal()">🔑 Masuk untuk Panel Lengkap</button>
        <button class="register-btn" onclick="openRegisterModal()">📝 Buat Akun</button>
    </div>
</div>
<header>
    <div class="logo-container">
        <div class="manuntung-logo">
            <svg viewBox="0 0 500 500"><circle cx="250" cy="250" r="230" fill="#003366" stroke="#ffd700" stroke-width="20"/><text x="250" y="270" font-size="75" fill="#ffffff" font-weight="900" text-anchor="middle">ITS</text></svg>
        </div>
        <div class="logo-text">
            <h2>CCTV KENDARAAN NON TOL NASIONAL</h2>
            <p>Pemantauan Ruas Non Tol & Analisis Kendaraan AI</p>
        </div>
    </div>
</header>
<section class="hero-banner">
    <h1>PEMANTAUAN CCTV KENDARAAN NON TOL</h1>
</section>

<div class="filter-section">
    <label>Cari Ruas / Lokasi :</label>
    <input type="text" id="searchInput" placeholder="Cari nama ruas atau kota...">
    <button class="btn-search" onclick="filterRuas()">Cari</button>
</div>

<div class="container">
    <div id="map-container"></div>

    <div id="groupedRuasContainer">
        {% set grouped = {} %}
        {% for cam in cams %}
            {% set city = get_city_group(cam) %}
            {% if city not in grouped %}
                {% set _ = grouped.update({city: []}) %}
            {% endif %}
            {% set _ = grouped[city].append(cam) %}
        {% endfor %}

        {% for city, city_cams in grouped.items() %}
        <div class="city-group-block" data-city="{{ city }}">
            <div class="city-section-title">📍 Wilayah / Kota: {{ city }}</div>
            <div class="cards-grid">
                {% for cam in city_cams %}
                <div class="ruas-card" data-name="{{ cam.name }} - {{ cam.ruas_name }}">
                    <h4>Ruas : {{ cam.ruas_name }}</h4>
                    <h3>{{ cam.name }}</h3>
                    <button class="btn-detail" onclick="showPublicAnalytics('{{ cam.id }}', '{{ cam.name }}')">Lihat CCTV (Terbatas) ➔</button>
                </div>
                {% endfor %}
            </div>
        </div>
        {% endfor %}
    </div>

    <div class="analytics-panel" id="analyticsPanel">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
            <div>
                <div style="font-size: 11px; text-transform: uppercase; color: var(--bahau-red); font-weight: bold;">MODE PUBLIK (AKSES TERBATAS)</div>
                <h3 id="selectedCamName" style="color: var(--bpn-blue);">Detail CCTV</h3>
            </div>
            <button class="btn-detail" style="background: var(--bahau-red); color: #fff;" onclick="closeAnalytics()">✕ Tutup</button>
        </div>
        
        <div class="video-box">
            <img id="activeVideoStream" src="" alt="Live Stream">
        </div>
        
        <div class="lock-notice">
            ⚠️ <b>Fitur Terbatas:</b> Anda menonton dalam mode publik. Grafik analitik detail, rekapitulasi golongan, dan ekspor data hanya tersedia untuk akun terdaftar. 
            <button class="auth-btn" style="margin-left: 10px; padding: 4px 10px; font-size: 11px;" onclick="openLoginModal()">Login Sekarang</button>
        </div>
    </div>
</div>

<div id="loginModal" class="modal">
    <div class="modal-content">
        <h3>Login Akun</h3>
        <input type="text" id="loginUser" placeholder="Username">
        <input type="password" id="loginPass" placeholder="Password">
        <button onclick="submitLogin()">Masuk</button>
        <button onclick="closeLoginModal()" style="background:#ccc; color:#333; margin-top:8px;">Batal</button>
    </div>
</div>

<div id="registerModal" class="modal">
    <div class="modal-content">
        <h3>Daftar Akun Baru</h3>
        <input type="text" id="regName" placeholder="Nama Lengkap">
        <input type="text" id="regUser" placeholder="Username">
        <input type="password" id="regPass" placeholder="Password">
        <button onclick="submitRegister()" style="background:#10b981;">Daftar</button>
        <button onclick="closeRegisterModal()" style="background:#ccc; color:#333; margin-top:8px;">Batal</button>
    </div>
</div>

<footer>&copy; 2026 CCTV Kendaraan Non Tol Nasional</footer>

<script>
const camData = {{ cams | tojson }};
let map, markers = {};

function initMap() {
    map = L.map('map-container').setView([-3.0, 116.0], 5);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);
    camData.forEach(cam => {
        const marker = L.marker([cam.lat, cam.lng]).addTo(map);
        marker.bindPopup(`<b>${cam.name}</b><br>${cam.ruas_name}<br><button onclick="showPublicAnalytics('${cam.id}', '${cam.name}')" style="margin-top:5px; background:#0b2545; color:#ffd700; border:none; padding:4px 8px; border-radius:3px; cursor:pointer; font-weight:bold;">Lihat CCTV</button>`);
        markers[cam.id] = marker;
    });
}
window.onload = initMap;

function filterRuas() {
    const q = document.getElementById('searchInput').value.toLowerCase();
    document.querySelectorAll('.city-group-block').forEach(group => {
        let visibleCount = 0;
        group.querySelectorAll('.ruas-card').forEach(card => {
            const name = card.getAttribute('data-name').toLowerCase();
            if (name.includes(q)) {
                card.style.display = 'block';
                visibleCount++;
            } else {
                card.style.display = 'none';
            }
        });
        group.style.display = visibleCount > 0 ? 'block' : 'none';
    });
}

function showPublicAnalytics(camId, camName) {
    document.getElementById('selectedCamName').textContent = camName;
    document.getElementById('activeVideoStream').src = `/video_feed/${camId}`;
    const panel = document.getElementById('analyticsPanel');
    panel.classList.add('active');
    panel.scrollIntoView({ behavior: 'smooth' });
}

function closeAnalytics() {
    document.getElementById('analyticsPanel').classList.remove('active');
    document.getElementById('activeVideoStream').src = '';
}

function openLoginModal() { document.getElementById('loginModal').style.display = 'flex'; }
function closeLoginModal() { document.getElementById('loginModal').style.display = 'none'; }
function openRegisterModal() { document.getElementById('registerModal').style.display = 'flex'; }
function closeRegisterModal() { document.getElementById('registerModal').style.display = 'none'; }

function submitLogin() {
    const u = document.getElementById('loginUser').value;
    const p = document.getElementById('loginPass').value;
    fetch('/login', { method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body: `username=${u}&password=${p}` }).then(r => r.json()).then(res => { if(res.success) location.href = res.redirect; else alert(res.message); });
}
function submitRegister() {
    const name = document.getElementById('regName').value;
    const u = document.getElementById('regUser').value;
    const p = document.getElementById('regPass').value;
    fetch('/register', { method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body: `name=${name}&username=${u}&password=${p}` }).then(r => r.json()).then(res => { if(res.success) { alert('Berhasil!'); location.reload(); } else alert(res.message); });
}
</script>
</body>
</html>
"""

USER_TEMPLATE = """
<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CCTV KENDARAAN NON TOL NASIONAL - DASHBOARD</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
    :root { 
        --bahau-red: #8b0000; --bahau-yellow: #ffd700; --bahau-black: #0d0d0d; 
        --bpn-blue: #0b2545; --bg-gray: #f4f6f9; --border-color: #d1d5db;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', Arial, sans-serif; }
    body { background: var(--bg-gray); color: #1f2937; }
    .dayak-bahau-pattern-bar {
        height: 14px; background-color: var(--bahau-black);
        background-image: radial-gradient(circle at 50% 50%, var(--bahau-yellow) 20%, transparent 22%), linear-gradient(45deg, var(--bahau-black) 25%, var(--bahau-yellow) 25%, var(--bahau-yellow) 50%, var(--bahau-red) 50%, var(--bahau-red) 75%, var(--bahau-black) 75%);
        background-size: 38px 14px; border-bottom: 2px solid var(--bahau-yellow);
    }
    .top-bar { background: var(--bahau-black); color: var(--bahau-yellow); padding: 8px 30px; font-size: 12px; display: flex; justify-content: space-between; font-weight: bold; align-items: center; }
    header { background: linear-gradient(135deg, var(--bpn-blue), #061527); color: #fff; padding: 14px 30px; display: flex; align-items: center; justify-content: space-between; border-bottom: 4px solid var(--bahau-red); }
    .logo-container { display: flex; align-items: center; gap: 16px; }
    .manuntung-logo { width: 48px; height: 48px; background: #ffffff; border-radius: 50%; display: flex; align-items: center; justify-content: center; border: 2px solid var(--bahau-yellow); }
    .manuntung-logo svg { width: 100%; height: 100%; }
    .logo-text h2 { font-size: 18px; color: var(--bahau-yellow); font-weight: 800; }
    .logo-text p { font-size: 11px; color: #9ca3af; }
    .auth-group { display: flex; gap: 8px; align-items: center; }
    .hero-banner { background: linear-gradient(rgba(11, 37, 69, 0.9), rgba(13, 13, 13, 0.92)); padding: 25px 20px; text-align: center; color: #fff; border-bottom: 4px solid var(--bahau-yellow); }
    .hero-banner h1 { font-size: 24px; font-weight: 900; color: var(--bahau-yellow); margin-bottom: 4px; }
    
    .filter-section { max-width: 1100px; margin: 25px auto; background: #fff; padding: 18px; border-radius: 10px; box-shadow: 0 6px 20px rgba(0,0,0,0.12); display: flex; gap: 15px; align-items: center; border: 2px solid var(--bahau-yellow); }
    .filter-section label { font-size: 13px; font-weight: bold; color: var(--bpn-blue); }
    .filter-section input { padding: 10px; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 13px; flex: 1; }
    .btn-search { background: linear-gradient(to right, var(--bahau-red), #e74c3c); color: #fff; border: none; padding: 10px 28px; border-radius: 6px; cursor: pointer; font-weight: bold; }

    .container { max-width: 1280px; margin: 25px auto; padding: 0 15px; }
    #map-container { height: 280px; width: 100%; border-radius: 10px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.15); margin-bottom: 25px; border: 3px solid var(--bpn-blue); }
    
    .city-section-title { font-size: 16px; font-weight: 800; color: var(--bpn-blue); margin: 25px 0 12px; border-left: 5px solid var(--bahau-red); padding-left: 10px; text-transform: uppercase; }
    .cards-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 18px; margin-bottom: 20px; }
    .ruas-card { background: #fff; border-radius: 10px; padding: 20px; text-align: center; border-top: 5px solid var(--bahau-yellow); box-shadow: 0 4px 12px rgba(0,0,0,0.05); }
    .ruas-card h4 { font-size: 11px; text-transform: uppercase; color: #6b7280; font-weight: bold; }
    .ruas-card h3 { font-size: 14px; color: var(--bpn-blue); margin: 5px 0 14px; font-weight: 800; }
    .btn-detail { display: inline-block; background: var(--bpn-blue); color: var(--bahau-yellow); border: none; padding: 8px 20px; border-radius: 20px; font-weight: bold; cursor: pointer; font-size: 12px; }
    
    .analytics-panel { background: #fff; border-radius: 12px; padding: 22px; box-shadow: 0 6px 25px rgba(0,0,0,0.1); display: none; border-top: 5px solid var(--bpn-blue); margin-top: 25px; }
    .analytics-panel.active { display: block; }
    .dashboard-grid { display: grid; grid-template-columns: 1.2fr 1fr; gap: 20px; }
    
    .video-box { background: #000; border-radius: 8px; overflow: hidden; aspect-ratio: 16/9; position: relative; border: 2px solid var(--bpn-blue); margin-bottom: 12px; cursor: grab; }
    .video-box:active { cursor: grabbing; }
    .video-wrapper { width: 100%; height: 100%; position: relative; display: flex; align-items: center; justify-content: center; overflow: hidden; }
    .video-box img { width: 100%; height: 100%; object-fit: contain; transform-origin: center center; transition: transform 0.05s ease-out; user-select: none; pointer-events: none; }
    
    .video-box:fullscreen { width: 100vw; height: 100vh; background: #000; display: flex; align-items: center; justify-content: center; border: none; border-radius: 0; }
    .video-box:-webkit-full-screen { width: 100vw; height: 100vh; background: #000; }
    .video-box:-moz-full-screen { width: 100vw; height: 100vh; background: #000; }

    .zoom-controls { position: absolute; bottom: 10px; right: 10px; display: flex; gap: 5px; z-index: 50; background: rgba(0,0,0,0.6); padding: 5px; border-radius: 5px; pointer-events: auto; }
    .zoom-btn { background: var(--bahau-yellow); color: #000; border: none; padding: 4px 8px; font-weight: bold; cursor: pointer; border-radius: 3px; font-size: 12px; }

    .chart-card { background: #fdfdfd; border: 1px solid var(--border-color); border-radius: 8px; padding: 15px; margin-bottom: 18px; }
    footer { background: var(--bahau-black); color: var(--bahau-yellow); text-align: center; padding: 18px; font-size: 12px; margin-top: 35px; border-top: 4px solid var(--bahau-red); font-weight: bold; }
</style>
</head>
<body>
<div class="dayak-bahau-pattern-bar"></div>
<div class="top-bar">
    <span>SISTEM PEMANTAUAN KENDARAAN NON TOL NASIONAL</span>
    <div class="auth-group">
        <span style="color:#fff; margin-right: 10px;">👤 {{ current_user.name }} (<b>{{ current_user.role.upper() }}</b>)</span>
        {% if current_user.role == 'admin' %}
            <a href="/admin/dashboard" style="color:var(--bahau-yellow); text-decoration:none; margin-right: 10px;">[ ⚙️ Panel Kontrol Admin ]</a>
        {% endif %}
        <a href="/logout" style="color:var(--bahau-yellow); text-decoration:none;">[ Logout ]</a>
    </div>
</div>
<header>
    <div class="logo-container">
        <div class="manuntung-logo">
            <svg viewBox="0 0 500 500"><circle cx="250" cy="250" r="230" fill="#003366" stroke="#ffd700" stroke-width="20"/><text x="250" y="270" font-size="75" fill="#ffffff" font-weight="900" text-anchor="middle">ITS</text></svg>
        </div>
        <div class="logo-text">
            <h2>CCTV KENDARAAN NON TOL NASIONAL</h2>
            <p>Pemantauan Ruas Non Tol & Analisis Kendaraan AI</p>
        </div>
    </div>
</header>
<section class="hero-banner">
    <h1>DASHBOARD ANALISIS KENDARAAN NON TOL</h1>
</section>

<div class="filter-section">
    <label>Cari Ruas / Lokasi :</label>
    <input type="text" id="searchInput" placeholder="Cari nama ruas atau kota...">
    <button class="btn-search" onclick="filterRuas()">Cari</button>
</div>

<div class="container">
    <div id="map-container"></div>

    <div id="groupedRuasContainer">
        {% set grouped = {} %}
        {% for cam in cams %}
            {% set city = get_city_group(cam) %}
            {% if city not in grouped %}
                {% set _ = grouped.update({city: []}) %}
            {% endif %}
            {% set _ = grouped[city].append(cam) %}
        {% endfor %}

        {% for city, city_cams in grouped.items() %}
        <div class="city-group-block" data-city="{{ city }}">
            <div class="city-section-title">📍 Wilayah / Kota: {{ city }}</div>
            <div class="cards-grid">
                {% for cam in city_cams %}
                <div class="ruas-card" data-name="{{ cam.name }} - {{ cam.ruas_name }}">
                    <h4>Ruas : {{ cam.ruas_name }}</h4>
                    <h3>{{ cam.name }}</h3>
                    <button class="btn-detail" onclick="showAnalytics('{{ cam.id }}', '{{ cam.name }}')">Pantau CCTV ➔</button>
                </div>
                {% endfor %}
            </div>
        </div>
        {% endfor %}
    </div>

    <div class="analytics-panel" id="analyticsPanel">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; flex-wrap: wrap; gap: 10px;">
            <div>
                <div style="font-size: 11px; text-transform: uppercase; color: var(--bahau-red); font-weight: bold;">KENDARAAN NON TOL TERKLASIFIKASI AI</div>
                <h3 id="selectedCamName" style="color: var(--bpn-blue);">Detail CCTV</h3>
            </div>
            
            <div style="display: flex; gap: 8px; align-items: center; flex-wrap: wrap;">
                {% if current_user.role == 'admin' %}
                    <div style="display: flex; gap: 5px; align-items: center; background: #f8fafc; padding: 5px 10px; border-radius: 6px; border: 1px solid #cbd5e1;">
                        <input type="date" id="startDate" style="padding: 4px; font-size: 11px;">
                        <input type="date" id="endDate" style="padding: 4px; font-size: 11px;">
                        <button class="btn-detail" style="background:#10b981; color:#fff; border:none; padding: 6px 12px; font-size: 11px;" onclick="downloadCsv()">📥 CSV</button>
                    </div>
                {% endif %}
                <button class="btn-detail" style="background: var(--bahau-red); color: #fff;" onclick="closeAnalytics()">✕ Tutup</button>
            </div>
        </div>

        <div class="dashboard-grid">
            <div>
                <div class="chart-card">
                    <div style="margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center;">
                        <h4 id="streamTitle" style="margin: 0;">Live Stream AI</h4>
                        <span id="trafficStatusBadge" style="padding: 4px 12px; border-radius: 4px; font-weight: bold; font-size: 11px; background: #10b981; color: #fff;">STATUS: Normal</span>
                    </div>
                    <div class="video-box" id="videoBox">
                        <div class="video-wrapper" id="videoWrapper">
                            <img id="activeVideoStream" src="" alt="Live Stream">
                        </div>
                        <div class="zoom-controls">
                            <button class="zoom-btn" onclick="zoomIn()" title="Zoom In">+</button>
                            <button class="zoom-btn" onclick="zoomOut()" title="Zoom Out">-</button>
                            <button class="zoom-btn" onclick="resetZoom()" title="Reset Zoom">Reset</button>
                            <button class="zoom-btn" onclick="toggleFullscreen()" title="Perbesar Layar / Fullscreen" style="background:#10b981; color:#fff;">⛶</button>
                        </div>
                    </div>
                </div>

                <div class="chart-card">
                    <h4>Volume Kendaraan Non Tol (kend./5 menit)</h4>
                    <canvas id="golonganBarChart" height="140"></canvas>
                </div>
            </div>

            <div>
                <div class="chart-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; border-bottom: 2px solid #e5e7eb; padding-bottom: 8px;">
                        <h4 style="margin: 0; color: var(--bpn-blue);">📊 Rekapitulasi Volume</h4>
                        <span id="updateTimestamp" style="font-size: 10px; color: #6b7280; font-weight: bold;">Update: -</span>
                    </div>

                    <div style="overflow-x: auto; margin-bottom: 15px;">
                        <table style="width: 100%; border-collapse: collapse; font-size: 12px; text-align: center;">
                            <thead>
                                <tr style="background: var(--bpn-blue); color: #ffd700;">
                                    <th style="padding: 8px; text-align: left;">Golongan</th>
                                    <th style="padding: 8px;">Normal</th>
                                    <th style="padding: 8px;">Opposite</th>
                                    <th style="padding: 8px;">Total</th>
                                </tr>
                            </thead>
                            <tbody id="vehicleTableBody">
                                <tr style="border-bottom: 1px solid #f1f5f9;"><td style="padding: 8px; text-align: left; font-weight: bold;">🏍️ Motor</td><td id="tNormalMotor">0</td><td id="tOppMotor">0</td><td id="tSumMotor">0</td></tr>
                                <tr style="border-bottom: 1px solid #f1f5f9; background: #f8fafc;"><td style="padding: 8px; text-align: left; font-weight: bold;">🚗 Mobil</td><td id="tNormalMobil">0</td><td id="tOppMobil">0</td><td id="tSumMobil">0</td></tr>
                                <tr style="border-bottom: 1px solid #f1f5f9;"><td style="padding: 8px; text-align: left; font-weight: bold;">🚐 Sedang</td><td id="tNormalSedang">0</td><td id="tOppSedang">0</td><td id="tSumSedang">0</td></tr>
                                <tr style="border-bottom: 1px solid #f1f5f9; background: #f8fafc;"><td style="padding: 8px; text-align: left; font-weight: bold;">🚌 Bus</td><td id="tNormalBus">0</td><td id="tOppBus">0</td><td id="tSumBus">0</td></tr>
                                <tr><td style="padding: 8px; text-align: left; font-weight: bold;">🚚 Truk</td><td id="tNormalTruk">0</td><td id="tOppTruk">0</td><td id="tSumTruk">0</td></tr>
                            </tbody>
                        </table>
                    </div>

                    <div style="background: linear-gradient(135deg, #0b2545, #061527); color: #fff; padding: 12px; border-radius: 8px; display: flex; justify-content: space-between; align-items: center; border-left: 5px solid var(--bahau-yellow);">
                        <div>
                            <div style="font-size: 11px; color: #9ca3af; text-transform: uppercase; font-weight: bold;">Akumulasi Unik</div>
                            <div style="font-size: 12px; color: var(--bahau-yellow);">Kendaraan Terpantau</div>
                        </div>
                        <div id="grandTotalCount" style="font-size: 26px; font-weight: 900; color: #84cc16;">0</div>
                    </div>
                </div>

                <div class="chart-card">
                    <h4>Volume Harian (kend./jam)</h4>
                    <canvas id="harianLineChart" height="110"></canvas>
                </div>
            </div>
        </div>
    </div>
</div>

<footer>&copy; 2026 CCTV Kendaraan Non Tol Nasional</footer>

<script>
const camData = {{ cams | tojson }};
let map, markers = {}, activeCamId = null;
let barChartInstance = null, lineChartInstance = null;

let zoomLevel = 1;
let panX = 0, panY = 0;
let isDragging = false;
let startX = 0, startY = 0;

const videoImg = document.getElementById('activeVideoStream');
const videoBox = document.getElementById('videoBox');

videoBox.addEventListener('wheel', function(e) {
    e.preventDefault();
    if (e.deltaY < 0) {
        zoomLevel = Math.min(zoomLevel + 0.2, 4);
    } else {
        zoomLevel = Math.max(zoomLevel - 0.2, 1);
        if (zoomLevel === 1) { 
            panX = 0; 
            panY = 0; 
        }
    }
    clampPan();
    updateTransform();
});

videoBox.addEventListener('mousedown', (e) => {
    if (zoomLevel > 1) {
        isDragging = true;
        startX = e.clientX - panX;
        startY = e.clientY - panY;
    }
});

window.addEventListener('mousemove', (e) => {
    if (!isDragging) return;
    panX = e.clientX - startX;
    panY = e.clientY - startY;
    clampPan();
    updateTransform();
});

window.addEventListener('mouseup', () => { 
    isDragging = false; 
});

function clampPan() {
    if (zoomLevel <= 1) {
        panX = 0;
        panY = 0;
        return;
    }
    const maxPanX = (videoBox.clientWidth * (zoomLevel - 1)) / (2 * zoomLevel);
    const maxPanY = (videoBox.clientHeight * (zoomLevel - 1)) / (2 * zoomLevel);

    panX = Math.max(-maxPanX, Math.min(maxPanX, panX));
    panY = Math.max(-maxPanY, Math.min(maxPanY, panY));
}

function zoomIn() {
    zoomLevel = Math.min(zoomLevel + 0.3, 4);
    clampPan();
    updateTransform();
}

function zoomOut() {
    zoomLevel = Math.max(zoomLevel - 0.3, 1);
    if (zoomLevel === 1) { 
        panX = 0; 
        panY = 0; 
    }
    clampPan();
    updateTransform();
}

function resetZoom() {
    zoomLevel = 1;
    panX = 0; 
    panY = 0;
    updateTransform();
}

function updateTransform() {
    videoImg.style.transform = `scale(${zoomLevel}) translate(${panX}px, ${panY}px)`;
}

function toggleFullscreen() {
    if (!document.fullscreenElement) {
        if (videoBox.requestFullscreen) {
            videoBox.requestFullscreen();
        } else if (videoBox.webkitRequestFullscreen) {
            videoBox.webkitRequestFullscreen();
        } else if (videoBox.msRequestFullscreen) {
            videoBox.msRequestFullscreen();
        }
    } else {
        if (document.exitFullscreen) {
            document.exitFullscreen();
        } else if (document.webkitExitFullscreen) {
            document.webkitExitFullscreen();
        }
    }
}

function initMap() {
    map = L.map('map-container').setView([-3.0, 116.0], 5);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);
    camData.forEach(cam => {
        const marker = L.marker([cam.lat, cam.lng]).addTo(map);
        marker.bindPopup(`<b>${cam.name}</b><br>${cam.ruas_name}<br><button onclick="showAnalytics('${cam.id}', '${cam.name}')" style="margin-top:5px; background:#0b2545; color:#ffd700; border:none; padding:5px 10px; border-radius:3px; cursor:pointer; font-weight:bold;">Pantau CCTV</button>`);
        markers[cam.id] = marker;
    });
}
window.onload = function() { initMap(); initCharts(); };

function filterRuas() {
    const q = document.getElementById('searchInput').value.toLowerCase();
    document.querySelectorAll('.city-group-block').forEach(group => {
        let visibleCount = 0;
        group.querySelectorAll('.ruas-card').forEach(card => {
            const name = card.getAttribute('data-name').toLowerCase();
            if (name.includes(q)) {
                card.style.display = 'block';
                visibleCount++;
            } else {
                card.style.display = 'none';
            }
        });
        group.style.display = visibleCount > 0 ? 'block' : 'none';
    });
}

function showAnalytics(camId, camName) {
    activeCamId = camId;
    resetZoom();
    document.getElementById('selectedCamName').textContent = camName;
    document.getElementById('streamTitle').textContent = `Live Streaming AI: ${camName}`;
    document.getElementById('activeVideoStream').src = `/video_feed/${camId}`;
    
    const panel = document.getElementById('analyticsPanel');
    panel.classList.add('active');
    panel.scrollIntoView({ behavior: 'smooth' });
    pollData();
}

function closeAnalytics() {
    document.getElementById('analyticsPanel').classList.remove('active');
    document.getElementById('activeVideoStream').src = '';
    activeCamId = null;
}

function initCharts() {
    const barEl = document.getElementById('golonganBarChart');
    if (barEl) {
        barChartInstance = new Chart(barEl.getContext('2d'), {
            type: 'bar',
            data: {
                labels: ['Sepeda Motor', 'Mobil Penumpang', 'Kendaraan Sedang', 'Bus Besar', 'Truk Barang'],
                datasets: [{ label: 'Normal', data: [0,0,0,0,0], backgroundColor: '#0b2545' }, { label: 'Opposite', data: [0,0,0,0,0], backgroundColor: '#00bcd4' }]
            },
            options: { responsive: true, scales: { y: { beginAtZero: true } } }
        });
    }

    const lineEl = document.getElementById('harianLineChart');
    if (lineEl) {
        lineChartInstance = new Chart(lineEl.getContext('2d'), {
            type: 'bar',
            data: {
                labels: ['00-02', '02-04', '04-06', '06-08', '08-10', '10-12', '12-14', '14-16', '16-18', '18-20'],
                datasets: [{ type: 'line', label: 'Rata-rata 7 Hari', data: [300,200,450,1200,2800,3100,2900,3300,4500,3800], borderColor: '#10b981', borderWidth: 2, fill: false }, { type: 'bar', label: 'Hari Ini', data: [280,190,480,1350,2950,3200,2850,3400,4700,3900], backgroundColor: '#0b2545' }]
            },
            options: { responsive: true, scales: { y: { beginAtZero: true } } }
        });
    }
}

async function pollData() {
    if (!activeCamId) return;
    try {
        const res = await fetch(`/api/stats/${activeCamId}`);
        const s = await res.json();
        const norm = s.counts_frame || {};
        const opp = s.counts_opposite || {};

        document.getElementById('updateTimestamp').textContent = `Update: ${s.updated_at || '-'}`;
        const categories = ['Sepeda Motor', 'Mobil Penumpang', 'Kendaraan Sedang', 'Bus Besar', 'Truk Barang'];
        const keys = ['Motor', 'Mobil', 'Sedang', 'Bus', 'Truk'];
        let absoluteGrandTotal = 0;

        keys.forEach((key, idx) => {
            const cat = categories[idx];
            const nVal = norm[cat] ?? 0;
            const oVal = opp[cat] ?? 0;
            const sumVal = nVal + oVal;
            absoluteGrandTotal += sumVal;

            document.getElementById(`tNormal${key}`).textContent = nVal;
            document.getElementById(`tOpp${key}`).textContent = oVal;
            document.getElementById(`tSum${key}`).textContent = sumVal;
        });

        document.getElementById('grandTotalCount').textContent = (s.total_kumulatif > 0) ? s.total_kumulatif : absoluteGrandTotal;

        if (barChartInstance) {
            barChartInstance.data.datasets[0].data = keys.map(k => norm[categories[keys.indexOf(k)]] ?? 0);
            barChartInstance.data.datasets[1].data = keys.map(k => opp[categories[keys.indexOf(k)]] ?? 0);
            barChartInstance.update('none');
        }
    } catch (e) {}
}
setInterval(pollData, 3000);

setInterval(() => {
    fetch('/api/heartbeat').catch(e => {});
}, 3000);

function downloadCsv() {
    if (!activeCamId) return;
    window.location.href = `/api/export_csv/${activeCamId}`;
}
</script>
</body>
</html>
"""

ADMIN_TEMPLATE = """
<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PANEL ADMIN - CCTV Kendaraan Non Tol</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
    :root { 
        --bahau-red: #8b0000; --bahau-yellow: #ffd700; --bahau-black: #0d0d0d; 
        --bpn-blue: #0b2545; --bg-gray: #f4f6f9; --text-color: #1f2937;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', Arial, sans-serif; }
    body { background: var(--bg-gray); color: var(--text-color); display: flex; flex-direction: column; height: 100vh; overflow: hidden; }
    .dayak-bahau-pattern-bar {
        height: 14px; background-color: var(--bahau-black); flex-shrink: 0;
        background-image: radial-gradient(circle at 50% 50%, var(--bahau-yellow) 20%, transparent 22%), linear-gradient(45deg, var(--bahau-black) 25%, var(--bahau-yellow) 25%, var(--bahau-yellow) 50%, var(--bahau-red) 50%, var(--bahau-red) 75%, var(--bahau-black) 75%);
        background-size: 38px 14px; border-bottom: 2px solid var(--bahau-yellow);
    }
    .top-bar { background: var(--bahau-black); color: var(--bahau-yellow); padding: 8px 30px; font-size: 12px; display: flex; justify-content: space-between; font-weight: bold; align-items: center; flex-shrink: 0; }
    header { background: linear-gradient(135deg, var(--bpn-blue), #061527); color: #fff; padding: 12px 30px; display: flex; justify-content: space-between; border-bottom: 4px solid var(--bahau-red); align-items: center; flex-shrink: 0; }
    
    /* Layout Dua Kolom */
    .admin-layout { display: flex; flex: 1; overflow: hidden; width: 100%; }
    
    /* Sidebar Kiri (Menu Navigasi Judul Saja) */
    .admin-sidebar { width: 320px; background: #0d0d0d; color: #fff; border-right: 2px solid #333; display: flex; flex-direction: column; overflow-y: auto; padding: 15px; flex-shrink: 0; }
    .sidebar-menu-title { font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 1px; margin: 15px 0 8px 5px; font-weight: bold; }
    
    .menu-btn { width: 100%; background: #1a1a1a; color: #ffd700; border: 1px solid #333; padding: 12px 15px; text-align: left; border-radius: 6px; font-weight: bold; font-size: 13px; cursor: pointer; margin-bottom: 8px; transition: 0.2s; display: flex; justify-content: space-between; align-items: center; }
    .menu-btn:hover { background: #262626; border-color: var(--bahau-yellow); }
    .menu-btn.active { background: var(--bpn-blue); color: var(--bahau-yellow); border-color: var(--bahau-yellow); }

    /* Konten Kanan */
    .admin-content { flex: 1; background: var(--bg-gray); overflow-y: auto; padding: 25px; }
    .tab-section { display: none; }
    .tab-section.active { display: block; }
    
    .admin-box { background: #ffffff; border: 2px solid var(--bahau-yellow); border-top: 5px solid var(--bahau-red); border-radius: 10px; padding: 25px; margin-bottom: 25px; box-shadow: 0 4px 15px rgba(0,0,0,0.08); }
    .admin-box h3 { color: var(--bpn-blue); margin-bottom: 15px; font-size: 18px; font-weight: 800; }
    .form-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 15px; margin-bottom: 15px; }
    .form-group label { display: block; font-size: 12px; font-weight: bold; color: var(--bpn-blue); margin-bottom: 5px; }
    .form-group input { width: 100%; padding: 10px; background: #ffffff; border: 1px solid #cbd5e1; color: #1f2937; border-radius: 6px; font-size: 13px; }
    .btn-add { background: linear-gradient(to right, var(--bahau-red), #e74c3c); color: #fff; border: none; padding: 10px 22px; border-radius: 6px; font-weight: bold; cursor: pointer; }
    .logout-btn { background: var(--bahau-yellow); color: #000; padding: 5px 12px; border-radius: 4px; text-decoration: none; font-weight: bold; }
    
    .cctv-table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 13px; }
    .cctv-table th, .cctv-table td { padding: 10px 12px; border: 1px solid #e2e8f0; text-align: left; }
    .cctv-table th { background: var(--bpn-blue); color: var(--bahau-yellow); }
    .cctv-table tr:nth-child(even) { background: #f8fafc; }

    .badge-online { background: #10b981; color: #fff; padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; }
    .badge-offline { background: #6b7280; color: #fff; padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; }
    
    .modal { display: none; position: fixed; z-index: 2000; left: 0; top: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.6); align-items: center; justify-content: center; }
    .modal-content { background: #ffffff; padding: 25px; border-radius: 10px; width: 650px; text-align: center; border-top: 6px solid var(--bahau-red); position: relative; z-index: 2001; }
    .modal-content img { width: 100%; aspect-ratio: 16/9; background: #000; border-radius: 6px; object-fit: contain; }
    
    footer { background: var(--bahau-black); color: var(--bahau-yellow); text-align: center; padding: 12px; font-size: 11px; flex-shrink: 0; border-top: 4px solid var(--bahau-red); font-weight: bold; }
</style>
</head>
<body>
<div class="dayak-bahau-pattern-bar"></div>
<div class="top-bar">
    <span>⚡ PANEL KHUSUS ADMINISTRATOR CCTV KENDARAAN NON TOL</span>
    <div><span>👤 Administrator</span> | <a href="/logout" class="logout-btn">Logout</a></div>
</div>
<header>
    <h2>⚙️ DASHBOARD KONTROL ADMIN</h2>
    <a href="/" style="color: var(--bahau-yellow); text-decoration: none; font-weight: bold; background: rgba(255,255,255,0.1); padding: 6px 14px; border-radius: 6px; display: inline-block; cursor: pointer; font-size: 12px;">← Kembali ke Beranda</a>
</header>

<div class="admin-layout">
    <!-- SIDEBAR KIRI (HANYA JUDUL MENU) -->
    <div class="admin-sidebar">
        <div class="sidebar-menu-title">Menu Navigasi Admin</div>
        
        <button class="menu-btn" onclick="switchTab('tab-tambah', this)">
            <span>➕ Tambah Titik CCTV</span> <span>➔</span>
        </button>
        <button class="menu-btn active" onclick="switchTab('tab-cctv', this)">
            <span>📋 Daftar CCTV Aktif</span> <span>➔</span>
        </button>
        <button class="menu-btn" onclick="switchTab('tab-users', this)">
            <span>👥 Status Pengguna</span> <span>➔</span>
        </button>
    </div>

    <!-- KONTEN UTAMA KANAN (TAMPIL SESUAI MENU YANG DIKLIK) -->
    <div class="admin-content">
        <!-- TAB 1: TAMBAH CCTV -->
        <div id="tab-tambah" class="tab-section">
            <div class="admin-box">
                <h3>➕ Tambah Titik CCTV Non Tol Baru ke Sistem</h3>
                <form action="/admin/add_cctv" method="POST">
                    <div class="form-grid">
                        <div class="form-group"><label>ID Unik</label><input type="text" name="cam_id" required placeholder="cctv_baru"></div>
                        <div class="form-group"><label>Nama Kamera</label><input type="text" name="name" required placeholder="CCTV Simpang Baru" style="text-transform: uppercase;"></div>
                        <div class="form-group"><label>Nama Ruas Jalan</label><input type="text" id="ruasInput" name="ruas_name" required placeholder="JL. RAYA" style="text-transform: uppercase;"></div>
                        <div class="form-group" style="grid-column: span 2;">
                            <label>URL Stream (.m3u8)</label>
                            <input type="text" name="stream_url" required placeholder="https://..." oninput="autoDetectCoordinates(this.value)">
                        </div>
                        <div class="form-group"><label>Latitude</label><input type="text" id="latInput" name="lat" required value="-1.25"></div>
                        <div class="form-group"><label>Longitude</label><input type="text" id="lngInput" name="lng" required value="116.84"></div>
                    </div>
                    <button type="submit" class="btn-add">Daftarkan CCTV</button>
                </form>
            </div>
        </div>

        <!-- TAB 2: DAFTAR CCTV AKTIF -->
        <div id="tab-cctv" class="tab-section active">
            <div class="admin-box">
                <h3>📋 Daftar CCTV Non Tol Aktif</h3>
                <table class="cctv-table">
                    <thead><tr><th>ID</th><th>Nama</th><th>Ruas</th><th>Koordinat</th><th>Aksi</th></tr></thead>
                    <tbody>
                        {% for cam in cams %}
                        <tr>
                            <td><code>{{ cam.id }}</code></td>
                            <td><b>{{ cam.name }}</b></td>
                            <td>{{ cam.ruas_name }}</td>
                            <td>{{ cam.lat }}, {{ cam.lng }}</td>
                            <td>
                                <button onclick="previewCctv('{{ cam.id }}', '{{ cam.name }}')" style="background: var(--bpn-blue); color: var(--bahau-yellow); border: none; padding: 5px 10px; border-radius: 4px; cursor: pointer; font-weight: bold; margin-right: 4px;">👁️ Lihat</button>
                                <button type="button" class="btn-edit-trigger" data-id="{{ cam.id }}" data-name="{{ cam.name }}" data-ruas="{{ cam.ruas_name }}" data-lat="{{ cam.lat }}" data-lng="{{ cam.lng }}" data-url="{{ cam.stream_url }}" style="background: #f59e0b; color: #fff; border: none; padding: 5px 10px; border-radius: 4px; cursor: pointer; font-weight: bold;">✏️ Edit</button>
                            </td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>

        <!-- TAB 3: STATUS PENGGUNA -->
        <div id="tab-users" class="tab-section">
            <div class="admin-box">
                <h3>👥 Daftar Pengguna & Status Aktivitas (Online / Offline)</h3>
                <table class="cctv-table">
                    <thead>
                        <tr>
                            <th>Username</th>
                            <th>Nama Lengkap</th>
                            <th>Role</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody id="userTableBody"></tbody>
                </table>
            </div>
        </div>
    </div>
</div>

<div id="previewModal" class="modal">
    <div class="modal-content">
        <h3 id="previewTitle" style="color: var(--bpn-blue); margin-bottom: 12px;">Live Preview</h3>
        <img id="previewImg" src="" alt="Preview">
        <button onclick="closePreview()" style="background: var(--bahau-red); color: #fff; border: none; padding: 8px 20px; margin-top: 15px; border-radius: 4px; font-weight: bold; cursor: pointer;">Tutup</button>
    </div>
</div>

<div id="editModal" class="modal">
    <div class="modal-content" style="text-align: left;">
        <h3 style="color: var(--bpn-blue); margin-bottom: 15px; text-align: center;">✏️ Edit Data CCTV</h3>
        <form action="/admin/edit_cctv" method="POST">
            <input type="hidden" id="editCamId" name="cam_id">
            <div style="margin-bottom: 10px;">
                <label style="font-size: 12px; font-weight: bold; color: var(--bpn-blue);">Nama Kamera</label>
                <input type="text" id="editName" name="name" required style="width:100%; padding:8px; border:1px solid #ccc; border-radius:4px; text-transform: uppercase;">
            </div>
            <div style="margin-bottom: 10px;">
                <label style="font-size: 12px; font-weight: bold; color: var(--bpn-blue);">Nama Ruas Jalan</label>
                <input type="text" id="editRuas" name="ruas_name" required style="width:100%; padding:8px; border:1px solid #ccc; border-radius:4px; text-transform: uppercase;">
            </div>
            <div style="margin-bottom: 10px;">
                <label style="font-size: 12px; font-weight: bold; color: var(--bpn-blue);">URL Stream (.m3u8)</label>
                <input type="text" id="editStreamUrl" name="stream_url" required style="width:100%; padding:8px; border:1px solid #ccc; border-radius:4px;">
            </div>
            <div style="display: flex; gap: 10px; margin-bottom: 15px;">
                <div style="flex: 1;">
                    <label style="font-size: 12px; font-weight: bold; color: var(--bpn-blue);">Latitude</label>
                    <input type="text" id="editLat" name="lat" required style="width:100%; padding:8px; border:1px solid #ccc; border-radius:4px;">
                </div>
                <div style="flex: 1;">
                    <label style="font-size: 12px; font-weight: bold; color: var(--bpn-blue);">Longitude</label>
                    <input type="text" id="editLng" name="lng" required style="width:100%; padding:8px; border:1px solid #ccc; border-radius:4px;">
                </div>
            </div>
            <button type="submit" style="background: #10b981; color: #fff; border: none; padding: 10px; font-weight: bold; border-radius: 4px; width: 100%; cursor: pointer;">Simpan Perubahan</button>
            <button type="button" onclick="closeEditModal()" style="background: #ccc; color: #333; border: none; padding: 8px; font-weight: bold; border-radius: 4px; width: 100%; margin-top: 6px; cursor: pointer;">Batal</button>
        </form>
    </div>
</div>

<footer>&copy; 2026 CCTV Kendaraan Non Tol Nasional</footer>
<script>
function switchTab(tabId, btnElement) {
    document.querySelectorAll('.tab-section').forEach(sec => sec.classList.remove('active'));
    document.querySelectorAll('.menu-btn').forEach(btn => btn.classList.remove('active'));
    
    document.getElementById(tabId).classList.add('active');
    btnElement.classList.add('active');
}

function autoDetectCoordinates(url) {
    const lowerUrl = url.toLowerCase();
    let lat = -1.25, lng = 116.84;
    const ruasField = document.getElementById('ruasInput');
    if (lowerUrl.includes('denpasar') || lowerUrl.includes('gunungagung') || lowerUrl.includes('bali')) {
        lat = -8.65; lng = 115.2167;
        if (ruasField) ruasField.value = "SIMPANG GUNUNG AGUNG - KOTA DENPASAR, BALI";
    }
    document.getElementById('latInput').value = lat.toFixed(4);
    document.getElementById('lngInput').value = lng.toFixed(4);
}

function previewCctv(camId, camName) {
    document.getElementById('previewTitle').textContent = `Preview: ${camName}`;
    document.getElementById('previewImg').src = `/video_feed/${camId}`;
    document.getElementById('previewModal').style.display = 'flex';
}
function closePreview() {
    document.getElementById('previewImg').src = '';
    document.getElementById('previewModal').style.display = 'none';
}

document.addEventListener('click', function(e) {
    if (e.target && e.target.classList.contains('btn-edit-trigger')) {
        const btn = e.target;
        document.getElementById('editCamId').value = btn.getAttribute('data-id');
        document.getElementById('editName').value = btn.getAttribute('data-name');
        document.getElementById('editRuas').value = btn.getAttribute('data-ruas');
        document.getElementById('editLat').value = btn.getAttribute('data-lat');
        document.getElementById('editLng').value = btn.getAttribute('data-lng');
        document.getElementById('editStreamUrl').value = btn.getAttribute('data-url');
        document.getElementById('editModal').style.display = 'flex';
    }
});

function closeEditModal() {
    document.getElementById('editModal').style.display = 'none';
}

async function fetchUserStatus() {
    try {
        const res = await fetch('/api/admin/users_status');
        const data = await res.json();
        const tbody = document.getElementById('userTableBody');
        if (!tbody) return;
        tbody.innerHTML = '';
        
        for (const [uname, info] of Object.entries(data.users)) {
            const isOnline = (data.now - info.last_active) < 8;
            const badge = isOnline 
                ? '<span class="badge-online">🟢 Online</span>' 
                : '<span class="badge-offline">⚪ Offline</span>';
            
            tbody.innerHTML += `
                <tr>
                    <td><code>${uname}</code></td>
                    <td><b>${info.name}</b></td>
                    <td><span style="text-transform: uppercase; font-weight: bold; color: var(--bpn-blue);">${info.role}</span></td>
                    <td>${badge}</td>
                </tr>
            `;
        }
    } catch (e) {}
}

setInterval(fetchUserStatus, 2000);
window.onload = fetchUserStatus;
</script>
</body>
</html>
"""

def _mjpeg_generator(worker: SpatialFilteredStreamWorker):
    worker.start_if_needed()
    while True:
        jpeg = worker.get_jpeg()
        if jpeg is None:
            frame = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Menghubungkan Stream...", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            ok, buf = cv2.imencode(".jpg", frame)
            jpeg = buf.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")

@app.route("/")
def index():
    global CAMERA_CONFIGS
    cams = [{"id": cfg.cam_id, "name": cfg.name, "ruas_name": cfg.ruas_name, "lat": cfg.lat, "lng": cfg.lng, "stream_url": cfg.stream_url} for cfg in CAMERA_CONFIGS]
    user = session.get("user")
    
    if not user:
        return render_template_string(PUBLIC_TEMPLATE, cams=cams, get_city_group=get_city_group)
    
    return render_template_string(USER_TEMPLATE, cams=cams, current_user=user, get_city_group=get_city_group)

@app.route("/admin/dashboard")
def admin_page():
    user = session.get("user")
    if not user or user.get("role") != "admin":
        return redirect("/")
    global CAMERA_CONFIGS
    cams = [{"id": cfg.cam_id, "name": cfg.name, "ruas_name": cfg.ruas_name, "lat": cfg.lat, "lng": cfg.lng, "stream_url": cfg.stream_url} for cfg in CAMERA_CONFIGS]
    return render_template_string(ADMIN_TEMPLATE, cams=cams, current_user=user)

@app.route("/api/admin/users_status")
def api_users_status():
    user = session.get("user")
    if not user or user.get("role") != "admin":
        return jsonify({"error": "Unauthorized"}), 403
    users_db = load_users_db()
    return jsonify({"users": users_db, "now": time.time()})

@app.route("/api/heartbeat")
def api_heartbeat():
    user = session.get("user")
    if user:
        update_user_activity_helper(user.get("username"))
    return jsonify({"status": "ok"})

@app.route("/admin/add_cctv", methods=["POST"])
def admin_add_cctv():
    user = session.get("user")
    if not user or user.get("role") != "admin": return redirect("/")
    cam_id = request.form.get("cam_id", "").strip().lower().replace(" ", "_")
    name = request.form.get("name", "").strip().upper()
    ruas_name = request.form.get("ruas_name", "").strip().upper()
    lat_str = request.form.get("lat", "").strip()
    lng_str = request.form.get("lng", "").strip()
    stream_url = request.form.get("stream_url", "").strip()
    if not cam_id or not name or not stream_url: return "Error: Data tidak lengkap!", 400
    try: lat, lng = float(lat_str), float(lng_str)
    except ValueError: lat, lng = -1.25, 116.84
    raw_cams = load_cameras_registry()
    if not any(c["cam_id"] == cam_id for c in raw_cams):
        raw_cams.append({"cam_id": cam_id, "name": name, "ruas_name": ruas_name or "RUAS JALAN", "lat": lat, "lng": lng, "stream_url": stream_url})
        save_cameras_registry(raw_cams)
        global CAMERA_CONFIGS
        CAMERA_CONFIGS = load_all_workers()
    return redirect("/admin/dashboard")

@app.route("/admin/edit_cctv", methods=["POST"])
def admin_edit_cctv():
    user = session.get("user")
    if not user or user.get("role") != "admin": return redirect("/")
    cam_id = request.form.get("cam_id", "").strip()
    name = request.form.get("name", "").strip().upper()       # Memastikan Nama Kamera otomatis capslock
    ruas_name = request.form.get("ruas_name", "").strip().upper() # Memastikan Ruas Jalan otomatis capslock
    lat_str = request.form.get("lat", "").strip()
    lng_str = request.form.get("lng", "").strip()
    stream_url = request.form.get("stream_url", "").strip()
    
    try: lat, lng = float(lat_str), float(lng_str)
    except ValueError: lat, lng = -1.25, 116.84

    raw_cams = load_cameras_registry()
    for c in raw_cams:
        if c["cam_id"] == cam_id:
            c["name"] = name
            c["ruas_name"] = ruas_name
            c["lat"] = lat
            c["lng"] = lng
            c["stream_url"] = stream_url
            break
            
    save_cameras_registry(raw_cams)
    global CAMERA_CONFIGS
    CAMERA_CONFIGS = load_all_workers()
    return redirect("/admin/dashboard")

@app.route("/login", methods=["POST"])
def login():
    u = request.form.get("username", "").strip()
    p = request.form.get("password", "").strip()
    db = load_users_db()
    if u in db and db[u]["password"] == p:
        role = db[u]["role"]
        db[u]["last_active"] = time.time()
        save_users_db(db)
        session["user"] = {"name": db[u]["name"], "role": role, "username": u}
        redirect_url = "/admin/dashboard" if role == "admin" else "/"
        return jsonify({"success": True, "redirect": redirect_url})
    return jsonify({"success": False, "message": "Username atau Password salah!"})

@app.route("/register", methods=["POST"])
def register():
    u = request.form.get("username", "").strip()
    p = request.form.get("password", "").strip()
    name = request.form.get("name", "").strip()
    if not u or not p or not name: return jsonify({"success": False, "message": "Wajib diisi!"})
    
    db = load_users_db()
    if u in db: return jsonify({"success": False, "message": "Username sudah terdaftar!"})
    
    db[u] = {"password": p, "name": name, "role": "user", "last_active": time.time()}
    save_users_db(db)
    
    session["user"] = {"name": name, "role": "user", "username": u}
    return jsonify({"success": True})

@app.route("/logout")
def logout():
    if "user" in session:
        username = session["user"].get("username")
        if username:
            with json_lock:
                db = load_users_db()
                if username in db:
                    db[username]["last_active"] = 0
                    save_users_db(db)
    session.clear()
    return redirect("/")

@app.route("/video_feed/<cam_id>")
def video_feed(cam_id):
    worker = WORKERS.get(cam_id)
    if worker is None: abort(404)
    return Response(_mjpeg_generator(worker), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/stats/<cam_id>")
def api_stats(cam_id):
    user = session.get("user")
    if not user: abort(403)
    worker = WORKERS.get(cam_id)
    if worker is None: abort(404)
    worker.start_if_needed()
    return jsonify(worker.get_stats())

@app.route("/api/export_csv/<cam_id>")
def api_export_csv(cam_id):
    user = session.get("user")
    if not user or user.get("role") != "admin": return "Akses Ditolak", 403
    worker = WORKERS.get(cam_id)
    if worker is None: abort(404)
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT timestamp, stream_name, zone_label, mobil, motor, truk, bus, total FROM crossing_log")
            rows = cursor.fetchall()
    csv = "Timestamp,CCTV,Zona,Mobil,Motor,Truk,Bus,Total\n" + "\n".join([f"{r['timestamp']},{r['stream_name']},{r['zone_label']},{r['mobil']},{r['motor']},{r['truk']},{r['bus']},{r['total']}" for r in rows])
    return Response(csv, mimetype="text/csv", headers={"Content-disposition": f"attachment; filename=laporan_{cam_id}.csv"})

if __name__ == "__main__":
    start_database_logger(interval_seconds=60)
    print("Starting Flask server on http://127.0.0.1:8080")
    app.run(host="0.0.0.0", port=8080, threaded=True, debug=False)