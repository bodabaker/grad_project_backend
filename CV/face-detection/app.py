import os
import time
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
from collections import defaultdict

import cv2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
import numpy as np
import face_recognition
from fastapi import FastAPI, UploadFile, File, Form, Response, Query
from fastapi.responses import JSONResponse, HTMLResponse
from starlette.concurrency import run_in_threadpool

app = FastAPI(title="Face Detection Service", version="1.0.0")

SUPPORTED_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Cache for known face encodings with metadata
_ENCODINGS_CACHE = {}
_CACHE_METADATA = {}

def get_cached_encodings(persons_dir: str, model: str = "hog") -> tuple:
    """Get face encodings from cache or compute and cache them if not present"""
    cache_key = f"{persons_dir}:{model}"
    
    # Check if we need to refresh cache based on directory modification time
    dir_path = Path(persons_dir)
    if dir_path.exists():
        latest_mtime = max(p.stat().st_mtime for p in dir_path.glob('*') if p.is_file())
        cache_mtime = _CACHE_METADATA.get(cache_key, {}).get('mtime', 0)
        
        # Refresh cache if directory was modified
        if latest_mtime > cache_mtime:
            _ENCODINGS_CACHE.pop(cache_key, None)
    
    if cache_key not in _ENCODINGS_CACHE:
        logging.info(f"Computing encodings for {persons_dir} with model {model}")
        start_time = time.time()
        encodings, labels = load_known_encodings(Path(persons_dir), model=model)
        compute_time = time.time() - start_time
        logging.info(f"Cached {len(encodings)} encodings in {compute_time:.1f}s")
        
        # Store encodings and metadata
        _ENCODINGS_CACHE[cache_key] = (encodings, labels)
        _CACHE_METADATA[cache_key] = {
            'mtime': time.time(),
            'count': len(encodings),
            'compute_time': compute_time
        }
    return _ENCODINGS_CACHE[cache_key]

# -----------------------------
# Helpers
# -----------------------------
def is_url_like(s: str) -> bool:
    s = s or ""
    return s.startswith(("rtsp://", "rtsps://", "http://", "https://", "rtmp://", "rtmps://", "ws://", "wss://"))

def open_capture(source: Optional[str], fallback_index: int = 0) -> cv2.VideoCapture:
    """
    Open a camera/stream. If `source` looks like a URL, use FFMPEG backend.
    Else treat it as an integer webcam index (or use `fallback_index`).
    """
    if source and is_url_like(source):
        # For network streams (RTSP/HLS/HTTP), FFMPEG is the most robust backend
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        return cap
    # If source is given but not URL, try parsing as int; else fallback to index
    try:
        index = int(source) if source is not None else int(fallback_index)
    except ValueError:
        index = int(fallback_index)
    cap = cv2.VideoCapture(index)
    return cap

def load_known_encodings(persons_dir: Path, model: str = "hog"):
    encodings = []
    labels = []
    if not persons_dir.exists():
        raise FileNotFoundError(f"Persons directory not found: {persons_dir}")
    for img_path in sorted(persons_dir.iterdir()):
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in SUPPORTED_IMG_EXTS:
            continue
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model=model)
        if not boxes:
            continue
        enc = face_recognition.face_encodings(rgb, boxes)[0]
        encodings.append(enc)
        labels.append(img_path.stem)
    return encodings, labels

def annotate(image_bgr, boxes_xyxy, labels):
    for (x1, y1, x2, y2), label in zip(boxes_xyxy, labels):
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.rectangle(image_bgr, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
        cv2.putText(image_bgr, label, (x1 + 6, y2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return image_bgr

def detect_on_frame(bgr, known_encodings, known_labels, model, tolerance):
    start_time = time.time()
    
    # Calculate optimal scale factor to reduce processing time
    target_width = 640  # Smaller size for faster processing
    height, width = bgr.shape[:2]
    scale = target_width / width
    
    if scale < 1.0:
        # Resize image for faster processing
        small_frame = cv2.resize(bgr, (0, 0), fx=scale, fy=scale)
    else:
        small_frame = bgr
    
    logging.info(f"Processing frame of size: {width}x{height}, scaled to {small_frame.shape[1]}x{small_frame.shape[0]}")
    
    # Convert only once and store in memory
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    t1 = time.time()
    logging.info(f"BGR to RGB conversion: {(t1-start_time)*1000:.1f}ms")
    
    # Use smaller stride for faster face detection
    logging.info(f"Starting face detection with model: {model}")
    boxes = face_recognition.face_locations(rgb_small, model=model, number_of_times_to_upsample=1)
    t2 = time.time()
    logging.info(f"Face detection: {(t2-t1)*1000:.1f}ms, Found {len(boxes)} faces")
    
    # Batch process encodings for better performance
    encodings = face_recognition.face_encodings(rgb_small, boxes, num_jitters=1)
    t3 = time.time()
    logging.info(f"Face encoding: {(t3-t2)*1000:.1f}ms for {len(encodings)} faces")
    
    total_time = t3 - start_time
    logging.info(f"Total processing time: {total_time*1000:.1f}ms ({1/total_time:.1f} FPS)")

    results = []
    draw_labels = []
    draw_boxes = []
    
    # Scale back the boxes to original image size
    if scale < 1.0:
        scaled_boxes = []
        for (top, right, bottom, left) in boxes:
            scaled_boxes.append((
                int(top / scale),
                int(right / scale),
                int(bottom / scale),
                int(left / scale)
            ))
        boxes = scaled_boxes

    for enc, (top, right, bottom, left) in zip(encodings, boxes):
        if known_encodings:
            distances = face_recognition.face_distance(known_encodings, enc).tolist()
            best_idx = int(np.argmin(distances)) if distances else None
            if best_idx is not None and distances[best_idx] <= tolerance:
                name = known_labels[best_idx]
                best_distance = float(distances[best_idx])
            else:
                name = "Unknown"
                best_distance = float(distances[best_idx]) if best_idx is not None else None
        else:
            name = "Unknown"
            best_distance = None

        x1, y1, x2, y2 = left, top, right, bottom
        draw_boxes.append((x1, y1, x2, y2))
        draw_labels.append(name)
        results.append({
            "name": name,
            "distance": best_distance,
            "box": {"left": x1, "top": y1, "right": x2, "bottom": y2},
        })
    return results, draw_boxes, draw_labels

def ensure_dir(p: Optional[str]) -> Optional[Path]:
    if not p:
        return None
    d = Path(p)
    d.mkdir(parents=True, exist_ok=True)
    return d

# -----------------------------
# Endpoints
# -----------------------------
@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/detect-image")
async def detect_image(
    persons_dir: str = Form(...),
    model: str = Form("hog"),
    tolerance: float = Form(0.6),
    annotated_out: Optional[str] = Form(None),
    file: UploadFile = File(...),
):
    try:
        known_encodings, known_labels = get_cached_encodings(persons_dir, model=model)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Failed loading persons-dir: {e}", "persons_dir": persons_dir})

    data = await file.read()
    file_bytes = np.frombuffer(data, dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image upload"})

    results, boxes_xyxy, labels_for_draw = detect_on_frame(image_bgr, known_encodings, known_labels, model, tolerance)

    saved = None
    if annotated_out and results:
        out_path = Path(annotated_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ann = annotate(image_bgr.copy(), boxes_xyxy, labels_for_draw)
        if cv2.imwrite(str(out_path), ann):
            saved = str(out_path)

    return {
        "mode": "image",
        "persons_dir": persons_dir,
        "model": model,
        "tolerance": tolerance,
        "detections": results,
        "annotated_out": saved,
    }

@app.post("/detect-webcam")
async def detect_webcam(
    persons_dir: str = Form(...),
    # NEW: allow passing a stream URL; falls back to webcam index if not provided
    camera_url: Optional[str] = Form(None, description="e.g. http://mediamtx:8889/cam/webrtc"),
    webcam: int = Form(0),
    model: str = Form("hog"),
    tolerance: float = Form(0.6),
    max_seconds: int = Form(10),
    max_frames: int = Form(0),
    frame_stride: int = Form(5),
    stop_on_first: bool = Form(False),
    annotated_dir: Optional[str] = Form(None),
    save_all_frames: bool = Form(False),
    include_timeline: bool = Form(False),
):
    # If not provided, try environment variable
    camera_url = camera_url or os.getenv("CAMERA_URL")

    try:
        known_encodings, known_labels = get_cached_encodings(persons_dir, model=model)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Failed loading persons-dir: {e}", "persons_dir": persons_dir})

    cap = open_capture(camera_url, fallback_index=webcam)
    if not cap.isOpened():
        which = camera_url if (camera_url and is_url_like(camera_url)) else f"index {webcam}"
        return JSONResponse(status_code=500, content={"error": f"Cannot open capture ({which})"})

    # RTSP/Network stream optimizations for lower latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimum buffer size

    annotated_path = ensure_dir(annotated_dir)
    t0 = time.time()
    deadline = t0 + max_seconds if max_seconds else 0
    stride = max(1, frame_stride)

    frame_id = 0
    processed = 0
    detections_over_time = []
    match_counts = defaultdict(int)
    last_annotated_path = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                # small wait to avoid busy loop on transient failure
                if max_seconds and time.time() >= deadline:
                    break
                time.sleep(0.01)
                continue

            frame_id += 1
            if frame_id % stride != 0:
                continue

            processed += 1
            logging.info(f"Processing frame {frame_id} (processed {processed} frames total)")
            results, boxes_xyxy, labels_for_draw = detect_on_frame(frame, known_encodings, known_labels, model, tolerance)

            detections_over_time.append({
                "ts": time.time(),
                "frame_id": frame_id,
                "detections": results,
            })

            for lbl in labels_for_draw:
                match_counts[lbl] += 1

            if annotated_path and (results or save_all_frames):
                annotated = annotate(frame.copy(), boxes_xyxy, labels_for_draw) if results else frame
                out_path = annotated_path / f"frame_{frame_id:06d}.jpg"
                if cv2.imwrite(str(out_path), annotated):
                    last_annotated_path = str(out_path)

            # If stop_on_first is true and we found any known face, stop immediately
            if stop_on_first and any(lbl != "Unknown" for lbl in labels_for_draw):
                logging.info("Found known face and stop_on_first=True, stopping capture")
                break
            if max_frames and processed >= max_frames:
                break
            if max_seconds and time.time() >= deadline:
                break
    finally:
        cap.release()

    known_only_counts = {k: v for k, v in match_counts.items() if k != "Unknown"}
    summary = {
        "mode": "stream" if (camera_url and is_url_like(camera_url)) else "webcam",
        "source": camera_url or f"index {int(webcam)}",
        "persons_dir": persons_dir,
        "model": model,
        "tolerance": tolerance,
        "frames_processed": processed,
        "names_seen": known_only_counts,
        "unknown_frames": match_counts.get("Unknown", 0),
        "stop_reason": (
            "stop_on_first_match" if stop_on_first and any(
                any(det.get("name") != "Unknown" for det in snap.get("detections", []))
                for snap in detections_over_time
            ) else ("max_frames_reached" if (max_frames and processed >= max_frames)
                    else ("timeout_or_end_of_capture" if max_seconds else "end_of_capture"))
        ),
        "last_annotated_frame": last_annotated_path,
    }
    if include_timeline:
        summary["timeline"] = detections_over_time
    return summary

# -----------------------------
# MJPEG streaming endpoint
# -----------------------------
def mjpeg_generator(persons_dir: str,
                    source: Optional[str],
                    webcam_index: int,
                    model: str,
                    tolerance: float,
                    frame_stride: int,
                    annotated: bool):
    # Load encodings from cache
    known_encodings, known_labels = get_cached_encodings(persons_dir, model=model)

    cap = open_capture(source, fallback_index=webcam_index)
    if not cap.isOpened():
        # Yield a single frame with error text
        msg = f"Cannot open capture ({source if (source and is_url_like(source)) else f'index {webcam_index}'})"
        error_canvas = np.zeros((240, 640, 3), dtype=np.uint8)
        cv2.putText(error_canvas, msg, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        ret, buf = cv2.imencode(".jpg", error_canvas)
        frame = buf.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        return

    # Reduce buffer for lower latency (best effort)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frame_id = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                placeholder = np.zeros((240, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "No frame", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                ret, buf = cv2.imencode(".jpg", placeholder)
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
                time.sleep(0.02)
                continue

            frame_id += 1
            if frame_id % max(1, frame_stride) != 0:
                view = frame
            else:
                results, boxes_xyxy, labels_for_draw = detect_on_frame(
                    frame, known_encodings, known_labels, model, tolerance
                )
                view = annotate(frame.copy(), boxes_xyxy, labels_for_draw) if annotated else frame

            ret, buf = cv2.imencode(".jpg", view)
            if not ret:
                continue
            out = buf.tobytes()
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + out + b"\r\n")
    finally:
        cap.release()

@app.get("/stream")
async def stream(
    persons_dir: str = Query(..., description="Directory with known faces inside the container, e.g. /data/persons"),
    camera_url: Optional[str] = Form(None, description="e.g. http://mediamtx:8889/cam for WebRTC (preferred) or rtsp://mediamtx:8554/cam"),
    webcam: int = Query(0),
    model: str = Query("hog"),
    tolerance: float = Query(0.6),
    frame_stride: int = Query(5),
    annotated: bool = Query(True, description="Draw boxes and labels on stream")
):
    # If not provided, try environment
    camera_url = camera_url or os.getenv("CAMERA_URL")
    gen = mjpeg_generator(persons_dir, camera_url, webcam, model, tolerance, frame_stride, annotated)
    return Response(gen, media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/ui", response_class=HTMLResponse)
async def ui():
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width,initial-scale=1">
      <title>Face Service Live View</title>
      <style>
        body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu; margin:20px; color:#222;}
        .controls{display:flex; gap:8px; flex-wrap:wrap; margin-bottom:12px;}
        label{font-size:14px;}
        input,select{padding:6px 8px; font-size:14px;}
        img{max-width:100%; border-radius:12px; box-shadow:0 6px 24px rgba(0,0,0,0.12);}
      </style>
    </head>
    <body>
      <h2>Face Service Live View</h2>
      <div class="controls">
        <label>Persons dir <input id="persons" type="text" value="/data/persons" size="24"></label>
        <label>Source URL <input id="url" type="text" value="http://mediamtx:8889/cam" size="30"></label>
        <label>Webcam idx <input id="cam" type="number" value="0" style="width:5em"></label>
        <label>Model
          <select id="model">
            <option value="hog" selected>hog (CPU)</option>
            <option value="cnn">cnn (GPU if available)</option>
          </select>
        </label>
        <label>Tolerance <input id="tol" type="number" step="0.01" value="0.6" style="width:6em"></label>
        <label>Stride <input id="stride" type="number" value="5" style="width:6em"></label>
        <label><input id="ann" type="checkbox" checked> Annotate</label>
        <button id="go">Start</button>
      </div>
      <div>
        <img id="view" alt="stream will appear here">
      </div>
      <script>
        const btn = document.getElementById('go');
        const img = document.getElementById('view');
        btn.onclick = () => {
          const persons = document.getElementById('persons').value;
          const url = document.getElementById('url').value;
          const cam = document.getElementById('cam').value;
          const model = document.getElementById('model').value;
          const tol = document.getElementById('tol').value;
          const stride = document.getElementById('stride').value;
          const ann = document.getElementById('ann').checked;
          const q = new URLSearchParams({
            persons_dir: persons,
            camera_url: url,
            webcam: cam,
            model, tolerance: tol, frame_stride: stride, annotated: ann
          });
          img.src = '/stream?' + q.toString();
        };
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html, status_code=200)

