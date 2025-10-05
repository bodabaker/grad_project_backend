
import os
import time
from typing import List, Optional, Dict, Any
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import face_recognition
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse

app = FastAPI(title="Face Detection Service", version="1.0.0")

SUPPORTED_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

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
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model=model)
    encodings = face_recognition.face_encodings(rgb, boxes)

    results = []
    draw_labels = []
    draw_boxes = []

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
        known_encodings, known_labels = load_known_encodings(Path(persons_dir), model=model)
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
    try:
        known_encodings, known_labels = load_known_encodings(Path(persons_dir), model=model)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Failed loading persons-dir: {e}", "persons_dir": persons_dir})

    cap = cv2.VideoCapture(int(webcam))
    if not cap.isOpened():
        return JSONResponse(status_code=500, content={"error": f"Cannot open webcam index {webcam}"})

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
                if time.time() - t0 > 1.0:
                    break
                continue

            frame_id += 1
            if frame_id % stride != 0:
                continue

            processed += 1
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

            if stop_on_first and any(lbl != "Unknown" for lbl in labels_for_draw):
                break
            if max_frames and processed >= max_frames:
                break
            if max_seconds and time.time() >= deadline:
                break
    finally:
        cap.release()

    known_only_counts = {k: v for k, v in match_counts.items() if k != "Unknown"}
    summary = {
        "mode": "webcam",
        "webcam_index": int(webcam),
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


from fastapi import Response, Query
from fastapi.responses import HTMLResponse
from starlette.concurrency import run_in_threadpool

def mjpeg_generator(persons_dir: str, webcam: int, model: str, tolerance: float,
                    frame_stride: int, annotated: bool):
    # Load encodings once
    known_encodings, known_labels = load_known_encodings(Path(persons_dir), model=model)

    cap = cv2.VideoCapture(int(webcam))
    if not cap.isOpened():
        # Yield a single frame with error text
        error_canvas = np.zeros((240, 640, 3), dtype=np.uint8)
        cv2.putText(error_canvas, f"Cannot open webcam {webcam}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        ret, buf = cv2.imencode(".jpg", error_canvas)
        frame = buf.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        return

    frame_id = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                # brief placeholder
                placeholder = np.zeros((240, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "No frame", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                ret, buf = cv2.imencode(".jpg", placeholder)
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
                continue

            frame_id += 1
            if frame_id % max(1, frame_stride) != 0:
                # Still show raw frame for smoothness
                view = frame
            else:
                results, boxes_xyxy, labels_for_draw = detect_on_frame(
                    frame, known_encodings, known_labels, model, tolerance
                )
                view = annotate(frame.copy(), boxes_xyxy, labels_for_draw) if annotated else frame

            ret, buf = cv2.imencode(".jpg", view)
            if not ret:
                continue
            frame = buf.tobytes()
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
    finally:
        cap.release()

@app.get("/stream")
async def stream(
    persons_dir: str = Query(..., description="Directory with known faces inside the container, e.g. /data/persons"),
    webcam: int = Query(0),
    model: str = Query("hog"),
    tolerance: float = Query(0.6),
    frame_stride: int = Query(5),
    annotated: bool = Query(True, description="Draw boxes and labels on stream")
):
    gen = mjpeg_generator(persons_dir, webcam, model, tolerance, frame_stride, annotated)
    return Response(gen, media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/ui", response_class=HTMLResponse)
async def ui():
    # Lightweight viewer page that hits /stream with query params
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
        <label>Webcam <input id="cam" type="number" value="0" style="width:5em"></label>
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
          const cam = document.getElementById('cam').value;
          const model = document.getElementById('model').value;
          const tol = document.getElementById('tol').value;
          const stride = document.getElementById('stride').value;
          const ann = document.getElementById('ann').checked;
          const url = `/stream?persons_dir=${encodeURIComponent(persons)}&webcam=${cam}&model=${model}&tolerance=${tol}&frame_stride=${stride}&annotated=${ann}`;
          img.src = url;
        };
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html, status_code=200)
