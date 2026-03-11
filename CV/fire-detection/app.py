import os
import time
import logging
from pathlib import Path
from collections import defaultdict
from typing import Optional, Tuple, Dict, Any, List

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, Query, Response
from fastapi.responses import JSONResponse, HTMLResponse
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

app = FastAPI(title="Fire Detection Service", version="1.0.0")

_MODEL_CACHE: Dict[str, YOLO] = {}


def is_url_like(s: str) -> bool:
    s = s or ""
    return s.startswith(("rtsp://", "rtsps://", "http://", "https://", "rtmp://", "rtmps://", "ws://", "wss://"))


def open_capture(source: Optional[str], fallback_index: int = 0) -> cv2.VideoCapture:
    if source and is_url_like(source):
        return cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    try:
        index = int(source) if source is not None else int(fallback_index)
    except ValueError:
        index = int(fallback_index)
    return cv2.VideoCapture(index)


def ensure_dir(p: Optional[str]) -> Optional[Path]:
    if not p:
        return None
    d = Path(p)
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_model(model_path: str) -> YOLO:
    if model_path not in _MODEL_CACHE:
        logging.info("Loading YOLO model from: %s", model_path)
        _MODEL_CACHE[model_path] = YOLO(model_path)
    return _MODEL_CACHE[model_path]


def detect_fire_on_frame(
    bgr: np.ndarray,
    model: YOLO,
    conf_threshold: float,
    imgsz: int,
) -> Tuple[List[Dict[str, Any]], List[Tuple[int, int, int, int]], List[str]]:
    start = time.time()

    results = model.predict(
        source=bgr,
        conf=conf_threshold,
        imgsz=imgsz,
        verbose=False,
    )

    detections: List[Dict[str, Any]] = []
    draw_boxes: List[Tuple[int, int, int, int]] = []
    draw_labels: List[str] = []

    if not results:
        return detections, draw_boxes, draw_labels

    r = results[0]
    names = r.names if hasattr(r, "names") else {0: "fire"}

    if r.boxes is not None:
        for box in r.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0]) if box.cls is not None else 0
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            label = str(names.get(cls_id, "fire"))

            draw_boxes.append((x1, y1, x2, y2))
            draw_labels.append(f"{label} {conf:.2f}")
            detections.append(
                {
                    "label": label,
                    "confidence": conf,
                    "box": {"left": x1, "top": y1, "right": x2, "bottom": y2},
                }
            )

    elapsed = (time.time() - start) * 1000
    logging.info("Detection finished in %.1fms; detections=%d", elapsed, len(detections))
    return detections, draw_boxes, draw_labels


def annotate(frame: np.ndarray, boxes_xyxy: List[Tuple[int, int, int, int]], labels: List[str]) -> np.ndarray:
    for (x1, y1, x2, y2), label in zip(boxes_xyxy, labels):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.rectangle(frame, (x1, max(0, y1 - 28)), (x2, y1), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, label, (x1 + 6, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/detect-image")
async def detect_image(
    file: UploadFile = File(...),
    model_path: str = Form("/models/fire_model.pt"),
    conf_threshold: float = Form(0.5),
    imgsz: int = Form(640),
    annotated_out: Optional[str] = Form(None),
):
    try:
        model = get_model(model_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Failed loading model: {e}", "model_path": model_path})

    payload = await file.read()
    np_bytes = np.frombuffer(payload, dtype=np.uint8)
    image_bgr = cv2.imdecode(np_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image upload"})

    detections, boxes, labels = detect_fire_on_frame(image_bgr, model, conf_threshold, imgsz)

    saved = None
    if annotated_out and detections:
        out_path = Path(annotated_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ann = annotate(image_bgr.copy(), boxes, labels)
        if cv2.imwrite(str(out_path), ann):
            saved = str(out_path)

    return {
        "mode": "image",
        "model_path": model_path,
        "conf_threshold": conf_threshold,
        "imgsz": imgsz,
        "detections": detections,
        "annotated_out": saved,
    }


@app.post("/detect-webcam")
async def detect_webcam(
    camera_url: Optional[str] = Form(None),
    webcam: int = Form(0),
    model_path: str = Form("/models/fire_model.pt"),
    conf_threshold: float = Form(0.5),
    imgsz: int = Form(640),
    max_seconds: int = Form(10),
    max_frames: int = Form(0),
    frame_stride: int = Form(3),
    stop_on_first: bool = Form(False),
    annotated_dir: Optional[str] = Form(None),
    save_all_frames: bool = Form(False),
    include_timeline: bool = Form(False),
):
    camera_url = camera_url or os.getenv("CAMERA_URL")

    try:
        model = get_model(model_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Failed loading model: {e}", "model_path": model_path})

    cap = open_capture(camera_url, fallback_index=webcam)
    if not cap.isOpened():
        which = camera_url if (camera_url and is_url_like(camera_url)) else f"index {webcam}"
        return JSONResponse(status_code=500, content={"error": f"Cannot open capture ({which})"})

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    annotated_path = ensure_dir(annotated_dir)
    t0 = time.time()
    deadline = t0 + max_seconds if max_seconds else 0
    stride = max(1, frame_stride)

    frame_id = 0
    processed = 0
    detections_over_time = []
    frames_with_fire = 0
    last_annotated_path = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                if max_seconds and time.time() >= deadline:
                    break
                time.sleep(0.01)
                continue

            frame_id += 1
            if frame_id % stride != 0:
                continue

            processed += 1
            detections, boxes, labels = detect_fire_on_frame(frame, model, conf_threshold, imgsz)

            has_fire = len(detections) > 0
            if has_fire:
                frames_with_fire += 1

            snap = {
                "ts": time.time(),
                "frame_id": frame_id,
                "fire_detected": has_fire,
                "detections": detections,
            }
            detections_over_time.append(snap)

            if annotated_path and (has_fire or save_all_frames):
                annotated = annotate(frame.copy(), boxes, labels) if has_fire else frame
                out_path = annotated_path / f"frame_{frame_id:06d}.jpg"
                if cv2.imwrite(str(out_path), annotated):
                    last_annotated_path = str(out_path)

            if stop_on_first and has_fire:
                break
            if max_frames and processed >= max_frames:
                break
            if max_seconds and time.time() >= deadline:
                break
    finally:
        cap.release()

    summary = {
        "mode": "stream" if (camera_url and is_url_like(camera_url)) else "webcam",
        "source": camera_url or f"index {int(webcam)}",
        "model_path": model_path,
        "conf_threshold": conf_threshold,
        "imgsz": imgsz,
        "frames_processed": processed,
        "frames_with_fire": frames_with_fire,
        "frames_without_fire": max(0, processed - frames_with_fire),
        "fire_ratio": (frames_with_fire / processed) if processed else 0.0,
        "stop_reason": (
            "stop_on_first_fire" if (stop_on_first and frames_with_fire > 0)
            else ("max_frames_reached" if (max_frames and processed >= max_frames)
                  else ("timeout_or_end_of_capture" if max_seconds else "end_of_capture"))
        ),
        "last_annotated_frame": last_annotated_path,
    }
    if include_timeline:
        summary["timeline"] = detections_over_time
    return summary


def mjpeg_generator(
    source: Optional[str],
    webcam_index: int,
    model: YOLO,
    conf_threshold: float,
    imgsz: int,
    frame_stride: int,
    annotated: bool,
):
    cap = open_capture(source, fallback_index=webcam_index)
    if not cap.isOpened():
        msg = f"Cannot open capture ({source if (source and is_url_like(source)) else f'index {webcam_index}'})"
        canvas = np.zeros((240, 640, 3), dtype=np.uint8)
        cv2.putText(canvas, msg, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        ret, buf = cv2.imencode(".jpg", canvas)
        frame = buf.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    frame_id = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                placeholder = np.zeros((240, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "No frame", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                ret, buf = cv2.imencode(".jpg", placeholder)
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
                time.sleep(0.02)
                continue

            frame_id += 1
            if frame_id % max(1, frame_stride) != 0:
                view = frame
            else:
                detections, boxes, labels = detect_fire_on_frame(frame, model, conf_threshold, imgsz)
                view = annotate(frame.copy(), boxes, labels) if (annotated and detections) else frame

            ret, buf = cv2.imencode(".jpg", view)
            if not ret:
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
    finally:
        cap.release()


@app.get("/stream")
async def stream(
    camera_url: Optional[str] = Query(None),
    webcam: int = Query(0),
    model_path: str = Query("/models/fire_model.pt"),
    conf_threshold: float = Query(0.5),
    imgsz: int = Query(640),
    frame_stride: int = Query(3),
    annotated: bool = Query(True),
):
    camera_url = camera_url or os.getenv("CAMERA_URL")
    model = get_model(model_path)
    gen = mjpeg_generator(camera_url, webcam, model, conf_threshold, imgsz, frame_stride, annotated)
    return Response(gen, media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/ui", response_class=HTMLResponse)
async def ui():
    html = """
    <!DOCTYPE html>
    <html lang=\"en\">
    <head>
      <meta charset=\"utf-8\">
      <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
      <title>Fire Detection Live View</title>
      <style>
        body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu; margin:20px; color:#222;}
        .controls{display:flex; gap:8px; flex-wrap:wrap; margin-bottom:12px;}
        label{font-size:14px;}
        input{padding:6px 8px; font-size:14px;}
        img{max-width:100%; border-radius:12px; box-shadow:0 6px 24px rgba(0,0,0,0.12);}
      </style>
    </head>
    <body>
      <h2>Fire Detection Live View</h2>
      <div class=\"controls\">
        <label>Source URL <input id=\"url\" type=\"text\" value=\"http://mediamtx:8889/cam\" size=\"32\"></label>
        <label>Webcam idx <input id=\"cam\" type=\"number\" value=\"0\" style=\"width:5em\"></label>
        <label>Confidence <input id=\"conf\" type=\"number\" step=\"0.01\" value=\"0.5\" style=\"width:6em\"></label>
        <label>Stride <input id=\"stride\" type=\"number\" value=\"3\" style=\"width:6em\"></label>
        <label><input id=\"ann\" type=\"checkbox\" checked> Annotate</label>
        <button id=\"go\">Start</button>
      </div>
      <img id=\"view\" alt=\"stream\">
      <script>
        const btn = document.getElementById('go');
        const img = document.getElementById('view');
        btn.onclick = () => {
          const q = new URLSearchParams({
            camera_url: document.getElementById('url').value,
            webcam: document.getElementById('cam').value,
            conf_threshold: document.getElementById('conf').value,
            frame_stride: document.getElementById('stride').value,
            annotated: document.getElementById('ann').checked,
          });
          img.src = '/stream?' + q.toString();
        };
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html, status_code=200)
