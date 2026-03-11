import os
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, Query, Response
from fastapi.responses import JSONResponse, HTMLResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

app = FastAPI(title="Object Detection Service", version="1.0.0")

_MODEL_CACHE: Dict[str, cv2.dnn_DetectionModel] = {}
_CLASSNAMES_CACHE: Dict[str, List[str]] = {}


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


def load_class_names(class_file: str) -> List[str]:
    if class_file not in _CLASSNAMES_CACHE:
        with open(class_file, "rt", encoding="utf-8") as f:
            _CLASSNAMES_CACHE[class_file] = f.read().rstrip("\n").split("\n")
    return _CLASSNAMES_CACHE[class_file]


def get_model(weights_path: str, config_path: str, input_width: int = 320, input_height: int = 320) -> cv2.dnn_DetectionModel:
    key = f"{weights_path}|{config_path}|{input_width}|{input_height}"
    if key not in _MODEL_CACHE:
        logging.info("Loading DNN model weights=%s config=%s", weights_path, config_path)
        net = cv2.dnn_DetectionModel(weights_path, config_path)
        net.setInputSize(input_width, input_height)
        net.setInputScale(1.0 / 127.5)
        net.setInputMean((127.5, 127.5, 127.5))
        net.setInputSwapRB(True)
        _MODEL_CACHE[key] = net
    return _MODEL_CACHE[key]


def detect_objects_on_frame(
    bgr: np.ndarray,
    model: cv2.dnn_DetectionModel,
    class_names: List[str],
    conf_threshold: float,
    nms_threshold: float,
) -> Tuple[List[Dict[str, Any]], List[Tuple[int, int, int, int]], List[str]]:
    t0 = time.time()

    class_ids, confs, boxes = model.detect(bgr, confThreshold=conf_threshold, nmsThreshold=nms_threshold)

    detections: List[Dict[str, Any]] = []
    draw_boxes: List[Tuple[int, int, int, int]] = []
    draw_labels: List[str] = []

    if class_ids is None or len(class_ids) == 0:
        return detections, draw_boxes, draw_labels

    class_ids = class_ids.flatten()
    confs = confs.flatten()

    for class_id, confidence, box in zip(class_ids, confs, boxes):
        x, y, w, h = [int(v) for v in box]
        x2 = x + w
        y2 = y + h

        idx = int(class_id) - 1
        label = class_names[idx] if (0 <= idx < len(class_names)) else f"class_{class_id}"
        conf = float(confidence)

        draw_boxes.append((x, y, x2, y2))
        draw_labels.append(f"{label} {conf:.2f}")
        detections.append(
            {
                "label": label,
                "class_id": int(class_id),
                "confidence": conf,
                "box": {"left": x, "top": y, "right": x2, "bottom": y2},
            }
        )

    logging.info("Object detection in %.1fms; detections=%d", (time.time() - t0) * 1000, len(detections))
    return detections, draw_boxes, draw_labels


def annotate(frame: np.ndarray, boxes_xyxy: List[Tuple[int, int, int, int]], labels: List[str]) -> np.ndarray:
    for (x1, y1, x2, y2), label in zip(boxes_xyxy, labels):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(frame, (x1, max(0, y1 - 28)), (x2, y1), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, label, (x1 + 6, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    return frame


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/detect-image")
async def detect_image(
    file: UploadFile = File(...),
    weights_path: str = Form("/models/frozen_inference_graph.pb"),
    config_path: str = Form("/models/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"),
    class_file: str = Form("/models/coco.names"),
    conf_threshold: float = Form(0.5),
    nms_threshold: float = Form(0.2),
    input_width: int = Form(320),
    input_height: int = Form(320),
    annotated_out: Optional[str] = Form(None),
):
    try:
        class_names = load_class_names(class_file)
        model = get_model(weights_path, config_path, input_width, input_height)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Failed loading model resources: {e}"})

    payload = await file.read()
    np_bytes = np.frombuffer(payload, dtype=np.uint8)
    image_bgr = cv2.imdecode(np_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image upload"})

    detections, boxes, labels = detect_objects_on_frame(image_bgr, model, class_names, conf_threshold, nms_threshold)

    saved = None
    if annotated_out and detections:
        out_path = Path(annotated_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ann = annotate(image_bgr.copy(), boxes, labels)
        if cv2.imwrite(str(out_path), ann):
            saved = str(out_path)

    return {
        "mode": "image",
        "detections": detections,
        "conf_threshold": conf_threshold,
        "nms_threshold": nms_threshold,
        "annotated_out": saved,
    }


@app.post("/detect-webcam")
async def detect_webcam(
    camera_url: Optional[str] = Form(None),
    webcam: int = Form(0),
    weights_path: str = Form("/models/frozen_inference_graph.pb"),
    config_path: str = Form("/models/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"),
    class_file: str = Form("/models/coco.names"),
    conf_threshold: float = Form(0.5),
    nms_threshold: float = Form(0.2),
    input_width: int = Form(320),
    input_height: int = Form(320),
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
        class_names = load_class_names(class_file)
        model = get_model(weights_path, config_path, input_width, input_height)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Failed loading model resources: {e}"})

    cap = open_capture(camera_url, fallback_index=webcam)
    if not cap.isOpened():
        which = camera_url if (camera_url and is_url_like(camera_url)) else f"index {webcam}"
        return JSONResponse(status_code=500, content={"error": f"Cannot open capture ({which})"})

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    annotated_path = ensure_dir(annotated_dir)
    deadline = time.time() + max_seconds if max_seconds else 0
    stride = max(1, frame_stride)

    frame_id = 0
    processed = 0
    total_detections = 0
    by_label: Dict[str, int] = {}
    timeline = []
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
            detections, boxes, labels = detect_objects_on_frame(frame, model, class_names, conf_threshold, nms_threshold)
            total_detections += len(detections)
            for d in detections:
                by_label[d["label"]] = by_label.get(d["label"], 0) + 1

            timeline.append({
                "ts": time.time(),
                "frame_id": frame_id,
                "detections": detections,
            })

            if annotated_path and (len(detections) > 0 or save_all_frames):
                annotated = annotate(frame.copy(), boxes, labels) if detections else frame
                out_path = annotated_path / f"frame_{frame_id:06d}.jpg"
                if cv2.imwrite(str(out_path), annotated):
                    last_annotated_path = str(out_path)

            if stop_on_first and len(detections) > 0:
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
        "frames_processed": processed,
        "total_detections": total_detections,
        "detections_by_label": by_label,
        "stop_reason": (
            "stop_on_first_detection" if (stop_on_first and total_detections > 0)
            else ("max_frames_reached" if (max_frames and processed >= max_frames)
                  else ("timeout_or_end_of_capture" if max_seconds else "end_of_capture"))
        ),
        "last_annotated_frame": last_annotated_path,
    }
    if include_timeline:
        summary["timeline"] = timeline
    return summary


def mjpeg_generator(
    source: Optional[str],
    webcam_index: int,
    model: cv2.dnn_DetectionModel,
    class_names: List[str],
    conf_threshold: float,
    nms_threshold: float,
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
                detections, boxes, labels = detect_objects_on_frame(frame, model, class_names, conf_threshold, nms_threshold)
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
    weights_path: str = Query("/models/frozen_inference_graph.pb"),
    config_path: str = Query("/models/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"),
    class_file: str = Query("/models/coco.names"),
    conf_threshold: float = Query(0.5),
    nms_threshold: float = Query(0.2),
    input_width: int = Query(320),
    input_height: int = Query(320),
    frame_stride: int = Query(3),
    annotated: bool = Query(True),
):
    camera_url = camera_url or os.getenv("CAMERA_URL")
    class_names = load_class_names(class_file)
    model = get_model(weights_path, config_path, input_width, input_height)
    gen = mjpeg_generator(camera_url, webcam, model, class_names, conf_threshold, nms_threshold, frame_stride, annotated)
    return Response(gen, media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/ui", response_class=HTMLResponse)
async def ui():
    html = """
    <!DOCTYPE html>
    <html lang=\"en\">
    <head>
      <meta charset=\"utf-8\">
      <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
      <title>Object Detection Live View</title>
      <style>
        body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu; margin:20px; color:#222;}
        .controls{display:flex; gap:8px; flex-wrap:wrap; margin-bottom:12px;}
        label{font-size:14px;}
        input{padding:6px 8px; font-size:14px;}
        img{max-width:100%; border-radius:12px; box-shadow:0 6px 24px rgba(0,0,0,0.12);}
      </style>
    </head>
    <body>
      <h2>Object Detection Live View</h2>
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
