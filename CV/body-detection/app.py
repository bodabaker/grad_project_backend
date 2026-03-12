import os
import time
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

app = FastAPI(title="Body Detection Service", version="1.0.0")

_MODEL_CACHE: Dict[str, YOLO] = {}

# COCO keypoint connections for skeleton rendering
POSE_CONNECTIONS = [
    (0, 5), (0, 6), (5, 6),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (12, 14), (13, 15), (14, 16),
    (5, 7), (7, 9), (6, 8), (8, 10),
]


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
        logging.info("Loading YOLO pose model from: %s", model_path)
        _MODEL_CACHE[model_path] = YOLO(model_path)
    return _MODEL_CACHE[model_path]


def valid_point(x: float, y: float, w: int, h: int) -> bool:
    return 0 < x < w and 0 < y < h


def torso_angle(kp: np.ndarray) -> float:
    shoulder = (kp[5] + kp[6]) / 2.0
    hip = (kp[11] + kp[12]) / 2.0
    dx = shoulder[0] - hip[0]
    dy = shoulder[1] - hip[1]
    return abs(float(np.degrees(np.arctan2(dx, dy))))


def classify_posture_and_state(
    kp: np.ndarray,
    frame_h: int,
    frame_w: int,
    prev_height: Optional[float],
    no_motion_start: Optional[float],
    now_ts: float,
    fall_speed_threshold: float,
    unconscious_seconds: float,
) -> Tuple[Optional[Dict[str, Any]], Optional[float], Optional[float]]:
    if kp.shape[0] < 17:
        return None, prev_height, no_motion_start

    ankles_y = []
    for idx in (15, 16):
        x, y = kp[idx]
        if valid_point(x, y, frame_w, frame_h):
            ankles_y.append(y)

    if len(ankles_y) == 0:
        knees_y = []
        for idx in (13, 14):
            x, y = kp[idx]
            if valid_point(x, y, frame_w, frame_h):
                knees_y.append(y)
        if len(knees_y) == 0:
            return None, prev_height, no_motion_start
        ankle_y = max(knees_y)
    else:
        ankle_y = max(ankles_y)

    head_x, head_y = kp[0]
    if not valid_point(head_x, head_y, frame_w, frame_h):
        return None, prev_height, no_motion_start

    body_height = float(ankle_y - head_y)
    angle = torso_angle(kp)
    normalized_height = body_height / max(frame_h, 1)

    if normalized_height > 0.6 and angle > 60:
        posture = "STANDING"
    elif normalized_height > 0.3 and angle > 25:
        posture = "SITTING"
    else:
        posture = "LYING"

    state = posture
    if prev_height is not None:
        rel_speed = abs(body_height - prev_height) / (abs(prev_height) + 1e-6)
        if rel_speed > fall_speed_threshold:
            state = "FALL"

    if posture == "LYING":
        if no_motion_start is None:
            no_motion_start = now_ts
        elif now_ts - no_motion_start > unconscious_seconds:
            state = "UNCONSCIOUS"
    else:
        no_motion_start = None

    person = {
        "posture": posture,
        "state": state,
        "normalized_height": normalized_height,
        "torso_angle": angle,
        "body_height": body_height,
    }
    return person, body_height, no_motion_start


def detect_bodies_on_frame(
    frame: np.ndarray,
    model: YOLO,
    conf_threshold: float,
    imgsz: int,
    prev_heights: Optional[Dict[int, float]] = None,
    no_motion_starts: Optional[Dict[int, float]] = None,
    fall_speed_threshold: float = 0.3,
    unconscious_seconds: float = 15.0,
) -> Tuple[List[Dict[str, Any]], List[np.ndarray], List[str], Dict[int, float], Dict[int, float]]:
    start = time.time()

    if prev_heights is None:
        prev_heights = {}
    if no_motion_starts is None:
        no_motion_starts = {}

    results = model.predict(source=frame, conf=conf_threshold, imgsz=imgsz, verbose=False)

    detections: List[Dict[str, Any]] = []
    draw_points: List[np.ndarray] = []
    draw_labels: List[str] = []

    h, w = frame.shape[:2]
    now_ts = time.time()

    if results:
        r = results[0]
        if r.keypoints is not None and r.keypoints.xy is not None:
            for person_id, keypoints in enumerate(r.keypoints.xy):
                kp = keypoints.cpu().numpy()
                person_prev_h = prev_heights.get(person_id)
                person_no_motion = no_motion_starts.get(person_id)

                person, new_h, new_no_motion = classify_posture_and_state(
                    kp,
                    frame_h=h,
                    frame_w=w,
                    prev_height=person_prev_h,
                    no_motion_start=person_no_motion,
                    now_ts=now_ts,
                    fall_speed_threshold=fall_speed_threshold,
                    unconscious_seconds=unconscious_seconds,
                )
                if person is None:
                    continue

                prev_heights[person_id] = float(new_h)
                no_motion_starts[person_id] = new_no_motion

                center_x = int(np.mean(kp[:, 0]))
                center_y = int(np.mean(kp[:, 1]))
                label = f"P{person_id} {person['state']}"

                detections.append(
                    {
                        "person_id": person_id,
                        "state": person["state"],
                        "posture": person["posture"],
                        "normalized_height": person["normalized_height"],
                        "torso_angle": person["torso_angle"],
                        "center": {"x": center_x, "y": center_y},
                        "keypoints": kp.astype(float).tolist(),
                    }
                )
                draw_points.append(kp)
                draw_labels.append(label)

    elapsed = (time.time() - start) * 1000
    logging.info("Body detection finished in %.1fms; persons=%d", elapsed, len(detections))
    return detections, draw_points, draw_labels, prev_heights, no_motion_starts


def annotate(frame: np.ndarray, people_kps: List[np.ndarray], labels: List[str], line_thickness: int = 3) -> np.ndarray:
    h, w = frame.shape[:2]
    for kp, label in zip(people_kps, labels):
        is_alert = any(s in label for s in ("FALL", "UNCONSCIOUS", "LYING"))
        color = (0, 0, 255) if is_alert else (0, 255, 0)

        for x, y in kp:
            if valid_point(float(x), float(y), w, h):
                cv2.circle(frame, (int(x), int(y)), 4, color, -1)

        for p1, p2 in POSE_CONNECTIONS:
            if p1 < kp.shape[0] and p2 < kp.shape[0]:
                x1, y1 = kp[p1]
                x2, y2 = kp[p2]
                if valid_point(float(x1), float(y1), w, h) and valid_point(float(x2), float(y2), w, h):
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, line_thickness)

        cx = int(np.mean(kp[:, 0]))
        cy = int(np.mean(kp[:, 1]))
        cv2.putText(frame, label, (cx - 60, cy - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    return frame


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/detect-image")
async def detect_image(
    file: UploadFile = File(...),
    model_path: str = Form("/models/yolov8s-pose.pt"),
    conf_threshold: float = Form(0.5),
    imgsz: int = Form(640),
    fall_speed_threshold: float = Form(0.3),
    unconscious_seconds: float = Form(15.0),
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

    detections, points, labels, _, _ = detect_bodies_on_frame(
        image_bgr,
        model,
        conf_threshold,
        imgsz,
        fall_speed_threshold=fall_speed_threshold,
        unconscious_seconds=unconscious_seconds,
    )

    saved = None
    if annotated_out and detections:
        out_path = Path(annotated_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ann = annotate(image_bgr.copy(), points, labels)
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
    model_path: str = Form("/models/yolov8s-pose.pt"),
    conf_threshold: float = Form(0.5),
    imgsz: int = Form(640),
    max_seconds: int = Form(10),
    max_frames: int = Form(0),
    frame_stride: int = Form(3),
    stop_on_alert: bool = Form(False),
    fall_speed_threshold: float = Form(0.3),
    unconscious_seconds: float = Form(15.0),
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
    timeline: List[Dict[str, Any]] = []
    frames_with_people = 0
    frames_with_alerts = 0
    last_annotated_path = None

    prev_heights: Dict[int, float] = {}
    no_motion_starts: Dict[int, float] = {}

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
            detections, points, labels, prev_heights, no_motion_starts = detect_bodies_on_frame(
                frame,
                model,
                conf_threshold,
                imgsz,
                prev_heights=prev_heights,
                no_motion_starts=no_motion_starts,
                fall_speed_threshold=fall_speed_threshold,
                unconscious_seconds=unconscious_seconds,
            )

            has_people = len(detections) > 0
            has_alert = any(d["state"] in ("FALL", "UNCONSCIOUS") for d in detections)

            if has_people:
                frames_with_people += 1
            if has_alert:
                frames_with_alerts += 1

            snap = {
                "ts": time.time(),
                "frame_id": frame_id,
                "people_detected": has_people,
                "alert_detected": has_alert,
                "detections": detections,
            }
            timeline.append(snap)

            if annotated_path and (has_people or save_all_frames):
                annotated = annotate(frame.copy(), points, labels) if has_people else frame
                out_path = annotated_path / f"frame_{frame_id:06d}.jpg"
                if cv2.imwrite(str(out_path), annotated):
                    last_annotated_path = str(out_path)

            if stop_on_alert and has_alert:
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
        "frames_with_people": frames_with_people,
        "frames_with_alerts": frames_with_alerts,
        "alert_ratio": (frames_with_alerts / processed) if processed else 0.0,
        "stop_reason": (
            "stop_on_alert" if (stop_on_alert and frames_with_alerts > 0)
            else ("max_frames_reached" if (max_frames and processed >= max_frames)
                  else ("timeout_or_end_of_capture" if max_seconds else "end_of_capture"))
        ),
        "last_annotated_frame": last_annotated_path,
    }
    if include_timeline:
        summary["timeline"] = timeline
    return summary


def mjpeg_generator(source: Optional[str], webcam_index: int, model: YOLO, conf_threshold: float, imgsz: int):
    cap = open_capture(source, fallback_index=webcam_index)
    if not cap.isOpened():
        raise RuntimeError("Cannot open source for stream")

    prev_heights: Dict[int, float] = {}
    no_motion_starts: Dict[int, float] = {}

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            detections, points, labels, prev_heights, no_motion_starts = detect_bodies_on_frame(
                frame,
                model,
                conf_threshold,
                imgsz,
                prev_heights=prev_heights,
                no_motion_starts=no_motion_starts,
            )

            annotated = annotate(frame, points, labels) if detections else frame

            ok_jpg, encoded = cv2.imencode(".jpg", annotated)
            if not ok_jpg:
                continue
            jpeg = encoded.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Cache-Control: no-cache\r\n"
                b"Pragma: no-cache\r\n"
                b"\r\n" + jpeg + b"\r\n"
            )
    finally:
        cap.release()


@app.get("/stream")
def stream(
    camera_url: Optional[str] = Query(None),
    webcam: int = Query(0),
    model_path: str = Query("/models/yolov8s-pose.pt"),
    conf_threshold: float = Query(0.5),
    imgsz: int = Query(640),
):
    camera_url = camera_url or os.getenv("CAMERA_URL")

    try:
        model = get_model(model_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Failed loading model: {e}", "model_path": model_path})

    def iterator():
        try:
            yield from mjpeg_generator(camera_url, webcam, model, conf_threshold, imgsz)
        except RuntimeError as e:
            msg = f"Stream error: {e}".encode("utf-8")
            yield (
                b"--frame\r\n"
                b"Content-Type: text/plain\r\n\r\n" + msg + b"\r\n"
            )

    return StreamingResponse(iterator(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/ui")
def ui():
    return HTMLResponse(
        """
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>Body Detection Service</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; }
    .grid { display: grid; grid-template-columns: 360px 1fr; gap: 20px; }
    label { display: block; margin-top: 10px; }
    input, select, button { width: 100%; padding: 8px; margin-top: 4px; }
    pre { background: #111; color: #0f0; padding: 10px; min-height: 180px; overflow: auto; }
    img { max-width: 100%; border: 1px solid #ccc; }
  </style>
</head>
<body>
  <h2>Body Detection Service</h2>
  <div class=\"grid\">
    <div>
      <label>Model Path <input id=\"model\" value=\"/models/yolov8s-pose.pt\" /></label>
      <label>Confidence <input id=\"conf\" type=\"number\" step=\"0.05\" min=\"0\" max=\"1\" value=\"0.5\" /></label>
      <label>Image Size <input id=\"imgsz\" type=\"number\" value=\"640\" /></label>
      <label>Max Seconds <input id=\"secs\" type=\"number\" value=\"10\" /></label>
      <label>Max Frames (0 = unlimited) <input id=\"frames\" type=\"number\" value=\"0\" /></label>
      <label>Frame Stride <input id=\"stride\" type=\"number\" value=\"3\" /></label>
      <label><input id=\"stop\" type=\"checkbox\" /> Stop on alert</label>
      <button onclick=\"runDetect()\">Run /detect-webcam</button>
      <button onclick=\"startStream()\">Start Stream</button>
      <button onclick=\"stopStream()\">Stop Stream</button>
    </div>
    <div>
      <img id=\"mjpeg\" alt=\"stream\" />
      <h3>Result</h3>
      <pre id=\"out\"></pre>
    </div>
  </div>

<script>
async function runDetect() {
  const fd = new FormData();
  fd.append('model_path', document.getElementById('model').value);
  fd.append('conf_threshold', document.getElementById('conf').value);
  fd.append('imgsz', document.getElementById('imgsz').value);
  fd.append('max_seconds', document.getElementById('secs').value);
  fd.append('max_frames', document.getElementById('frames').value);
  fd.append('frame_stride', document.getElementById('stride').value);
  fd.append('stop_on_alert', document.getElementById('stop').checked ? 'true' : 'false');
  const r = await fetch('/detect-webcam', { method: 'POST', body: fd });
  const j = await r.json();
  document.getElementById('out').textContent = JSON.stringify(j, null, 2);
}

function startStream() {
  const model = encodeURIComponent(document.getElementById('model').value);
  const conf = encodeURIComponent(document.getElementById('conf').value);
  const imgsz = encodeURIComponent(document.getElementById('imgsz').value);
  document.getElementById('mjpeg').src = `/stream?model_path=${model}&conf_threshold=${conf}&imgsz=${imgsz}`;
}

function stopStream() {
  document.getElementById('mjpeg').src = '';
}
</script>
</body>
</html>
        """
    )
