import os
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

import cv2
import numpy as np
import easyocr
from fastapi import FastAPI, UploadFile, File, Form, Query, Response
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

app = FastAPI(title="OCR Service", version="1.0.0")

_READER_CACHE: Dict[str, easyocr.Reader] = {}


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


def parse_rotation_info(rotation_info: Optional[str]) -> Optional[List[int]]:
    if not rotation_info:
        return None
    values: List[int] = []
    for item in rotation_info.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            angle = int(item)
        except ValueError:
            continue
        if angle in (90, 180, 270):
            values.append(angle)
    return values or None


def get_reader(languages: str = "en", gpu: bool = False) -> easyocr.Reader:
    """Get or create a cached EasyOCR reader."""
    cache_key = f"{languages}_{gpu}"
    if cache_key not in _READER_CACHE:
        lang_list = [l.strip() for l in languages.split(",") if l.strip()]
        logging.info("Loading EasyOCR reader for languages: %s (gpu=%s)", lang_list, gpu)
        _READER_CACHE[cache_key] = easyocr.Reader(lang_list, gpu=gpu)
    return _READER_CACHE[cache_key]


def preprocess_for_ocr(bgr: np.ndarray, upscale_factor: float = 2.0) -> np.ndarray:
    """
    Preprocess image to improve OCR accuracy while keeping color
    (EasyOCR recognition is trained on color images — converting to grayscale hurts it).
    1. Denoise on original resolution (avoids amplifying noise after upscale)
    2. Upscale with cubic interpolation
    3. CLAHE on the L-channel (LAB space) — improves contrast without color distortion
    4. Unsharp mask to sharpen edges
    """
    # 1. Denoise before upscaling
    bgr = cv2.fastNlMeansDenoisingColored(bgr, None, h=6, hColor=6,
                                           templateWindowSize=7, searchWindowSize=21)

    # 2. Upscale
    if upscale_factor != 1.0:
        h, w = bgr.shape[:2]
        bgr = cv2.resize(bgr, (int(w * upscale_factor), int(h * upscale_factor)),
                         interpolation=cv2.INTER_CUBIC)

    # 3. CLAHE in LAB space (boosts contrast, keeps colour)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    bgr = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # 4. Sharpen
    blurred = cv2.GaussianBlur(bgr, (0, 0), 3)
    bgr = cv2.addWeighted(bgr, 1.5, blurred, -0.5, 0)

    return bgr


def preprocess_binary_for_ocr(bgr: np.ndarray, upscale_factor: float = 2.0) -> np.ndarray:
    """
    Secondary preprocessing path focused on tiny/high-contrast text.
    Produces a binary image (adaptive threshold), useful when color path misses text.
    """
    if upscale_factor != 1.0:
        h, w = bgr.shape[:2]
        bgr = cv2.resize(bgr, (int(w * upscale_factor), int(h * upscale_factor)), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, h=8, templateWindowSize=7, searchWindowSize=21)
    bin_img = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        9,
    )
    # EasyOCR works with grayscale too; keep this path separate from color pipeline.
    return bin_img


def _box_to_xyxy(points: np.ndarray):
    return (int(points[:, 0].min()), int(points[:, 1].min()),
            int(points[:, 0].max()), int(points[:, 1].max()))


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0.0


def _deduplicate(raw: list, iou_thr: float = 0.4) -> list:
    """NMS-style deduplication: keep highest-confidence detection when boxes overlap."""
    raw = sorted(raw, key=lambda d: d["confidence"], reverse=True)
    kept = []
    for det in raw:
        b = det["bbox"]
        box = (b["x1"], b["y1"], b["x2"], b["y2"])
        if not any(_iou(box, (k["bbox"]["x1"], k["bbox"]["y1"],
                              k["bbox"]["x2"], k["bbox"]["y2"])) > iou_thr
                   for k in kept):
            kept.append(det)
    return kept


def detect_text_on_frame(
    bgr: np.ndarray,
    reader: easyocr.Reader,
    conf_threshold: float = 0.3,
    upscale_factor: float = 2.0,
    multi_scale: bool = True,
    beam_width: int = 10,
    allowlist: Optional[str] = None,
    rotation_info: Optional[List[int]] = None,
    min_size: int = 5,
    text_threshold: float = 0.5,
    low_text: float = 0.25,
    link_threshold: float = 0.3,
    canvas_size: int = 3200,
    mag_ratio: float = 2.0,
    contrast_ths: float = 0.05,
    adjust_contrast: float = 0.7,
    width_ths: float = 0.8,
    add_margin: float = 0.2,
    use_binary_variant: bool = True,
    min_text_len: int = 2,
    single_char_min_conf: float = 0.9,
    min_box_area: int = 120,
    allow_single_char: bool = False,
) -> tuple:
    """
    Multi-scale OCR:
    Runs EasyOCR at two scales (upscale_factor and upscale_factor×1.5) then
    merges and deduplicates results.  This catches both large and small text.
    """
    start = time.time()

    scales = [upscale_factor, upscale_factor * 1.5] if multi_scale else [upscale_factor]
    raw_detections: List[Dict[str, Any]] = []

    for scale in scales:
        variants = [preprocess_for_ocr(bgr, upscale_factor=scale)]
        if use_binary_variant:
            variants.append(preprocess_binary_for_ocr(bgr, upscale_factor=scale))

        for processed in variants:
            results = reader.readtext(
                processed,
                detail=1,
                paragraph=False,
                decoder="beamsearch",
                beamWidth=beam_width,
                allowlist=allowlist,
                rotation_info=rotation_info,
                min_size=min_size,
                contrast_ths=contrast_ths,
                adjust_contrast=adjust_contrast,
                text_threshold=text_threshold,
                low_text=low_text,
                link_threshold=link_threshold,
                canvas_size=canvas_size,
                mag_ratio=mag_ratio,
                width_ths=width_ths,
                add_margin=add_margin,
            ) or []

            inv_scale = 1.0 / scale
            for bbox, text, score in results:
                text = text.strip()
                if not text or score < conf_threshold:
                    continue
                points = (np.array(bbox, dtype=np.float32) * inv_scale).astype(np.int32)
                x1, y1, x2, y2 = _box_to_xyxy(points)
                w = max(0, x2 - x1)
                h = max(0, y2 - y1)
                area = w * h

                # Filter tiny/noisy boxes and isolated characters unless very confident.
                if area < min_box_area:
                    continue
                if len(text) == 1 and not allow_single_char:
                    continue
                if len(text) < min_text_len and float(score) < single_char_min_conf:
                    continue

                raw_detections.append({
                    "text": text,
                    "confidence": float(score),
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "_points": points,
                })

    deduped = _deduplicate(raw_detections)

    detections = [{"text": d["text"], "confidence": d["confidence"], "bbox": d["bbox"]}
                  for d in deduped]
    draw_boxes = [d["_points"] for d in deduped]
    draw_labels = [f"{d['text']} {d['confidence']:.2f}" for d in deduped]

    elapsed = (time.time() - start) * 1000
    logging.info("OCR finished in %.1fms across %d scales; detections=%d",
                 elapsed, len(scales), len(detections))
    return detections, draw_boxes, draw_labels


def annotate(frame: np.ndarray, boxes: List[np.ndarray], labels: List[str]) -> np.ndarray:
    """Annotate frame with detected text boxes and labels."""
    for points, label in zip(boxes, labels):
        # Draw polygon around text
        cv2.polylines(frame, [points], True, (0, 255, 0), 2)
        # Get top-left for label
        x1, y1 = points[0].astype(int)
        cv2.rectangle(frame, (x1, max(0, y1 - 28)), (x1 + len(label) * 8, y1), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, label, (x1 + 4, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/detect-image")
async def detect_image(
    file: UploadFile = File(...),
    languages: str = Form("en"),
    gpu: bool = Form(False),
    conf_threshold: float = Form(0.3),
    upscale_factor: float = Form(2.0),
    multi_scale: bool = Form(True),
    beam_width: int = Form(10),
    allowlist: Optional[str] = Form(None),
    rotation_info: Optional[str] = Form(None),
    min_size: int = Form(5),
    text_threshold: float = Form(0.5),
    low_text: float = Form(0.25),
    link_threshold: float = Form(0.3),
    canvas_size: int = Form(3200),
    mag_ratio: float = Form(2.0),
    contrast_ths: float = Form(0.05),
    adjust_contrast: float = Form(0.7),
    width_ths: float = Form(0.8),
    add_margin: float = Form(0.2),
    use_binary_variant: bool = Form(True),
    min_text_len: int = Form(2),
    single_char_min_conf: float = Form(0.9),
    min_box_area: int = Form(120),
    allow_single_char: bool = Form(False),
    annotated_out: Optional[str] = Form(None),
):
    """Detect text in uploaded image."""
    try:
        reader = get_reader(languages, gpu)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Failed loading reader: {e}"})

    payload = await file.read()
    np_bytes = np.frombuffer(payload, dtype=np.uint8)
    image_bgr = cv2.imdecode(np_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image upload"})

    parsed_rotation_info = parse_rotation_info(rotation_info)
    detections, boxes, labels = detect_text_on_frame(
        image_bgr,
        reader,
        conf_threshold,
        upscale_factor,
        multi_scale,
        beam_width,
        allowlist,
        parsed_rotation_info,
        min_size,
        text_threshold,
        low_text,
        link_threshold,
        canvas_size,
        mag_ratio,
        contrast_ths,
        adjust_contrast,
        width_ths,
        add_margin,
        use_binary_variant,
        min_text_len,
        single_char_min_conf,
        min_box_area,
        allow_single_char,
    )

    saved = None
    if annotated_out and detections:
        out_path = Path(annotated_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ann = annotate(image_bgr.copy(), boxes, labels)
        if cv2.imwrite(str(out_path), ann):
            saved = str(out_path)

    return {
        "mode": "image",
        "languages": languages,
        "gpu": gpu,
        "conf_threshold": conf_threshold,
        "upscale_factor": upscale_factor,
        "multi_scale": multi_scale,
        "beam_width": beam_width,
        "allowlist": allowlist,
        "rotation_info": parsed_rotation_info,
        "min_size": min_size,
        "text_threshold": text_threshold,
        "low_text": low_text,
        "link_threshold": link_threshold,
        "canvas_size": canvas_size,
        "mag_ratio": mag_ratio,
        "contrast_ths": contrast_ths,
        "adjust_contrast": adjust_contrast,
        "width_ths": width_ths,
        "add_margin": add_margin,
        "use_binary_variant": use_binary_variant,
        "min_text_len": min_text_len,
        "single_char_min_conf": single_char_min_conf,
        "min_box_area": min_box_area,
        "allow_single_char": allow_single_char,
        "detections": detections,
        "annotated_out": saved,
    }


@app.post("/detect-webcam")
async def detect_webcam(
    camera_url: Optional[str] = Query(None),
    languages: str = Query("en"),
    gpu: bool = Query(False),
    conf_threshold: float = Query(0.3),
    upscale_factor: float = Query(2.0),
    multi_scale: bool = Query(False),
    beam_width: int = Query(5),
    allowlist: Optional[str] = Query(None),
    rotation_info: Optional[str] = Query(None),
    min_size: int = Query(5),
    text_threshold: float = Query(0.5),
    low_text: float = Query(0.25),
    link_threshold: float = Query(0.3),
    canvas_size: int = Query(3200),
    mag_ratio: float = Query(2.0),
    contrast_ths: float = Query(0.05),
    adjust_contrast: float = Query(0.7),
    width_ths: float = Query(0.8),
    add_margin: float = Query(0.2),
    use_binary_variant: bool = Query(True),
    min_text_len: int = Query(2),
    single_char_min_conf: float = Query(0.9),
    min_box_area: int = Query(120),
    allow_single_char: bool = Query(False),
    frame_max_width: int = Query(0),
    max_frames: int = Query(5),
    timeout_seconds: float = Query(30.0),
    stop_on_text: bool = Query(True),
    annotated_out: Optional[str] = Query(None),
):
    """Detect text in live camera stream.
    Stops as soon as text is found (stop_on_text=true) or when
    timeout_seconds / max_frames is reached, whichever comes first.
    Note: each frame takes ~4s on CPU, so keep max_frames small.
    """
    camera_url = camera_url or os.getenv("CAMERA_URL")

    try:
        reader = get_reader(languages, gpu)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Failed loading reader: {e}"})

    cap = open_capture(camera_url)
    if not cap.isOpened():
        return JSONResponse(status_code=400, content={"error": "Failed to open camera"})

    out_dir = ensure_dir(annotated_out) if annotated_out else None
    frames_processed = 0
    frames_with_text = 0
    all_detections = []
    stop_reason = "max_frames_reached"
    deadline = time.time() + timeout_seconds
    parsed_rotation_info = parse_rotation_info(rotation_info)

    try:
        while frames_processed < max_frames:
            if time.time() >= deadline:
                stop_reason = "timeout"
                break

            ret, frame = cap.read()
            if not ret:
                stop_reason = "end_of_stream"
                break

            if frame_max_width > 0:
                h, w = frame.shape[:2]
                if w > frame_max_width:
                    nh = int(h * frame_max_width / w)
                    frame = cv2.resize(frame, (frame_max_width, nh), interpolation=cv2.INTER_AREA)
            detections, boxes, labels = detect_text_on_frame(
                frame,
                reader,
                conf_threshold,
                upscale_factor,
                multi_scale,
                beam_width,
                allowlist,
                parsed_rotation_info,
                min_size,
                text_threshold,
                low_text,
                link_threshold,
                canvas_size,
                mag_ratio,
                contrast_ths,
                adjust_contrast,
                width_ths,
                add_margin,
                use_binary_variant,
                min_text_len,
                single_char_min_conf,
                min_box_area,
                allow_single_char,
            )
            frames_processed += 1

            # Check timeout again after OCR (each frame can take several seconds on CPU)
            timed_out = time.time() >= deadline

            if detections:
                frames_with_text += 1
                all_detections.extend(detections)
                if out_dir:
                    ann = annotate(frame.copy(), boxes, labels)
                    cv2.imwrite(str(out_dir / f"frame_{frames_processed:05d}.jpg"), ann)
                if stop_on_text:
                    stop_reason = "text_found"
                    break

            if timed_out:
                stop_reason = "timeout"
                break

    finally:
        cap.release()

    return {
        "mode": "webcam",
        "camera_url": camera_url,
        "languages": languages,
        "gpu": gpu,
        "conf_threshold": conf_threshold,
        "upscale_factor": upscale_factor,
        "multi_scale": multi_scale,
        "beam_width": beam_width,
        "allowlist": allowlist,
        "rotation_info": parsed_rotation_info,
        "min_size": min_size,
        "text_threshold": text_threshold,
        "low_text": low_text,
        "link_threshold": link_threshold,
        "canvas_size": canvas_size,
        "mag_ratio": mag_ratio,
        "contrast_ths": contrast_ths,
        "adjust_contrast": adjust_contrast,
        "width_ths": width_ths,
        "add_margin": add_margin,
        "use_binary_variant": use_binary_variant,
        "min_text_len": min_text_len,
        "single_char_min_conf": single_char_min_conf,
        "min_box_area": min_box_area,
        "allow_single_char": allow_single_char,
        "frame_max_width": frame_max_width,
        "frames_processed": frames_processed,
        "frames_with_text": frames_with_text,
        "frames_without_text": frames_processed - frames_with_text,
        "text_ratio": frames_with_text / frames_processed if frames_processed > 0 else 0.0,
        "all_detections": all_detections,
        "stop_reason": stop_reason,
    }


@app.get("/stream")
async def stream(
    camera_url: Optional[str] = Query(None),
    languages: str = Query("en"),
    gpu: bool = Query(False),
    conf_threshold: float = Query(0.3),
    upscale_factor: float = Query(2.0),
    multi_scale: bool = Query(False),
    beam_width: int = Query(5),
    allowlist: Optional[str] = Query(None),
    rotation_info: Optional[str] = Query(None),
    min_size: int = Query(5),
    text_threshold: float = Query(0.5),
    low_text: float = Query(0.25),
    link_threshold: float = Query(0.3),
    canvas_size: int = Query(3200),
    mag_ratio: float = Query(2.0),
    contrast_ths: float = Query(0.05),
    adjust_contrast: float = Query(0.7),
    width_ths: float = Query(0.8),
    add_margin: float = Query(0.2),
    use_binary_variant: bool = Query(True),
    min_text_len: int = Query(2),
    single_char_min_conf: float = Query(0.9),
    min_box_area: int = Query(120),
    allow_single_char: bool = Query(False),
    frame_max_width: int = Query(0),
):
    """Stream annotated video with OCR overlays."""
    camera_url = camera_url or os.getenv("CAMERA_URL")

    try:
        reader = get_reader(languages, gpu)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Failed loading reader: {e}"})

    parsed_rotation_info = parse_rotation_info(rotation_info)

    async def generate():
        cap = open_capture(camera_url)
        if not cap.isOpened():
            return

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_max_width > 0:
                    h, w = frame.shape[:2]
                    if w > frame_max_width:
                        nh = int(h * frame_max_width / w)
                        frame = cv2.resize(frame, (frame_max_width, nh), interpolation=cv2.INTER_AREA)
                detections, boxes, labels = detect_text_on_frame(
                    frame,
                    reader,
                    conf_threshold,
                    upscale_factor,
                    multi_scale,
                    beam_width,
                    allowlist,
                    parsed_rotation_info,
                    min_size,
                    text_threshold,
                    low_text,
                    link_threshold,
                    canvas_size,
                    mag_ratio,
                    contrast_ths,
                    adjust_contrast,
                    width_ths,
                    add_margin,
                    use_binary_variant,
                    min_text_len,
                    single_char_min_conf,
                    min_box_area,
                    allow_single_char,
                )

                if detections:
                    frame = annotate(frame, boxes, labels)
                
                ret, buffer = cv2.imencode(".jpg", frame)
                frame_bytes = buffer.tobytes()
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n"
                       b"Content-length: " + str(len(frame_bytes)).encode() + b"\r\n\r\n" + frame_bytes + b"\r\n")
        finally:
            cap.release()
    
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/ui")
async def ui():
    """Simple HTML UI for testing."""
    return HTMLResponse("""
    <html>
    <head>
        <title>OCR Service</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
            input, select { padding: 8px; margin: 5px 0; }
            button { padding: 8px 15px; background: #007bff; color: white; border: none; border-radius: 3px; cursor: pointer; }
            button:hover { background: #0056b3; }
            img, video { max-width: 100%; margin: 10px 0; border: 1px solid #ccc; }
        </style>
    </head>
    <body>
        <h1>OCR Service</h1>
        
        <div class="section">
            <h2>Upload Image</h2>
            <form id="imageForm">
                <input type="file" id="imageFile" accept="image/*" required><br>
                <label>Languages: <input type="text" id="languages" value="en"></label><br>
                <label>Confidence Threshold: <input type="number" id="confThreshold" value="0.1" step="0.01" min="0" max="1"></label><br>
                <button type="submit">Detect Text</button>
            </form>
            <div id="imageResult"></div>
        </div>
        
        <div class="section">
            <h2>Live Camera Stream</h2>
            <button onclick="toggleStream()">Start Stream</button>
            <img id="stream" style="display:none;" src="/stream">
        </div>
        
        <script>
            document.getElementById('imageForm').onsubmit = async (e) => {
                e.preventDefault();
                const formData = new FormData();
                formData.append('file', document.getElementById('imageFile').files[0]);
                formData.append('languages', document.getElementById('languages').value);
                formData.append('conf_threshold', document.getElementById('confThreshold').value);
                
                const response = await fetch('/detect-image', { method: 'POST', body: formData });
                const data = await response.json();
                document.getElementById('imageResult').innerHTML = '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
            };
            
            function toggleStream() {
                const stream = document.getElementById('stream');
                stream.style.display = stream.style.display === 'none' ? 'block' : 'none';
            }
        </script>
    </body>
    </html>
    """)
