import os
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import cv2
import numpy as np
import mediapipe as mp
from fastapi import FastAPI, Form, Query
from fastapi.responses import JSONResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

app = FastAPI(title="Gesture Control Service", version="1.0.0")


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


def load_ref_orb(ref_image_path: str, orb_features: int = 1500) -> Tuple[Optional[List[cv2.KeyPoint]], Optional[np.ndarray], Optional[str]]:
    if not ref_image_path:
        return None, None, "ref_image_path is empty"

    p = Path(ref_image_path)
    if not p.exists():
        return None, None, f"Reference image not found: {ref_image_path}"

    ref_img = cv2.imread(str(p))
    if ref_img is None:
        return None, None, f"Failed reading reference image: {ref_image_path}"

    orb = cv2.ORB_create(nfeatures=orb_features)
    kp, des = orb.detectAndCompute(ref_img, None)
    if des is None or len(des) == 0:
        return None, None, "Reference image has no detectable ORB descriptors"

    return kp, des, None


def run_auth(gray_frame: np.ndarray, ref_des: np.ndarray, match_threshold: int, good_distance: int, orb_features: int) -> Tuple[bool, int]:
    orb = cv2.ORB_create(nfeatures=orb_features)
    kp2, des2 = orb.detectAndCompute(gray_frame, None)
    if des2 is None or len(des2) == 0:
        return False, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(ref_des, des2), key=lambda x: x.distance)
    good = [m for m in matches if m.distance < good_distance]
    return len(good) > match_threshold, len(good)


def process_frame(
    frame: np.ndarray,
    hands,
    authorized: bool,
    require_authorization: bool,
    ref_des: Optional[np.ndarray],
    match_threshold: int,
    good_distance: int,
    orb_features: int,
    last_action_time: float,
    cooldown_duration: float,
    first_floor: str,
    second_floor: str,
) -> Tuple[np.ndarray, bool, int, float, Optional[Dict[str, Any]]]:
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    now = time.time()
    action = None
    match_count = 0

    if require_authorization and not authorized:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if ref_des is not None:
            ok, match_count = run_auth(gray, ref_des, match_threshold, good_distance, orb_features)
            if ok:
                authorized = True
        return frame, authorized, match_count, last_action_time, action

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    can_action = (now - last_action_time) > cooldown_duration

    if results.multi_hand_landmarks:
        if can_action:
            hand_lms = results.multi_hand_landmarks[0]
            y_tip = hand_lms.landmark[8].y * h
            target_floor = first_floor if y_tip < h // 2 else second_floor

            last_action_time = now
            action = {
                "target_floor": target_floor,
                "ts": now,
            }

    return frame, authorized, match_count, last_action_time, action


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/detect-webcam")
async def detect_webcam(
    camera_url: Optional[str] = Form(None),
    webcam: int = Form(0),
    first_floor: str = Form("first_floor"),
    second_floor: str = Form("second_floor"),
    require_authorization: bool = Form(False),
    ref_image_path: str = Form("/data/references/ref.jpg"),
    match_threshold: int = Form(30),
    auth_good_distance: int = Form(45),
    orb_features: int = Form(1500),
    cooldown_duration: float = Form(3.0),
    max_seconds: int = Form(20),
    max_frames: int = Form(0),
    frame_stride: int = Form(2),
    stop_on_first_action: bool = Form(False),
    include_timeline: bool = Form(False),
):
    camera_url = camera_url or os.getenv("CAMERA_URL")

    ref_des = None
    auth_error = None
    if require_authorization:
        _, ref_des, auth_error = load_ref_orb(ref_image_path, orb_features=orb_features)
        if auth_error:
            return JSONResponse(status_code=400, content={"error": auth_error, "ref_image_path": ref_image_path})

    cap = open_capture(camera_url, fallback_index=webcam)
    if not cap.isOpened():
        which = camera_url if (camera_url and is_url_like(camera_url)) else f"index {webcam}"
        return JSONResponse(status_code=500, content={"error": f"Cannot open capture ({which})"})

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    t0 = time.time()
    deadline = t0 + max_seconds if max_seconds else 0
    stride = max(1, frame_stride)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

    authorized = not require_authorization
    last_action_time = 0.0

    frame_id = 0
    processed = 0
    actions_count = 0
    floor_hits = {first_floor: 0, second_floor: 0}
    last_detected_floor: Optional[str] = None
    last_match_count = 0
    timeline: List[Dict[str, Any]] = []
    stop_reason = "max_frames_reached"

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                if max_seconds and time.time() >= deadline:
                    stop_reason = "timeout_or_end_of_capture"
                    break
                time.sleep(0.01)
                continue

            frame_id += 1
            if frame_id % stride != 0:
                continue

            processed += 1

            _, authorized, match_count, last_action_time, action = process_frame(
                frame=frame,
                hands=hands,
                authorized=authorized,
                require_authorization=require_authorization,
                ref_des=ref_des,
                match_threshold=match_threshold,
                good_distance=auth_good_distance,
                orb_features=orb_features,
                last_action_time=last_action_time,
                cooldown_duration=cooldown_duration,
                first_floor=first_floor,
                second_floor=second_floor,
            )

            last_match_count = match_count
            if action is not None:
                actions_count += 1
                hit_floor = action["target_floor"]
                if hit_floor in floor_hits:
                    floor_hits[hit_floor] += 1
                last_detected_floor = hit_floor

            if include_timeline:
                timeline.append(
                    {
                        "frame_id": frame_id,
                        "ts": time.time(),
                        "authorized": authorized,
                        "match_count": match_count,
                        "action": action,
                    }
                )

            if stop_on_first_action and action is not None:
                stop_reason = "stop_on_first_action"
                break
            if max_frames and processed >= max_frames:
                stop_reason = "max_frames_reached"
                break
            if max_seconds and time.time() >= deadline:
                stop_reason = "timeout_or_end_of_capture"
                break
    finally:
        cap.release()
        hands.close()

    result = {
        "mode": "stream" if (camera_url and is_url_like(camera_url)) else "webcam",
        "source": camera_url or f"index {int(webcam)}",
        "require_authorization": require_authorization,
        "authorized": authorized,
        "last_match_count": last_match_count,
        "frames_processed": processed,
        "detections_count": actions_count,
        "last_detected_floor": last_detected_floor,
        "floor_hits": floor_hits,
        "first_floor": first_floor,
        "second_floor": second_floor,
        "stop_reason": stop_reason,
    }
    if include_timeline:
        result["timeline"] = timeline
    return result
