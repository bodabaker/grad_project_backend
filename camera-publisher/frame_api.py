import os
import base64
import subprocess
import time
from typing import Optional
from urllib.parse import urlencode

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, Response

app = FastAPI(title="Camera Publisher API", version="1.0.0")


def _quality_to_qscale(quality: int) -> int:
    # quality: 1..100 (higher is better) -> ffmpeg q:v: 31..2 (lower is better)
    quality = max(1, min(100, int(quality)))
    return max(2, min(31, int(round(31 - (quality - 1) * (29 / 99)))))


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.get("/frame")
async def frame(
    camera_url: Optional[str] = Query(None, description="Defaults to FRAME_SOURCE env var"),
    timeout_seconds: float = Query(4.0, ge=0.5, le=30.0),
    quality: int = Query(90, ge=1, le=100),
    response_format: str = Query("binary", pattern="^(binary|base64|data_url|json_ref)$"),
):
    source = camera_url or os.getenv("FRAME_SOURCE") or os.getenv("RTSP_URL") or "rtsp://mediamtx:8554/cam"
    qscale = _quality_to_qscale(quality)

    if response_format == "json_ref":
        host = os.getenv("FRAME_PUBLIC_HOST", "camera-publisher:8060")
        query = {
            "response_format": "binary",
            "timeout_seconds": timeout_seconds,
            "quality": quality,
        }
        if camera_url:
            query["camera_url"] = camera_url
        return JSONResponse(
            content={
                "mime_type": "image/jpeg",
                "source": source,
                "frame_url": f"http://{host}/frame?{urlencode(query)}",
            }
        )

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-fflags",
        "nobuffer",
        "-flags",
        "low_delay",
        "-probesize",
        "32",
        "-analyzeduration",
        "0",
        "-avioflags",
        "direct",
        "-rtsp_transport",
        "tcp",
        "-i",
        source,
        "-frames:v",
        "1",
        "-q:v",
        str(qscale),
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "-",
    ]

    # Try a few times for transient DNS/RTSP/connect errors
    attempts = 3
    completed = None
    for attempt in range(1, attempts + 1):
        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                timeout=float(timeout_seconds),
                check=False,
            )
        except subprocess.TimeoutExpired:
            return JSONResponse(status_code=504, content={"error": "Timed out waiting for camera frame"})
        except Exception as e:
            if attempt == attempts:
                return JSONResponse(status_code=500, content={"error": f"Frame capture failed: {e}"})
            time.sleep(0.8)
            continue

        if completed is not None and completed.returncode == 0:
            break

        # non-zero return code: retry a couple times for transient network issues
        if attempt < attempts:
            time.sleep(0.8)
            continue

    if completed is None or completed.returncode != 0:
        stderr = (completed.stderr or b"").decode("utf-8", errors="ignore").strip() if completed else ""
        return JSONResponse(
            status_code=500,
            content={
                "error": "FFmpeg failed to capture frame",
                "source": source,
                "details": stderr[:500],
            },
        )

    data = completed.stdout or b""
    if not data:
        return JSONResponse(status_code=500, content={"error": "Empty frame output from FFmpeg", "source": source})

    if response_format == "base64":
        encoded = base64.b64encode(data).decode("ascii")
        return JSONResponse(
            content={
                "mime_type": "image/jpeg",
                "source": source,
                "frame_base64": encoded,
            }
        )

    if response_format == "data_url":
        encoded = base64.b64encode(data).decode("ascii")
        return JSONResponse(
            content={
                "mime_type": "image/jpeg",
                "source": source,
                "image_url": f"data:image/jpeg;base64,{encoded}",
            }
        )

    return Response(
        content=data,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store",
            "X-Camera-Source": source,
        },
    )
