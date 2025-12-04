import json
import os
import subprocess
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.responses import Response

MODEL_PATH = os.environ.get("PIPER_MODEL", "/app/models/ar_JO-kareem-medium/model.onnx")
CONFIG_PATH = os.environ.get("PIPER_CONFIG", "/app/models/ar_JO-kareem-medium/model.json")

app = FastAPI(title="Piper HTTP Wrapper", version="1.0.0")


class SynthesisRequest(BaseModel):
    text: str
    speaker: int | None = None
    noise_scale: float | None = None
    length_scale: float | None = None
    noise_w: float | None = None


def _load_sample_rate(config_path: str) -> int:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return int(data.get("audio", {}).get("sample_rate", 22050))
    except Exception:
        return 22050


@app.post("/synthesize", response_class=Response)
def synthesize(body: SynthesisRequest):
    model = Path(MODEL_PATH)
    config = Path(CONFIG_PATH)
    if not model.exists() or not config.exists():
        raise HTTPException(status_code=500, detail="Model or config not found in container.")
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="text is required.")

    cmd = [
        "/opt/piper/piper",
        "--model",
        str(model),
        "--config",
        str(config),
        "--output_file",
        "-",  # write WAV to stdout
    ]
    if body.speaker is not None:
        cmd += ["--speaker", str(body.speaker)]
    if body.noise_scale is not None:
        cmd += ["--noise_scale", str(body.noise_scale)]
    if body.length_scale is not None:
        cmd += ["--length_scale", str(body.length_scale)]
    if body.noise_w is not None:
        cmd += ["--noise_w", str(body.noise_w)]

    try:
        result = subprocess.run(
            cmd,
            input=body.text.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Piper failed: {e.stderr.decode('utf-8', errors='ignore')}",
        )

    wav_bytes = result.stdout
    sample_rate = _load_sample_rate(str(config))
    headers = {
        "Content-Type": "audio/wav",
        "X-Sample-Rate": str(sample_rate),
    }
    return Response(content=wav_bytes, media_type="audio/wav", headers=headers)
