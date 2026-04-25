import argparse
import base64
import io
import os
import time
import wave
from typing import Any, Dict, List

import numpy as np
import torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from huggingface_hub import snapshot_download
from pydantic import BaseModel


MODEL_ID = os.getenv("LAHGTNA_MODEL_ID", "oddadmix/lahgtna-chatterbox-v0")
DEVICE = os.getenv("LAHGTNA_DEVICE", "cuda")
DTYPE = os.getenv("LAHGTNA_DTYPE", "float16")
HF_HOME = os.getenv("HF_HOME", "/models/hf-cache")
LANGUAGE_ID = os.getenv("LAHGTNA_LANGUAGE_ID", "eg")
DEFAULT_SAMPLE_RATE = int(os.getenv("LAHGTNA_SAMPLE_RATE", "24000"))
CFG_WEIGHT = float(os.getenv("LAHGTNA_CFG_WEIGHT", "0.5"))
EXAGGERATION = float(os.getenv("LAHGTNA_EXAGGERATION", "0.5"))
TEMPERATURE = float(os.getenv("LAHGTNA_TEMPERATURE", "0.8"))


def _torch_dtype(dtype_name: str):
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    return torch.float16


def _download_model_snapshot() -> str:
    return snapshot_download(repo_id=MODEL_ID, cache_dir=HF_HOME)


app = FastAPI(title="lahgtna-openai-wrapper")
_tts = None
_start = time.time()


class SpeechRequest(BaseModel):
    model: str
    input: str
    voice: str | None = "default"
    response_format: str | None = "wav"


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: bool | None = False


def _load_tts():
    global _tts
    if _tts is not None:
        return _tts

    if DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("LAHGTNA_DEVICE=cuda but CUDA is not available")

    device = torch.device(DEVICE)
    model_path = _download_model_snapshot()
    _tts = ChatterboxMultilingualTTS.from_local(model_path, device)
    return _tts


def _extract_audio_and_rate(result: Any) -> tuple[np.ndarray, int]:
    if isinstance(result, dict):
        audio = result.get("audio")
        sr = int(result.get("sampling_rate", DEFAULT_SAMPLE_RATE))
        if audio is None:
            raise RuntimeError("No audio key in model output")
        return np.asarray(audio), sr

    if isinstance(result, tuple) and len(result) == 2:
        return np.asarray(result[0]), int(result[1])

    if hasattr(result, "audio"):
        sr = int(getattr(result, "sampling_rate", DEFAULT_SAMPLE_RATE))
        return np.asarray(result.audio), sr

    if isinstance(result, torch.Tensor):
        return result.detach().cpu().numpy(), DEFAULT_SAMPLE_RATE

    if isinstance(result, np.ndarray):
        return result, DEFAULT_SAMPLE_RATE

    raise RuntimeError(f"Unsupported audio output type: {type(result)}")


def _pcm16_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    if audio.ndim > 1:
        audio = audio.squeeze()
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


@app.on_event("startup")
def startup_event():
    # Eager load so /health accurately reflects readiness.
    _load_tts()


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "uptime_s": round(time.time() - _start, 2),
        "model_id": MODEL_ID,
        "device": DEVICE,
        "ready": _tts is not None,
    }


@app.get("/v1/models")
def list_models() -> Dict[str, Any]:
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": "lahgtna",
                "object": "model",
                "created": now,
                "owned_by": "oddadmix",
            }
        ],
    }


@app.post("/v1/audio/speech")
def audio_speech(body: SpeechRequest):
    if not body.input.strip():
        raise HTTPException(status_code=400, detail="input must not be empty")

    try:
        tts = _load_tts()
        output = tts.generate(
            text=body.input,
            language_id=LANGUAGE_ID,
            cfg_weight=CFG_WEIGHT,
            exaggeration=EXAGGERATION,
            temperature=TEMPERATURE,
        )

        audio, sr = _extract_audio_and_rate(output)
        wav_bytes = _pcm16_wav_bytes(audio, sr)

        response_format = (body.response_format or "wav").lower()
        if response_format == "wav":
            return Response(content=wav_bytes, media_type="audio/wav")

        if response_format == "json":
            return JSONResponse(
                {
                    "audio_base64": base64.b64encode(wav_bytes).decode("utf-8"),
                    "format": "wav",
                    "sample_rate": sr,
                    "model": body.model,
                }
            )

        raise HTTPException(status_code=400, detail="Unsupported response_format")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {exc}")


@app.post("/v1/chat/completions")
def chat_completions(body: ChatRequest):
    # Compatibility path so llama-swap can still route chat calls here.
    # Lahgtna is TTS-first, so we return guidance + echoed text.
    user_parts = [m.content for m in body.messages if m.role == "user" and m.content]
    latest = user_parts[-1] if user_parts else ""
    now = int(time.time())

    content = (
        "lahgtna is a TTS model. Use /v1/audio/speech for audio output. "
        f"Received text: {latest}"
    )

    return {
        "id": f"chatcmpl-lahgtna-{now}",
        "object": "chat.completion",
        "created": now,
        "model": body.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    return parser.parse_args()


if __name__ == "__main__":
    import uvicorn

    args = parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
