import base64
import io
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from huggingface_hub import snapshot_download
from pydantic import BaseModel, Field
from starlette.responses import Response
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

XTTS_ADAPTER_MODEL_ID = os.getenv(
    "EGTTS_MODEL_ID",
    os.getenv("XTTS_ADAPTER_MODEL_ID", "ahmednabilgm7/EGTTS-Custom"),
)
XTTS_BASE_MODEL_ID = os.getenv("XTTS_BASE_MODEL_ID", "coqui/XTTS-v2")
XTTS_MODEL_DIR = Path(os.getenv("XTTS_MODEL_DIR", "/models/xtts-egy"))
XTTS_LANGUAGE = os.getenv("XTTS_LANGUAGE", "ar")
XTTS_DEVICE = os.getenv("XTTS_DEVICE", "cpu")
XTTS_DEFAULT_SPEAKER_WAV = os.getenv("XTTS_DEFAULT_SPEAKER_WAV", "/references/speaker_reference.wav")

app = FastAPI(title="XTTS v2 Egyptian Arabic LoRA Wrapper", version="1.0.0")


class SynthesisRequest(BaseModel):
    text: str = Field(min_length=1)
    language: Optional[str] = None
    speaker_wav_path: Optional[str] = None
    speaker_wav_base64: Optional[str] = None


xtts_model: Optional[Xtts] = None
xtts_config: Optional[XttsConfig] = None
xtts_lock = threading.Lock()
checkpoint_file: Optional[Path] = None


def _resolve_device() -> str:
    if XTTS_DEVICE.startswith("cuda") and torch.cuda.is_available():
        return "cuda"
    if XTTS_DEVICE == "mps" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _find_first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise RuntimeError(f"None of the expected files exist: {[str(p) for p in paths]}")


def _decode_wav_b64(audio_b64: str) -> str:
    try:
        audio_bytes = base64.b64decode(audio_b64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid speaker_wav_base64: {exc}") from exc

    with tempfile.NamedTemporaryFile(prefix="xtts_ref_", suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        return tmp.name


def _build_runtime_config(base_config_path: Path, runtime_config_path: Path, merged_checkpoint_path: Path, vocab_path: Path, dvae_path: Path, mel_stats_path: Path) -> None:
    with base_config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    model_args = data.get("model_args", {})
    model_args["xtts_checkpoint"] = str(merged_checkpoint_path)
    model_args["tokenizer_file"] = str(vocab_path)
    model_args["dvae_checkpoint"] = str(dvae_path)
    model_args["mel_norm_file"] = str(mel_stats_path)
    data["model_args"] = model_args

    with runtime_config_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


@app.on_event("startup")
def startup_load_model() -> None:
    global xtts_model, xtts_config, checkpoint_file

    XTTS_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    try:
        adapter_path = Path(
            snapshot_download(
                repo_id=XTTS_ADAPTER_MODEL_ID,
                repo_type="model",
                cache_dir=str(XTTS_MODEL_DIR),
                token=os.getenv("HF_TOKEN") or None,
                resume_download=True,
            )
        )
        base_path = Path(
            snapshot_download(
                repo_id=XTTS_BASE_MODEL_ID,
                repo_type="model",
                cache_dir=str(XTTS_MODEL_DIR),
                token=os.getenv("HF_TOKEN") or None,
                resume_download=True,
            )
        )
    except Exception as exc:
        raise RuntimeError(f"Failed downloading XTTS artifacts: {exc}") from exc

    checkpoint_file = _find_first_existing(
        [
            adapter_path / "best_model_merged.pth",
            adapter_path / "model.pth",
        ]
    )
    base_config_path = _find_first_existing([base_path / "config.json"])
    vocab_path = _find_first_existing([adapter_path / "vocab.json", base_path / "vocab.json"])
    dvae_path = _find_first_existing([base_path / "dvae.pth"])
    mel_stats_path = _find_first_existing([base_path / "mel_stats.pth"])
    speaker_file_path = _find_first_existing([base_path / "speakers_xtts.pth"])

    runtime_config_path = XTTS_MODEL_DIR / "runtime_xtts_config.json"
    _build_runtime_config(
        base_config_path=base_config_path,
        runtime_config_path=runtime_config_path,
        merged_checkpoint_path=checkpoint_file,
        vocab_path=vocab_path,
        dvae_path=dvae_path,
        mel_stats_path=mel_stats_path,
    )

    try:
        config = XttsConfig()
        config.load_json(str(runtime_config_path))

        model = Xtts.init_from_config(config)
        model.load_checkpoint(
            config,
            checkpoint_dir=str(base_path),
            checkpoint_path=str(checkpoint_file),
            vocab_path=str(vocab_path),
            eval=True,
            use_deepspeed=False,
            speaker_file_path=str(speaker_file_path),
        )
        model.to(_resolve_device())

        xtts_config = config
        xtts_model = model
    except Exception as exc:
        raise RuntimeError(f"Failed loading XTTS runtime: {exc}") from exc


@app.get("/healthz")
def healthz() -> dict[str, str]:
    if xtts_model is None or xtts_config is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}


@app.post("/synthesize", response_class=Response)
def synthesize(body: SynthesisRequest) -> Response:
    if xtts_model is None or xtts_config is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    language = (body.language or XTTS_LANGUAGE).strip() or "ar"

    speaker_path = body.speaker_wav_path or XTTS_DEFAULT_SPEAKER_WAV
    tmp_path = None

    if body.speaker_wav_base64:
        tmp_path = _decode_wav_b64(body.speaker_wav_base64)
        speaker_path = tmp_path

    if not speaker_path or not Path(speaker_path).exists():
        raise HTTPException(
            status_code=400,
            detail=f"speaker reference WAV not found: {speaker_path}",
        )

    try:
        with xtts_lock:
            out = xtts_model.synthesize(
                text=text,
                config=xtts_config,
                speaker_wav=speaker_path,
                language=language,
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"XTTS synthesis failed: {exc}") from exc
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    wav = out.get("wav") if isinstance(out, dict) else None
    if wav is None:
        raise HTTPException(status_code=500, detail="XTTS returned empty audio")

    wav_np = np.asarray(wav, dtype=np.float32)
    if wav_np.ndim > 1:
        wav_np = wav_np.squeeze()
    if wav_np.size == 0:
        raise HTTPException(status_code=500, detail="XTTS returned empty audio")

    sample_rate = int(getattr(xtts_config.audio, "output_sample_rate", 24000))
    buf = io.BytesIO()
    sf.write(buf, wav_np, sample_rate, format="WAV")

    return Response(
        content=buf.getvalue(),
        media_type="audio/wav",
        headers={
            "X-Sample-Rate": str(sample_rate),
            "X-Model": XTTS_ADAPTER_MODEL_ID,
            "X-Base-Model": XTTS_BASE_MODEL_ID,
            "X-Checkpoint": checkpoint_file.name if checkpoint_file else "unknown",
            "X-Language": language,
        },
    )
