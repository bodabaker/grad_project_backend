import base64
import io
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Optional
from pydub import AudioSegment

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from huggingface_hub import snapshot_download
from pydantic import BaseModel, Field
from starlette.responses import Response
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

EGTTS_MODEL_ID = os.getenv("EGTTS_MODEL_ID", "ahmednabilgm7/EGTTS-Custom")
EGTTS_BASE_MODEL_ID = os.getenv("EGTTS_BASE_MODEL_ID", "coqui/XTTS-v2")
EGTTS_MODEL_DIR = Path(os.getenv("EGTTS_MODEL_DIR", "/models/egtts"))
EGTTS_LOCAL_MODEL_DIR = os.getenv("EGTTS_LOCAL_MODEL_DIR", "").strip()
EGTTS_LANGUAGE = os.getenv("EGTTS_LANGUAGE", "ar")
EGTTS_DEVICE = os.getenv("EGTTS_DEVICE", "cpu")
EGTTS_DEFAULT_SPEAKER_WAV = os.getenv("EGTTS_DEFAULT_SPEAKER_WAV", "/references/speaker_reference.wav")

app = FastAPI(title="egtts", version="2.0.1")


class SynthesisRequest(BaseModel):
    text: str = Field(min_length=1)
    language: Optional[str] = None
    speaker_wav_path: Optional[str] = None
    speaker_wav_base64: Optional[str] = None
    temperature: Optional[float] = Field(default=0.85, ge=0.0, le=2.0)
    length_penalty: Optional[float] = Field(default=1.0, ge=-2.0, le=2.0)
    repetition_penalty: Optional[float] = Field(default=2.0, ge=0.5, le=10.0)
    top_p: Optional[float] = Field(default=0.85, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=50, ge=0, le=1000)
    speed: Optional[float] = Field(default=1.0, ge=0.5, le=2.0)
    output_format: Optional[str] = Field(default="wav", pattern="^(wav|mp3)$")


egtts_model: Optional[Xtts] = None
egtts_config: Optional[XttsConfig] = None
egtts_lock = threading.Lock()
checkpoint_file: Optional[Path] = None


def _resolve_device() -> str:
    if EGTTS_DEVICE.startswith("cuda") and torch.cuda.is_available():
        return "cuda"
    if EGTTS_DEVICE == "mps" and torch.backends.mps.is_available():
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

    with tempfile.NamedTemporaryFile(prefix="egtts_ref_", suffix=".wav", delete=False) as tmp:
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
    global egtts_model, egtts_config, checkpoint_file

    EGTTS_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    try:
        if EGTTS_LOCAL_MODEL_DIR and Path(EGTTS_LOCAL_MODEL_DIR).exists():
            adapter_path = Path(EGTTS_LOCAL_MODEL_DIR)
        else:
            adapter_path = Path(
                snapshot_download(
                    repo_id=EGTTS_MODEL_ID,
                    repo_type="model",
                    cache_dir=str(EGTTS_MODEL_DIR),
                    token=os.getenv("HF_TOKEN") or None,
                    resume_download=True,
                )
            )
        base_path = Path(
            snapshot_download(
                repo_id=EGTTS_BASE_MODEL_ID,
                repo_type="model",
                cache_dir=str(EGTTS_MODEL_DIR),
                token=os.getenv("HF_TOKEN") or None,
                resume_download=True,
            )
        )
    except Exception as exc:
        raise RuntimeError(f"Failed downloading EGTTS artifacts: {exc}") from exc

    checkpoint_file = _find_first_existing(
        [
            adapter_path / "best_model_merged.pth",
            adapter_path / "best_model.pth",
            adapter_path / "model.pth",
            adapter_path / "checkpoint.pth",
        ]
    )
    base_config_path = _find_first_existing([adapter_path / "config.json", base_path / "config.json"])
    vocab_path = _find_first_existing([adapter_path / "vocab.json", base_path / "vocab.json"])
    dvae_path = _find_first_existing([base_path / "dvae.pth"])
    mel_stats_path = _find_first_existing([base_path / "mel_stats.pth"])
    speaker_file_path = _find_first_existing([base_path / "speakers_xtts.pth"])

    runtime_config_path = EGTTS_MODEL_DIR / "runtime_egtts_config.json"
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

        egtts_config = config
        egtts_model = model
    except Exception as exc:
        raise RuntimeError(f"Failed loading EGTTS runtime: {exc}") from exc


@app.get("/healthz")
def healthz() -> dict[str, str]:
    if egtts_model is None or egtts_config is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}


@app.post("/synthesize", response_class=Response)
def synthesize(body: SynthesisRequest) -> Response:
    if egtts_model is None or egtts_config is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    language = (body.language or EGTTS_LANGUAGE).strip() or "ar"

    speaker_path = body.speaker_wav_path or EGTTS_DEFAULT_SPEAKER_WAV
    tmp_path = None

    if body.speaker_wav_base64:
        tmp_path = _decode_wav_b64(body.speaker_wav_base64)
        speaker_path = tmp_path

    speaker_wav_for_infer: Optional[str] = None
    speaker_id_for_infer: Optional[int] = 0
    if speaker_path and Path(speaker_path).exists():
        speaker_wav_for_infer = speaker_path
        speaker_id_for_infer = None

    try:
        with egtts_lock:
            # Build kwargs with only provided parameters
            synthesis_kwargs = {
                "text": text,
                "config": egtts_config,
                "speaker_wav": speaker_wav_for_infer,
                "language": language,
                "speaker_id": speaker_id_for_infer,
            }
            # Add optional inference parameters if provided
            if body.temperature is not None:
                synthesis_kwargs["temperature"] = body.temperature
            if body.length_penalty is not None:
                synthesis_kwargs["length_penalty"] = body.length_penalty
            if body.repetition_penalty is not None:
                synthesis_kwargs["repetition_penalty"] = body.repetition_penalty
            if body.top_p is not None:
                synthesis_kwargs["top_p"] = body.top_p
            if body.top_k is not None:
                synthesis_kwargs["top_k"] = body.top_k
            if body.speed is not None:
                synthesis_kwargs["speed"] = body.speed
            
            out = egtts_model.synthesize(**synthesis_kwargs)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"EGTTS synthesis failed: {exc}") from exc
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    wav = out.get("wav") if isinstance(out, dict) else None
    if wav is None:
        raise HTTPException(status_code=500, detail="EGTTS returned empty audio")

    wav_np = np.asarray(wav, dtype=np.float32)
    if wav_np.ndim > 1:
        wav_np = wav_np.squeeze()
    if wav_np.size == 0:
        raise HTTPException(status_code=500, detail="EGTTS returned empty audio")

    sample_rate = int(getattr(egtts_config.audio, "output_sample_rate", 24000))
    output_format = (body.output_format or "wav").lower()

    buf = io.BytesIO()
    sf.write(buf, wav_np, sample_rate, format="WAV")
    buf.seek(0)

    if output_format == "mp3":
        try:
            audio = AudioSegment.from_wav(buf)
            mp3_buf = io.BytesIO()
            audio.export(mp3_buf, format="mp3", bitrate="192k")
            content = mp3_buf.getvalue()
            media_type = "audio/mpeg"
            content_disposition = 'inline; filename="audio.mp3"'
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"MP3 conversion failed: {exc}") from exc
    else:
        content = buf.getvalue()
        media_type = "audio/wav"
        content_disposition = 'inline; filename="audio.wav"'

    return Response(
        content=content,
        media_type=media_type,
        headers={
            "X-Sample-Rate": str(sample_rate),
            "X-Model": EGTTS_MODEL_ID,
            "X-Base-Model": EGTTS_BASE_MODEL_ID,
            "X-Checkpoint": checkpoint_file.name if checkpoint_file else "unknown",
            "X-Language": language,
            "X-Output-Format": output_format,
            "Content-Disposition": content_disposition,
        },
    )
