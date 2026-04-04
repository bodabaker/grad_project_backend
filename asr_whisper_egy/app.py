import io
import os
import importlib

import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

try:
    _peft = importlib.import_module("peft")
except ModuleNotFoundError:
    _peft = None

PeftConfig = getattr(_peft, "PeftConfig", None) if _peft is not None else None
PeftModel = getattr(_peft, "PeftModel", None) if _peft is not None else None

PORT = int(os.getenv("ASR_PORT", "5004"))
MODEL_ID = os.getenv("HF_WHISPER_MODEL", "MAdel121/whisper-small-egyptian-arabic")
LANGUAGE = os.getenv("ASR_LANGUAGE", "ar")
CHUNK_LENGTH_S = int(os.getenv("ASR_CHUNK_LENGTH_S", "20"))
BATCH_SIZE = int(os.getenv("ASR_BATCH_SIZE", "8"))
PRELOAD_MODEL = os.getenv("ASR_PRELOAD_MODEL", "true").strip().lower() in ("1", "true", "yes", "on")
CHUNK_OVERLAP_MS = int(os.getenv("ASR_CHUNK_OVERLAP_MS", "800"))
NUM_BEAMS = int(os.getenv("ASR_NUM_BEAMS", "5"))
NO_REPEAT_NGRAM_SIZE = int(os.getenv("ASR_NO_REPEAT_NGRAM_SIZE", "3"))
REPETITION_PENALTY = float(os.getenv("ASR_REPETITION_PENALTY", "1.08"))
LENGTH_PENALTY = float(os.getenv("ASR_LENGTH_PENALTY", "1.0"))
MAX_NEW_TOKENS = int(os.getenv("ASR_MAX_NEW_TOKENS", "256"))

app = FastAPI(title="Whisper Egyptian Arabic ASR", version="1.0")

# Global lazy loader
_model_cache = {"processor": None, "model": None, "asr_pipe": None}


def _normalize_audio(audio: AudioSegment) -> AudioSegment:
    audio = audio.set_channels(1).set_frame_rate(16000).high_pass_filter(80).low_pass_filter(7600)
    if audio.dBFS != float("-inf"):
        target_dbfs = -20.0
        gain = target_dbfs - audio.dBFS
        if abs(gain) <= 20:
            audio = audio.apply_gain(gain)
    return audio


def _trim_nonsilent(audio: AudioSegment) -> AudioSegment:
    if len(audio) < 300:
        return audio
    silence_thresh = min(audio.dBFS - 16, -38) if audio.dBFS != float("-inf") else -38
    regions = detect_nonsilent(audio, min_silence_len=250, silence_thresh=silence_thresh)
    if not regions:
        return audio
    trimmed = AudioSegment.silent(duration=0, frame_rate=audio.frame_rate)
    for start, end in regions:
        trimmed += audio[max(0, start - 80): min(len(audio), end + 80)]
    return trimmed if len(trimmed) > 0 else audio


def _iter_chunks(audio: AudioSegment):
    chunk_ms = max(1000, CHUNK_LENGTH_S * 1000)
    overlap_ms = max(0, min(CHUNK_OVERLAP_MS, chunk_ms // 3))
    step_ms = max(500, chunk_ms - overlap_ms)
    for start in range(0, len(audio), step_ms):
        chunk = audio[start:start + chunk_ms]
        if len(chunk) > 80:
            yield chunk


def _collapse_loops(text: str) -> str:
    tokens = text.split()
    if len(tokens) < 4:
        return text.strip()
    out = []
    i = 0
    max_ngram = 5
    while i < len(tokens):
        collapsed = False
        max_n = min(max_ngram, (len(tokens) - i) // 2)
        for n in range(max_n, 0, -1):
            phrase = tokens[i:i + n]
            j = i + n
            repeats = 1
            while j + n <= len(tokens) and tokens[j:j + n] == phrase:
                repeats += 1
                j += n
            if repeats > 1:
                out.extend(phrase)
                i = j
                collapsed = True
                break
        if not collapsed:
            out.append(tokens[i])
            i += 1
    return " ".join(out).strip()


def _load_model():
    """Load model on first request (lazy loading to not block startup)"""
    if _model_cache["processor"] is None:
        print(f"Loading model: {MODEL_ID}", flush=True)
        processor_source = MODEL_ID
        model = None

        if PeftConfig is not None and PeftModel is not None:
            try:
                peft_config = PeftConfig.from_pretrained(MODEL_ID)
                base_model_id = peft_config.base_model_name_or_path
                if base_model_id:
                    processor_source = base_model_id
                    base_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                        base_model_id,
                        torch_dtype=torch.float32,
                        low_cpu_mem_usage=True,
                    )
                    model = PeftModel.from_pretrained(base_model, MODEL_ID)
                    print(f"Loaded PEFT adapter on base model: {base_model_id}", flush=True)
            except Exception as exc:
                print(f"PEFT adapter load skipped for {MODEL_ID}: {exc}", flush=True)

        if model is None:
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
            print("Loaded full model directly", flush=True)

        _model_cache["processor"] = AutoProcessor.from_pretrained(processor_source)
        _model_cache["model"] = model.to("cpu")
        _model_cache["model"].eval()
        # Some Whisper checkpoints ship with forced decoder ids that conflict
        # with transformers' ASR pipeline when task/language are passed.
        # Clear them so generation works cleanly on CPU.
        if hasattr(_model_cache["model"], "config"):
            _model_cache["model"].config.forced_decoder_ids = None
        if hasattr(_model_cache["model"], "generation_config"):
            _model_cache["model"].generation_config.forced_decoder_ids = None
        print(f"Model loaded successfully", flush=True)
    return _model_cache["processor"], _model_cache["model"]


@app.on_event("startup")
def startup_event():
    if PRELOAD_MODEL:
        _load_model()


@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "model": MODEL_ID,
        "device": "cpu",
        "chunk_length_s": CHUNK_LENGTH_S,
        "chunk_overlap_ms": CHUNK_OVERLAP_MS,
        "num_beams": NUM_BEAMS,
        "batch_size": BATCH_SIZE,
    }


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form(LANGUAGE),
):
    processor, model = _load_model()
    lang = (language or LANGUAGE).strip().lower()
    # Whisper expects language names for decoder prompts.
    lang_map = {
        "ar": "arabic",
        "ar-eg": "arabic",
        "egy": "arabic",
        "egyptian": "arabic",
    }
    whisper_lang = lang_map.get(lang, lang)
    raw = await file.read()
    audio = _trim_nonsilent(_normalize_audio(AudioSegment.from_file(io.BytesIO(raw))))
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=whisper_lang, task="transcribe")

    chunk_texts = []
    for chunk in _iter_chunks(audio):
        samples = torch.tensor(chunk.get_array_of_samples(), dtype=torch.float32) / 32768.0
        inputs = processor(samples.tolist(), sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            generated_ids = model.generate(
                inputs["input_features"],
                forced_decoder_ids=forced_decoder_ids,
                num_beams=NUM_BEAMS,
                length_penalty=LENGTH_PENALTY,
                no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                repetition_penalty=REPETITION_PENALTY,
                max_new_tokens=MAX_NEW_TOKENS,
            )
        piece = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        if piece:
            chunk_texts.append(piece)

    text = _collapse_loops(" ".join(chunk_texts))

    return JSONResponse(
        {
            "text": text.strip(),
            "language": language,
        }
    )

