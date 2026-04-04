import os, io, tempfile
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydub import AudioSegment
from faster_whisper import WhisperModel


def _as_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")

PORT = int(os.getenv("ASR_PORT","5003"))
MODEL_NAME = os.getenv("WHISPER_MODEL","medium")
DEVICE = os.getenv("WHISPER_DEVICE","cpu")   # "cpu" or "cuda"
COMPUTE = os.getenv("WHISPER_COMPUTE","float32")  # cpu: int8_float32/float32, cuda: float16
VAD = os.getenv("VAD_FILTER","true").lower() in ("1","true","yes")
DEFAULT_LANGUAGE = os.getenv("ASR_LANGUAGE", "ar")
DEFAULT_BEAM_SIZE = int(os.getenv("ASR_BEAM_SIZE", "7"))
DEFAULT_BEST_OF = int(os.getenv("ASR_BEST_OF", "7"))
DEFAULT_TEMPERATURE = os.getenv("ASR_TEMPERATURE", "0.0,0.2")
DEFAULT_INITIAL_PROMPT = os.getenv("ASR_INITIAL_PROMPT", "")
DEFAULT_CONDITION_ON_PREVIOUS_TEXT = _as_bool("ASR_CONDITION_ON_PREVIOUS_TEXT", "false")
DEFAULT_REPETITION_PENALTY = float(os.getenv("ASR_REPETITION_PENALTY", "1.08"))
DEFAULT_NO_REPEAT_NGRAM_SIZE = int(os.getenv("ASR_NO_REPEAT_NGRAM_SIZE", "3"))
DEFAULT_COMPRESSION_RATIO_THRESHOLD = float(os.getenv("ASR_COMPRESSION_RATIO_THRESHOLD", "2.2"))

app = FastAPI(title="Faster-Whisper ASR", version="1.0")

# Load once at startup
model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE)

@app.get("/healthz")
def healthz():
    return {"ok": True, "model": MODEL_NAME, "device": DEVICE, "compute": COMPUTE}

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form(DEFAULT_LANGUAGE),
    beam_size: int = Form(DEFAULT_BEAM_SIZE),
    best_of: int = Form(DEFAULT_BEST_OF),
    temperature: str = Form(DEFAULT_TEMPERATURE),
    condition_on_previous_text: bool = Form(DEFAULT_CONDITION_ON_PREVIOUS_TEXT),
    initial_prompt: str = Form(DEFAULT_INITIAL_PROMPT),
    no_speech_threshold: float = Form(0.4),
    log_prob_threshold: float = Form(-1.0),
    repetition_penalty: float = Form(DEFAULT_REPETITION_PENALTY),
    no_repeat_ngram_size: int = Form(DEFAULT_NO_REPEAT_NGRAM_SIZE),
    compression_ratio_threshold: float = Form(DEFAULT_COMPRESSION_RATIO_THRESHOLD)
):
    # Accept ~any audio, normalize to wav in memory
    raw = await file.read()
    audio = AudioSegment.from_file(io.BytesIO(raw)).set_channels(1).set_frame_rate(16000)
    if audio.dBFS != float("-inf"):
        target_dbfs = -20.0
        gain = target_dbfs - audio.dBFS
        if abs(gain) <= 20:
            audio = audio.apply_gain(gain)
    temps = [float(t) for t in temperature.split(",") if t.strip()]
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        audio.export(tmp.name, format="wav")
        segments, info = model.transcribe(
            tmp.name,
            task="transcribe",
            language=language,
            beam_size=beam_size,
            vad_filter=VAD,
            temperature=temps,
            best_of=best_of,
            condition_on_previous_text=condition_on_previous_text,
            initial_prompt=initial_prompt or None,
            no_speech_threshold=no_speech_threshold,
            log_prob_threshold=log_prob_threshold,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            compression_ratio_threshold=compression_ratio_threshold
        )
        text = " ".join(seg.text.strip() for seg in segments if seg.text)
    return JSONResponse({
        "text": text.strip(),
        "language": info.language,
        "duration": getattr(info, "duration", None)
    })
