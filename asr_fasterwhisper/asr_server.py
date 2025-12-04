import os, io, tempfile
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydub import AudioSegment
from faster_whisper import WhisperModel

PORT = int(os.getenv("ASR_PORT","5003"))
MODEL_NAME = os.getenv("WHISPER_MODEL","medium")
DEVICE = os.getenv("WHISPER_DEVICE","cpu")   # "cpu" or "cuda"
COMPUTE = os.getenv("WHISPER_COMPUTE","float32")  # cpu: int8_float32/float32, cuda: float16
VAD = os.getenv("VAD_FILTER","true").lower() in ("1","true","yes")

app = FastAPI(title="Faster-Whisper ASR", version="1.0")

# Load once at startup
model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE)

@app.get("/healthz")
def healthz():
    return {"ok": True, "model": MODEL_NAME, "device": DEVICE, "compute": COMPUTE}

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form("ar"),
    beam_size: int = Form(5),
    temperature: str = Form("0.0,0.2,0.4")
):
    # Accept ~any audio, normalize to wav in memory
    raw = await file.read()
    audio = AudioSegment.from_file(io.BytesIO(raw))
    temps = [float(t) for t in temperature.split(",") if t.strip()]
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        audio.export(tmp.name, format="wav")
        segments, info = model.transcribe(
            tmp.name,
            language=language,
            beam_size=beam_size,
            vad_filter=VAD,
            temperature=temps,
            best_of=5
        )
        text = "".join(seg.text for seg in segments)
    return JSONResponse({
        "text": text.strip(),
        "language": info.language,
        "duration": getattr(info, "duration", None)
    })
