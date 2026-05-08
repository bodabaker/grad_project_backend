import json
import os
import subprocess
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import StreamingResponse

MODEL_PATH = os.getenv("OMNIVOICE_MODEL", "/models/omnivoice-base-Q8_0.gguf")
CODEC_PATH = os.getenv("OMNIVOICE_CODEC", "/models/omnivoice-tokenizer-Q8_0.gguf")
GGML_BACKEND = os.getenv("GGML_BACKEND", "CUDA0")
DEFAULT_LANG = os.getenv("OMNIVOICE_DEFAULT_LANG", "English")
DEFAULT_STEPS = int(os.getenv("OMNIVOICE_STEPS", "32"))
CLONE_ENABLED = os.getenv("OMNIVOICE_CLONE", "").lower() in ("1", "true", "yes")


class OmniVoiceServer:
    """Persistent subprocess wrapper for omnivoice-tts --server mode."""

    def __init__(self):
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._ready = False

    def _start(self):
        if self._proc and self._proc.poll() is None:
            return
        cmd = [
            "omnivoice-tts",
            "--model", MODEL_PATH,
            "--codec", CODEC_PATH,
            "--server",
            "--steps", str(DEFAULT_STEPS),
        ]
        if CLONE_ENABLED:
            cmd.append("--offload-encode")
        else:
            cmd.append("--decode-only")
        env = os.environ.copy()
        env["GGML_BACKEND"] = GGML_BACKEND
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            bufsize=1,
        )
        for line in self._proc.stderr:
            if "[Server] Ready" in line:
                self._ready = True
                break
            if self._proc.poll() is not None:
                raise RuntimeError(f"omnivoice-tts exited during startup: {line}")
        if not self._ready:
            raise RuntimeError("omnivoice-tts did not signal ready")

    def request(self, req: dict, timeout: float = 120.0) -> str:
        with self._lock:
            self._start()
            line = json.dumps(req, ensure_ascii=False) + "\n"
            self._proc.stdin.write(line)
            self._proc.stdin.flush()
            resp = self._proc.stdout.readline().strip()
            if not resp or resp == "ERROR":
                raise RuntimeError("omnivoice-tts server returned error")
            return resp

    def health(self) -> dict:
        alive = self._proc is not None and self._proc.poll() is None
        return {"alive": alive, "ready": self._ready}

    def shutdown(self):
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.stdin.close()
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()


server = OmniVoiceServer()


def _check_model_files():
    missing = []
    if not Path(MODEL_PATH).is_file():
        missing.append(MODEL_PATH)
    if not Path(CODEC_PATH).is_file():
        missing.append(CODEC_PATH)
    return missing


@asynccontextmanager
async def lifespan(app):
    missing = _check_model_files()
    if not missing:
        try:
            server._start()
            print("[App] OmniVoice server started")
        except Exception as e:
            print(f"[App] WARNING: failed to pre-start server: {e}")
    yield
    server.shutdown()


app = FastAPI(title="OmniVoice TTS Service", version="2.0.0", lifespan=lifespan)


@app.get("/healthz")
def healthz():
    missing = _check_model_files()
    if missing:
        raise HTTPException(status_code=503, detail=f"Missing model files: {missing}")
    h = server.health()
    if not h["alive"]:
        raise HTTPException(status_code=503, detail="omnivoice-tts server not running")
    return {"status": "ok", "server": h, "model": MODEL_PATH, "codec": CODEC_PATH, "backend": GGML_BACKEND, "clone": CLONE_ENABLED}


@app.post("/tts")
def tts(
    text: str = Form(..., description="Text to synthesize"),
    lang: str = Form(DEFAULT_LANG, description="Language"),
    instruct: Optional[str] = Form(None, description="Voice design instruct"),
    seed: int = Form(-1, description="RNG seed, -1 = random"),
    steps: int = Form(DEFAULT_STEPS, description="MaskGIT inference steps"),
    fmt: str = Form("wav16", description="Output format: wav16 | wav24 | wav32"),
):
    missing = _check_model_files()
    if missing:
        raise HTTPException(status_code=503, detail=f"Missing model files: {missing}")

    start = time.time()
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = str(Path(tmpdir) / "output.wav")
        req = {
            "text": text,
            "lang": lang,
            "steps": steps,
            "seed": seed,
            "fmt": fmt,
            "out": out_path,
        }
        if instruct:
            req["instruct"] = instruct
        try:
            result_path = server.request(req)
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=str(e))

        data = Path(result_path).read_bytes()

    duration = time.time() - start
    print(f"[/tts] lang={lang} steps={steps} len={len(text)} chars time={duration:.2f}s")
    return StreamingResponse(
        iter([data]),
        media_type="audio/wav",
        headers={"X-Generation-Time": str(duration)},
    )


@app.post("/clone")
async def clone(
    text: str = Form(..., description="Text to synthesize in cloned voice"),
    ref_wav: UploadFile = File(..., description="Reference WAV file"),
    ref_text: str = Form(..., description="Transcript matching the reference WAV"),
    lang: str = Form(DEFAULT_LANG, description="Language"),
    seed: int = Form(-1, description="RNG seed, -1 = random"),
    steps: int = Form(DEFAULT_STEPS, description="MaskGIT inference steps"),
    fmt: str = Form("wav16", description="Output format: wav16 | wav24 | wav32"),
):
    missing = _check_model_files()
    if missing:
        raise HTTPException(status_code=503, detail=f"Missing model files: {missing}")

    start = time.time()
    with tempfile.TemporaryDirectory() as tmpdir:
        ref_wav_path = Path(tmpdir) / "reference.wav"
        ref_wav_path.write_bytes(await ref_wav.read())

        ref_text_path = Path(tmpdir) / "ref_text.txt"
        ref_text_path.write_text(ref_text)

        out_path = str(Path(tmpdir) / "output.wav")
        req = {
            "text": text,
            "lang": lang,
            "steps": steps,
            "seed": seed,
            "fmt": fmt,
            "ref_wav": str(ref_wav_path),
            "ref_text": str(ref_text_path),
            "out": out_path,
        }
        try:
            result_path = server.request(req)
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=str(e))

        data = Path(result_path).read_bytes()

    duration = time.time() - start
    print(f"[/clone] lang={lang} steps={steps} len={len(text)} chars time={duration:.2f}s")
    return StreamingResponse(
        iter([data]),
        media_type="audio/wav",
        headers={"X-Generation-Time": str(duration)},
    )


@app.get("/")
def root():
    return {
        "service": "omnivoice-tts",
        "version": "2.0.0",
        "mode": "persistent",
        "endpoints": {"health": "/healthz", "voice_design": "POST /tts", "voice_clone": "POST /clone"},
        "model": MODEL_PATH,
        "codec": CODEC_PATH,
        "backend": GGML_BACKEND,
        "clone": CLONE_ENABLED,
    }
