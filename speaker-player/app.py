import os
import subprocess
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

AUDIO_LIBRARY_DIR = Path(os.getenv("AUDIO_LIBRARY_DIR", "/audio")).resolve()
PULSE_SERVER = os.getenv("PULSE_SERVER", "unix:/tmp/pulse/native")
PULSE_COOKIE = os.getenv("PULSE_COOKIE", "/tmp/pulse/cookie")
SDL_AUDIODRIVER = os.getenv("SDL_AUDIODRIVER", "pulse")
PLAYER_BIN = os.getenv("PLAYER_BIN", "ffplay")
PLAYER_LOGLEVEL = os.getenv("PLAYER_LOGLEVEL", "error")
CHUNK_SIZE = int(os.getenv("PLAYBACK_CHUNK_SIZE", "65536"))

app = FastAPI(title="speaker-player", version="1.0.0")


class PlayStoredRequest(BaseModel):
    filename: str = Field(min_length=1)


class PlayResponse(BaseModel):
    status: str
    source: str
    detail: Optional[str] = None


def _player_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PULSE_SERVER"] = PULSE_SERVER
    env["PULSE_COOKIE"] = PULSE_COOKIE
    env["SDL_AUDIODRIVER"] = SDL_AUDIODRIVER
    return env


def _build_player_args() -> list[str]:
    return [
        PLAYER_BIN,
        "-nodisp",
        "-autoexit",
        "-hide_banner",
        "-nostats",
        "-loglevel",
        PLAYER_LOGLEVEL,
    ]


def _run_player_for_path(path: Path) -> None:
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Audio file not found")

    proc = subprocess.run(
        _build_player_args() + ["-i", str(path)],
        env=_player_env(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or "").strip() or f"Player exited with code {proc.returncode}"
        raise HTTPException(status_code=500, detail=detail)


async def _stream_upload_to_player(file: UploadFile) -> None:
    proc = subprocess.Popen(
        _build_player_args() + ["-i", "pipe:0"],
        env=_player_env(),
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=False,
    )

    assert proc.stdin is not None
    try:
        while True:
            chunk = await file.read(CHUNK_SIZE)
            if not chunk:
                break
            proc.stdin.write(chunk)
            proc.stdin.flush()
    except Exception:
        proc.kill()
        raise
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass

    proc.wait()
    stderr = proc.stderr.read() if proc.stderr else b""
    if proc.returncode != 0:
        detail = (stderr or b"").decode("utf-8", errors="replace").strip() or f"Player exited with code {proc.returncode}"
        raise HTTPException(status_code=500, detail=detail)


def _resolve_stored_audio(filename: str) -> Path:
    candidate = (AUDIO_LIBRARY_DIR / filename).resolve()
    if not candidate.is_relative_to(AUDIO_LIBRARY_DIR):
        raise HTTPException(status_code=400, detail="filename must stay inside the audio library folder")
    return candidate


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {
        "status": "ok",
        "audio_library_dir": str(AUDIO_LIBRARY_DIR),
        "player": PLAYER_BIN,
        "backend": SDL_AUDIODRIVER,
    }


@app.get("/files")
def list_files() -> dict[str, list[str]]:
    if not AUDIO_LIBRARY_DIR.exists():
        return {"files": []}

    files: list[str] = []
    for path in sorted(AUDIO_LIBRARY_DIR.rglob("*")):
        if path.is_file():
            files.append(str(path.relative_to(AUDIO_LIBRARY_DIR)))
    return {"files": files}


@app.post("/play", response_model=PlayResponse)
async def play_upload(file: UploadFile = File(...)) -> PlayResponse:
    if file is None:
        raise HTTPException(status_code=400, detail="file is required")

    await _stream_upload_to_player(file)
    return PlayResponse(status="played", source="upload")


@app.post("/play-stored", response_model=PlayResponse)
def play_stored(body: PlayStoredRequest) -> PlayResponse:
    audio_path = _resolve_stored_audio(body.filename)
    _run_player_for_path(audio_path)
    return PlayResponse(status="played", source=body.filename)
