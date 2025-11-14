import base64
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from vector_db import SongVectorDB

DB_PATH = Path("lyriccovers_output/songs.db")
TOP_K_DEFAULT = 20

try:
    VECTOR_DB = SongVectorDB(DB_PATH)
except FileNotFoundError:
    raise RuntimeError(
        f"SQLite database not found at {DB_PATH}. Create and populate it before starting the API."
    ) from None

app = FastAPI(title="LyricCovers Vector API", version="1.0")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(TOP_K_DEFAULT, ge=1)


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.post("/search/lyrics")
def search_lyrics(request: SearchRequest):
    try:
        return VECTOR_DB.search_lyrics(request.query, top_k=request.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/search/audio")
def search_audio(request: SearchRequest):
    try:
        return VECTOR_DB.search_audio(request.query, top_k=request.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _song_stream(songs):
    for song in songs:
        audio_bytes = Path(song["audio_path"]).read_bytes()
        cover_bytes = Path(song["cover_path"]).read_bytes()
        payload = {
            "title": song["title"],
            "artist": song["artist"],
            "audio": base64.b64encode(audio_bytes).decode("ascii"),
            "cover": base64.b64encode(cover_bytes).decode("ascii"),
        }
        yield json.dumps(payload).encode("utf-8") + b"\n"


@app.post("/search/combined")
def search_combined(request: SearchRequest):
    try:
        songs = VECTOR_DB.search_joint(request.query, top_k=request.top_k)
        return StreamingResponse(_song_stream(songs), media_type="application/x-ndjson")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
