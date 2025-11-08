"""FastAPI service for querying LyricCovers vectors."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from vector_db import SongVectorDB

DB_PATH = Path("lyriccovers_output/songs.db")

try:
    VECTOR_DB = SongVectorDB(DB_PATH)
except FileNotFoundError:
    raise RuntimeError(f"SQLite database not found at {DB_PATH}. Run lyriccovers_sampler.py first.") from None

app = FastAPI(title="LyricCovers Vector API", version="1.0")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(20, ge=1, le=100)


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok", "songs": len(VECTOR_DB.records)}


@app.post("/search/lyrics")
def search_lyrics(request: SearchRequest):
    try:
        return VECTOR_DB.search_lyrics(request.query, top_k=request.top_k)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/search/audio")
def search_audio(request: SearchRequest):
    try:
        return VECTOR_DB.search_audio(request.query, top_k=request.top_k)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
