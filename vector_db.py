"""FAISS-backed vector database for LyricCovers songs."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, List

import faiss  # type: ignore
import numpy as np

from vector_encoders import encode_lyrics_biencoder, encode_text_clap


class SongVectorDB:
    """Loads embeddings from SQLite and exposes search over lyrics/audio."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(self.db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.records: List[Dict] = []
        self._load_records()
        self._build_indexes()

    def _load_records(self) -> None:
        cur = self.conn.cursor()
        rows = cur.execute(
            """
            SELECT song_id, title, artist, lyrics, audio_path, cover_path,
                   lyrics_vector, audio_vector
            FROM songs
            """
        ).fetchall()
        if not rows:
            raise RuntimeError("songs table is empty")
        for row in rows:
            lyric_vec = np.frombuffer(row["lyrics_vector"], dtype=np.float32)
            audio_vec = np.frombuffer(row["audio_vector"], dtype=np.float32)
            record = {
                "song_id": row["song_id"],
                "title": row["title"],
                "artist": row["artist"],
                "lyrics": row["lyrics"],
                "audio_path": row["audio_path"],
                "cover_path": row["cover_path"],
                "lyrics_vector": lyric_vec,
                "audio_vector": audio_vec,
            }
            self.records.append(record)

    def _build_indexes(self) -> None:
        lyric_matrix = np.stack([r["lyrics_vector"] for r in self.records])
        audio_matrix = np.stack([r["audio_vector"] for r in self.records])
        faiss.normalize_L2(lyric_matrix)
        faiss.normalize_L2(audio_matrix)
        lyric_dim = lyric_matrix.shape[1]
        audio_dim = audio_matrix.shape[1]
        self._lyric_index = faiss.IndexFlatIP(lyric_dim)
        self._audio_index = faiss.IndexFlatIP(audio_dim)
        self._lyric_index.add(lyric_matrix.astype(np.float32))
        self._audio_index.add(audio_matrix.astype(np.float32))

    def _format_results(self, indices: np.ndarray, scores: np.ndarray) -> List[Dict]:
        results: List[Dict] = []
        for idx, score in zip(indices, scores):
            record = self.records[int(idx)]
            results.append(
                {
                    "song_id": record["song_id"],
                    "title": record["title"],
                    "artist": record["artist"],
                    "audio_path": record["audio_path"],
                    "cover_path": record["cover_path"],
                    "score": float(score),
                }
            )
        return results

    def search_lyrics(self, query: str, top_k: int = 20) -> List[Dict]:
        top_k = max(1, min(top_k, len(self.records)))
        query_vec = encode_lyrics_biencoder([query])[0].reshape(1, -1)
        faiss.normalize_L2(query_vec)
        scores, indices = self._lyric_index.search(query_vec.astype(np.float32), top_k)
        return self._format_results(indices[0], scores[0])

    def search_audio(self, text_query: str, top_k: int = 20) -> List[Dict]:
        top_k = max(1, min(top_k, len(self.records)))
        query_vec = encode_text_clap([text_query])[0].reshape(1, -1)
        faiss.normalize_L2(query_vec)
        scores, indices = self._audio_index.search(query_vec.astype(np.float32), top_k)
        return self._format_results(indices[0], scores[0])
