import sqlite3
from pathlib import Path
from typing import Dict, List

import numpy as np

from vector_encoders import (
    encode_audio_clap,
    encode_fusion_query,
    encode_fusion_tracks,
    encode_text_biencoder,
    encode_text_clap,
)


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
            SELECT song_id, title, artist, lyrics, audio_path, cover_path, lyrics_vector, audio_vector
            FROM songs
            """
        ).fetchall()
        if not rows:
            raise RuntimeError("songs table is empty")
        for row in rows:
            lyric_vec = np.frombuffer(row["lyrics_vector"], dtype=np.float32)
            audio_vec = np.frombuffer(row["audio_vector"], dtype=np.float32)
            record: Dict[str, object] = {
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
        audio_matrix = np.stack([r["audio_vector"] for r in self.records])
        lyric_matrix = np.stack([r["lyrics_vector"] for r in self.records])
        fusion_matrix = encode_fusion_tracks(audio_matrix, lyric_matrix)
        fusion_matrix /= np.linalg.norm(fusion_matrix, axis=1, keepdims=True) + 1e-8
        self._fusion_matrix = fusion_matrix.astype(np.float32)

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

    def search(self, query: str, top_k: int = 20) -> List[Dict]:
        top_k = max(1, min(top_k, len(self.records)))
        query_vec = encode_fusion_query([query])[0]
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        scores = self._fusion_matrix @ query_vec.astype(np.float32)
        top_idx = np.argpartition(-scores, range(top_k))[:top_k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return self._format_results(top_idx, scores[top_idx])

    def search_joint(self, query: str, top_k: int) -> List[Dict]:
        """Совместимый вызов старого API; теперь использует Fusion."""
        return self.search(query, top_k)

    def recompute_vectors(self) -> None:
        """Перекодировать все векторы в базе заново."""
        cur = self.conn.cursor()

        for record in self.records:
            text_vec = encode_text_biencoder([record["lyrics"]])[0]
            audio_vec = encode_audio_clap(Path(record["audio_path"]))

            record["lyrics_vector"] = text_vec
            record["audio_vector"] = audio_vec

            cur.execute(
                """
                UPDATE songs
                SET lyrics_vector = ?, audio_vector = ?
                WHERE song_id = ?
                """,
                (text_vec.tobytes(), audio_vec.tobytes(), record["song_id"]),
            )

        self.conn.commit()
        self._build_indexes()
