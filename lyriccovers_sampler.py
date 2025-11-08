#!/usr/bin/env python3
"""
Sample 1,000 diverse entries from the LyricCovers2.0 dataset, download their
audio and cover images, and materialize the result into a SQLite database.
"""

from __future__ import annotations

import argparse
import logging
import re
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from vector_encoders import encode_audio_clap, encode_lyrics_biencoder

REPO_URL = "https://github.com/Maxl94/LyricCovers2.0.git"
DEFAULT_REPO_DIR = Path("LyricCovers2.0")
OUTPUT_ROOT = Path("lyriccovers_output")
AUDIO_DIR = OUTPUT_ROOT / "audio"
COVER_DIR = OUTPUT_ROOT / "covers"
DB_PATH = OUTPUT_ROOT / "songs.db"
SAMPLE_SIZE = 1000
MAX_WORKERS = 4
HTTP_TIMEOUT = 20
YTDLP_TIMEOUT = 240
GENIUS_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}
LYRICS_CACHE: Dict[str, str] = {}


@dataclass
class SongRecord:
    """Holds the finalized metadata for a downloaded song."""

    song_id: str
    title: str
    artist: str
    lyrics: str
    original_id: int
    language: str
    release_year: Optional[int]
    youtube_url: str
    audio_path: Path
    cover_path: Path


def cleanup_assets(record: SongRecord) -> None:
    """Remove downloaded assets when a later stage fails."""
    try:
        record.audio_path.unlink(missing_ok=True)
    except Exception:
        pass
    try:
        record.cover_path.unlink(missing_ok=True)
    except Exception:
        pass


def run_command(cmd: List[str], *, cwd: Optional[Path] = None, timeout: Optional[int] = None) -> bool:
    """Execute a command and return True on success."""
    try:
        subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            text=True,
        )
        return True
    except subprocess.CalledProcessError as exc:
        # Suppress verbose error logging for yt-dlp
        if "yt-dlp" not in " ".join(cmd):
            stderr = (exc.stderr or "").strip()
            logging.warning(
                "Command failed (%s): %s\n%s",
                exc.returncode,
                " ".join(cmd),
                stderr[-400:],
            )
    except subprocess.TimeoutExpired:
        if "yt-dlp" not in " ".join(cmd):
            logging.warning("Command timed out: %s", " ".join(cmd))
    return False


def ensure_repo(repo_dir: Path = DEFAULT_REPO_DIR, repo_url: str = REPO_URL) -> None:
    """Clone the dataset repository if it is missing."""
    if repo_dir.exists():
        return
    logging.info("Cloning LyricCovers2.0 into %s", repo_dir)
    run_command(["git", "clone", repo_url, str(repo_dir)])


def pick_dataset_file(data_dir: Path) -> Path:
    """Return the preferred dataset file inside data/."""
    if not data_dir.exists():
        raise FileNotFoundError(f"{data_dir} does not exist")
    candidates: List[Path] = [p for p in data_dir.rglob("*") if p.is_file()]
    priority = [".parquet", ".pqt", ".pkl", ".pickle", ".csv"]
    for ext in priority:
        choices = [p for p in candidates if p.suffix.lower() == ext]
        if choices:
            choices.sort(key=lambda p: p.stat().st_size, reverse=True)
            return choices[0]
    raise RuntimeError("No supported dataset file found under data/")


def load_dataframe(dataset_path: Path) -> pd.DataFrame:
    """Load the dataset into memory."""
    suffix = dataset_path.suffix.lower()
    if suffix in {".parquet", ".pqt"}:
        df = pd.read_parquet(dataset_path)
    elif suffix in {".pkl", ".pickle"}:
        df = pd.read_pickle(dataset_path)
    elif suffix == ".csv":
        df = pd.read_csv(dataset_path)
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_path}")
    logging.info("Dataset loaded: %s rows x %s columns", df.shape[0], df.shape[1])
    return df


def normalize_language(value: Optional[str]) -> str:
    if not isinstance(value, str):
        return "unknown"
    norm = value.strip().lower()
    return norm if norm else "unknown"


def normalize_release_year(value) -> Optional[int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return int(value.year)
    if isinstance(value, (int, float)) and not pd.isna(value):
        return int(value)
    if isinstance(value, str) and value.strip():
        try:
            return int(float(value.strip()))
        except ValueError:
            return None
    return None


def release_decade(year: Optional[float]) -> str:
    if pd.isna(year):
        return "unknown"
    try:
        year_int = int(year)
    except (TypeError, ValueError):
        return "unknown"
    decade = (year_int // 10) * 10
    return f"{decade}s"


def extract_primary_artist(raw: Optional[str]) -> str:
    if not isinstance(raw, str):
        return "unknown"
    cleaned = raw.strip()
    if not cleaned:
        return "unknown"
    tokens = re.split(r"\s*(?:,|;|&|feat\.?|featuring|with|x|\+)\s*", cleaned, maxsplit=1, flags=re.IGNORECASE)
    primary = tokens[0].strip()
    return primary if primary else "unknown"


def extract_youtube_id(url: Optional[str]) -> Optional[str]:
    if not isinstance(url, str):
        return None
    patterns = [
        r"(?:v=|\/)([A-Za-z0-9_-]{6,})",
        r"youtu\.be\/([A-Za-z0-9_-]{6,})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def fetch_lyrics(url: Optional[str]) -> Optional[str]:
    """Scrape lyrics from Genius (or compatible) pages."""
    if not isinstance(url, str) or not url.strip():
        return None
    if url in LYRICS_CACHE:
        return LYRICS_CACHE[url]
    try:
        logging.info("📝 Fetching lyrics from Genius...")
        resp = requests.get(url, headers=GENIUS_HEADERS, timeout=HTTP_TIMEOUT)
    except requests.RequestException:
        return None
    if resp.status_code != 200:
        return None
    soup = BeautifulSoup(resp.text, "html.parser")
    containers = soup.select("div[data-lyrics-container='true']")
    if not containers:
        return None
    text_parts: List[str] = []
    for div in containers:
        chunk = div.get_text(separator="\n").strip()
        if chunk:
            text_parts.append(chunk)
    lyrics = "\n\n".join(text_parts).strip()
    if lyrics:
        LYRICS_CACHE[url] = lyrics
    return lyrics or None


def ensure_mp3(path: Path) -> bool:
    """Re-encode the file to MP3 if necessary."""
    if path.suffix.lower() == ".mp3":
        return True
    mp3_path = path.with_suffix(".mp3")
    cmd = ["ffmpeg", "-y", "-i", str(path), str(mp3_path)]
    ok = run_command(cmd)
    if not ok or not mp3_path.exists():
        return False
    try:
        path.unlink(missing_ok=True)
    except FileNotFoundError:
        pass
    mp3_path.rename(path)
    return True


def deduplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure one row per original_id with a valid YouTube URL."""
    filtered = df[df["youtube_url"].astype(str).str.strip().ne("")]
    filtered = filtered.dropna(subset=["original_id"])
    metadata_cols = [
        "language",
        "lyrics_state",
        "spotify_url",
        "soundcloud_url",
        "release_date",
        "release_year",
    ]
    filtered = filtered.copy()
    filtered["metadata_score"] = filtered[metadata_cols].notna().sum(axis=1)
    filtered = filtered.sort_values(["original_id", "metadata_score"], ascending=[True, False])
    deduped = filtered.drop_duplicates(subset=["original_id"], keep="first")
    logging.info("Deduplicated to %s unique original_id rows", deduped.shape[0])
    return deduped


def prioritize_candidates(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Sort rows to surface diverse entries first."""
    df = df.copy()
    df["language_norm"] = df["language"].map(normalize_language)
    df["release_year_num"] = pd.to_numeric(df["release_year"], errors="coerce")
    df["decade"] = df["release_year_num"].map(release_decade)
    df["cover_tag"] = df["is_cover"].map(lambda x: "cover" if bool(x) else "original")
    df["youtube_type_norm"] = df["youtube_type"].fillna("unknown")
    df["diversity_key"] = list(
        zip(df["language_norm"], df["cover_tag"], df["decade"], df["youtube_type_norm"])
    )
    rarity = df.groupby("diversity_key")["original_id"].transform("count")
    df["rarity_score"] = rarity
    df["metadata_score"] = df["metadata_score"].fillna(0)
    df = df.sample(frac=1.0, random_state=seed)
    df = df.sort_values(
        ["rarity_score", "metadata_score", "language_norm", "release_year_num"],
        ascending=[True, False, True, True],
    )
    return df


def download_cover(video_id: str, dest: Path) -> bool:
    """Download a YouTube thumbnail as the cover art."""
    candidates = [
        f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg",
        f"https://img.youtube.com/vi/{video_id}/sddefault.jpg",
        f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
        f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg",
        f"https://img.youtube.com/vi/{video_id}/default.jpg",
    ]
    dest.parent.mkdir(parents=True, exist_ok=True)
    for url in candidates:
        try:
            resp = requests.get(url, timeout=HTTP_TIMEOUT)
        except requests.RequestException:
            continue
        if resp.status_code == 200 and resp.content:
            dest.write_bytes(resp.content)
            return True
    logging.debug("Failed to download cover for %s", video_id)
    return False


def download_audio(youtube_url: str, dest: Path) -> bool:
    """Download audio via yt-dlp, storing it as MP3."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_template = dest.with_suffix("")
    cmd = [
        "yt-dlp",
        "-f",
        "bestaudio/best",
        "--extract-audio",
        "--audio-format",
        "mp3",
        "--audio-quality",
        "0",
        "--no-keep-video",
        "--no-progress",
        "-o",
        f"{tmp_template}.%(ext)s",
        youtube_url,
    ]
    # Log that we're starting YouTube download
    logging.info("📥 Downloading audio from YouTube...")
    ok = run_command(cmd, timeout=YTDLP_TIMEOUT)
    if not ok:
        logging.info("✗ YouTube download failed")
        return False
    if dest.exists():
        result = ensure_mp3(dest)
    else:
        result = False
        for file in dest.parent.glob(dest.stem + ".*"):
            if file == dest:
                continue
            file.rename(dest)
            result = ensure_mp3(dest)
            break
    # Clean up any residual files with the same stem (e.g., stray .webm)
    for extra in dest.parent.glob(dest.stem + ".*"):
        if extra != dest and extra.exists():
            extra.unlink(missing_ok=True)
    return result


def process_row(row: Dict[str, object], attempt_num: int = 0) -> Optional[SongRecord]:
    song_id = str(row.get("id") or row["original_id"])
    title = str(row.get("title") or "").strip() or "unknown"
    artist = extract_primary_artist(row.get("artist"))
    original_id = int(row["original_id"])
    youtube_url = str(row["youtube_url"])
    video_id = extract_youtube_id(youtube_url)

    if not video_id:
        logging.info("✗ No YouTube video ID: %s - %s (attempt #%d)", artist, title, attempt_num)
        return None

    logging.info("▶ Trying: %s - %s (attempt #%d)", artist, title, attempt_num)

    lyrics = row.get("lyrics")
    if not isinstance(lyrics, str) or not lyrics.strip():
        lyrics = fetch_lyrics(row.get("url"))
    if not lyrics:
        logging.info("✗ No lyrics found: %s - %s (attempt #%d)", artist, title, attempt_num)
        return None

    audio_dest = AUDIO_DIR / f"{song_id}.mp3"
    cover_dest = COVER_DIR / f"{song_id}.jpg"
    if not audio_dest.exists():
        if not download_audio(youtube_url, audio_dest):
            logging.info("✗ Audio download failed: %s - %s (attempt #%d)", artist, title, attempt_num)
            return None
    if not cover_dest.exists():
        if not download_cover(video_id, cover_dest):
            audio_dest.unlink(missing_ok=True)
            logging.info("✗ Cover download failed: %s - %s (attempt #%d)", artist, title, attempt_num)
            return None

    record = SongRecord(
        song_id=song_id,
        title=title,
        artist=artist,
        lyrics=lyrics,
        original_id=original_id,
        language=normalize_language(row.get("language")),
        release_year=normalize_release_year(row.get("release_year")),
        youtube_url=youtube_url,
        audio_path=audio_dest,
        cover_path=cover_dest,
    )
    return record


def has_complete_metadata(row) -> bool:
    """Check if a row has all required metadata for a complete download."""
    required_fields = ["original_id", "title", "artist", "youtube_url"]

    # Check required fields exist and are not empty
    for field in required_fields:
        value = row.get(field) if isinstance(row, dict) else row[field]
        if pd.isna(value) or not str(value).strip():
            return False

    # Check YouTube URL is valid and has video ID
    youtube_url = str(row.get("youtube_url") if isinstance(row, dict) else row["youtube_url"])
    if not extract_youtube_id(youtube_url):
        return False

    # Check if lyrics are available either directly or via URL
    lyrics_value = row.get("lyrics") if isinstance(row, dict) else row.get("lyrics", None)
    has_lyrics = isinstance(lyrics_value, str) and bool(lyrics_value.strip())
    genius_url = row.get("url") if isinstance(row, dict) else row.get("url", None)
    has_genius_url = isinstance(genius_url, str) and bool(genius_url.strip())

    return has_lyrics or has_genius_url


def process_samples_sequential(
    rows: pd.DataFrame,
    target_size: int,
    db_conn: sqlite3.Connection,
) -> int:
    """Process songs strictly one at a time, ensuring DB insert before continuing."""
    successes = 0
    attempts = 0
    row_dicts = rows.to_dict("records")

    for row_dict in row_dicts:
        if successes >= target_size:
            break
        attempts += 1

        if not has_complete_metadata(row_dict):
            logging.info(
                "⚠ Skipping incomplete metadata: %s (attempt #%d)",
                row_dict.get("title", "unknown"),
                attempts,
            )
            continue

        record = process_row(row_dict, attempts)
        if not record:
            continue

        try:
            lyric_vector = encode_lyrics_biencoder([record.lyrics])[0]
            audio_vector = encode_audio_clap([record.audio_path])[0]
            add_song_to_database(db_conn, record, lyric_vector, audio_vector)
            successes += 1
            logging.info(
                "✓ Downloaded & Added to DB: %s - %s (%d/%d)",
                record.artist,
                record.title,
                successes,
                target_size,
            )
        except Exception as exc:  # noqa: BLE001
            logging.error(
                "✗ Failed after download (cleanup triggered): %s - %s - %s",
                record.artist,
                record.title,
                exc,
            )
            cleanup_assets(record)

    if successes < target_size:
        logging.warning("Only %d/%d songs were successfully processed", successes, target_size)
    else:
        logging.info("Successfully processed %d songs sequentially", successes)
    return successes


def initialize_database(db_path: Path) -> sqlite3.Connection:
    """Create and initialize the SQLite database."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS songs (
            song_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            artist TEXT,
            lyrics TEXT,
            original_id INTEGER,
            language TEXT,
            release_year INTEGER,
            youtube_url TEXT,
            audio_path TEXT NOT NULL,
            cover_path TEXT NOT NULL,
            lyrics_vector BLOB,
            audio_vector BLOB
        )
        """
    )
    cur.execute("DELETE FROM songs")
    conn.commit()
    return conn


def add_song_to_database(
    conn: sqlite3.Connection,
    record: SongRecord,
    lyric_vector: np.ndarray,
    audio_vector: np.ndarray,
) -> None:
    """Add a single song with its vectors to the database."""
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO songs (
            song_id, title, artist, lyrics, original_id, language, release_year,
            youtube_url, audio_path, cover_path, lyrics_vector, audio_vector
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.song_id,
            record.title,
            record.artist,
            record.lyrics,
            record.original_id,
            record.language,
            record.release_year,
            record.youtube_url,
            str(record.audio_path),
            str(record.cover_path),
            sqlite3.Binary(np.asarray(lyric_vector, dtype=np.float32).tobytes()),
            sqlite3.Binary(np.asarray(audio_vector, dtype=np.float32).tobytes()),
        ),
    )
    conn.commit()


def persist_to_sqlite(
    records: List[SongRecord],
    lyric_vectors: np.ndarray,
    audio_vectors: np.ndarray,
    db_path: Path,
) -> None:
    """Legacy function - kept for compatibility but should not be used."""
    conn = initialize_database(db_path)
    for rec, lyric_vec, audio_vec in zip(records, lyric_vectors, audio_vectors):
        add_song_to_database(conn, rec, lyric_vec, audio_vec)
    conn.close()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-url", default=REPO_URL)
    parser.add_argument("--repo-dir", default=str(DEFAULT_REPO_DIR))
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)




def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    output_root = Path(args.output_root)
    global OUTPUT_ROOT, AUDIO_DIR, COVER_DIR, DB_PATH  # noqa: PLW0603
    OUTPUT_ROOT = output_root
    AUDIO_DIR = OUTPUT_ROOT / "audio"
    COVER_DIR = OUTPUT_ROOT / "covers"
    DB_PATH = OUTPUT_ROOT / "songs.db"

    # Create directories
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    COVER_DIR.mkdir(parents=True, exist_ok=True)

    # Load and prepare dataset
    ensure_repo()
    dataset_file = pick_dataset_file(DEFAULT_REPO_DIR / "data")
    df = load_dataframe(dataset_file)
    df = deduplicate_rows(df)

    # Filter to ONLY rows with complete metadata BEFORE sampling
    logging.info("Filtering dataset for complete metadata...")
    complete_mask = df.apply(has_complete_metadata, axis=1)
    complete_df = df[complete_mask].copy()

    logging.info("Found %d rows with complete metadata out of %d total rows",
                len(complete_df), len(df))

    if len(complete_df) < args.sample_size:
        logging.error(
            "Only %d rows have complete metadata (need %d)",
            len(complete_df),
            args.sample_size,
        )
        return 1

    # Shuffle the full set so we can process sequentially until we reach the target size
    candidate_df = complete_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    logging.info(
        "Prepared %d candidate songs with complete metadata; need %d successes",
        len(candidate_df),
        args.sample_size,
    )

    logging.info("Starting download of %d songs", args.sample_size)
    start = time.time()

    # Initialize database immediately - create it empty
    logging.info("Initializing SQLite database at %s", DB_PATH)
    db_conn = initialize_database(DB_PATH)

    # Download songs sequentially, inserting into DB only after full success
    logging.info("Starting strict sequential download and database population")
    processed = process_samples_sequential(candidate_df, args.sample_size, db_conn)

    # Close database connection
    db_conn.close()

    elapsed = time.time() - start
    logging.info(
        "Completed download cycle (%d/%d songs) in %.1f minutes. SQLite database at %s",
        processed,
        args.sample_size,
        elapsed / 60,
        DB_PATH,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
