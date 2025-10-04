"""Minimal CLAP fine-tuning harness focused  on MusicCaps."""
from __future__ import annotations

import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterable, List, Optional, Sequence

import librosa
import mlflow
import numpy as np
import torch
from datasets import load_dataset
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, ClapModel
import yt_dlp

# -----------------------------------------------------------------------------
# Global configuration
# -----------------------------------------------------------------------------

MODEL_NAME: str = "laion/clap-htsat-unfused"
RANDOM_SEED: int = 42
AUDIO_SAMPLING_RATE: int = 48_000
CLIP_SECONDS: int = 10
MODE: str = "train"  # or "inference"
TRAIN_BATCH_SIZE: int = 22
EVAL_BATCH_SIZE: int = 2
LEARNING_RATE: float = 1e-5
WEIGHT_DECAY: float = 1e-4
NUM_EPOCHS: int = 1
VALIDATION_EVERY_STEPS: int = 10
MAX_STEPS: int = 50
GRADIENT_ACCUMULATION_STEPS: int = 1
AUDIO_CACHE_DIR: Path = Path("data/musiccaps/audio")
MUSICCAPS_DATASET: str = "google/musiccaps"
MUSICCAPS_SPLIT: str = "train"
INFERENCE_PROMPT: str = "a melancholic indie ballad with soft vocals and slow tempo"
INFERENCE_TOPK: int = 3
MLFLOW_TRACKING_URI: str = "file:mlruns"
MLFLOW_EXPERIMENT_NAME: str = "musiccaps_clap"
MLFLOW_RUN_NAME: str = "clap_musiccaps_run"

# yt-dlp download behaviour (set these before running main)
YTDLP_DOWNLOAD_ARCHIVE: Optional[Path] = AUDIO_CACHE_DIR.parent / "youtube_download_archive.txt"
YTDLP_COOKIES_FROM_BROWSER: Optional[str] = "chrome"  # e.g. "chrome", "firefox"; leave None to skip
YTDLP_COOKIES_FILE: Optional[Path] = None  # path to exported cookies.txt if not using browser picker
YTDLP_SLEEP_INTERVAL: Optional[float] = 5.0  # seconds; set None to disable throttling
YTDLP_MAX_SLEEP_INTERVAL: Optional[float] = 10.0
YTDLP_MAX_RETRIES: int = 10
YTDLP_FRAGMENT_RETRIES: int = 10
YTDLP_MAX_WORKERS: int = 4

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -----------------------------------------------------------------------------
# MusicCaps preparation
# -----------------------------------------------------------------------------


@dataclass
class MusicCapsSample:
    ytid: str
    start_s: float
    text: str
    audio_path: Path


def load_musiccaps_metadata(sample_limit: Optional[int] = None) -> List[Dict[str, str]]:
    dataset = load_dataset(MUSICCAPS_DATASET, split=MUSICCAPS_SPLIT)
    if sample_limit is None or sample_limit >= len(dataset):
        return list(dataset)
    return [dataset[i] for i in range(sample_limit)]


def download_audio(ytid: str, start_s: float) -> Path:
    AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    clip_path = AUDIO_CACHE_DIR / f"{ytid}_{int(start_s)}.wav"
    if clip_path.exists():
        return clip_path
    temp_template = str(AUDIO_CACHE_DIR / f"{ytid}.%(ext)s")
    ydl_opts = {
        "outtmpl": temp_template,
        "format": "bestaudio/best",
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": False,
        "retries": YTDLP_MAX_RETRIES,
        "fragment_retries": YTDLP_FRAGMENT_RETRIES,
        "retry_sleep_functions": {"http": lambda n: 2 ** (n - 1)},  # экспоненциальное ожидание
    }
    if YTDLP_SLEEP_INTERVAL is not None:
        ydl_opts["sleep_interval"] = YTDLP_SLEEP_INTERVAL
        if YTDLP_MAX_SLEEP_INTERVAL is not None:
            ydl_opts["max_sleep_interval"] = max(YTDLP_SLEEP_INTERVAL, YTDLP_MAX_SLEEP_INTERVAL)
    if YTDLP_DOWNLOAD_ARCHIVE is not None:
        YTDLP_DOWNLOAD_ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
        ydl_opts["download_archive"] = str(YTDLP_DOWNLOAD_ARCHIVE)
    if YTDLP_COOKIES_FROM_BROWSER:
        ydl_opts["cookiesfrombrowser"] = (YTDLP_COOKIES_FROM_BROWSER, None, None, None)
    elif YTDLP_COOKIES_FILE:
        YTDLP_COOKIES_FILE.parent.mkdir(parents=True, exist_ok=True)
        ydl_opts["cookiefile"] = str(YTDLP_COOKIES_FILE)
    url = f"https://www.youtube.com/watch?v={ytid}"
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(url, download=True)
        downloaded_path = Path(ydl.prepare_filename(result))
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            str(start_s),
            "-t",
            str(CLIP_SECONDS),
            "-i",
            str(downloaded_path),
            "-ar",
            str(AUDIO_SAMPLING_RATE),
            "-ac",
            "1",
            str(clip_path),
        ],
        check=True,
    )
    downloaded_path.unlink()
    return clip_path


def prepare_musiccaps_samples(sample_limit: Optional[int] = None) -> List[MusicCapsSample]:
    metadata = load_musiccaps_metadata(sample_limit)
    ordered_rows: List[Dict[str, str]] = list(metadata)
    order_index = {
        (row["ytid"], float(row["start_s"])): idx for idx, row in enumerate(ordered_rows)
    }

    existing_samples: List[MusicCapsSample] = []
    rows_to_fetch: List[Dict[str, str]] = []
    for row in ordered_rows:
        clip_path = AUDIO_CACHE_DIR / f"{row['ytid']}_{int(row['start_s'])}.wav"
        if clip_path.exists():
            existing_samples.append(
                MusicCapsSample(
                    ytid=row["ytid"],
                    start_s=row["start_s"],
                    text=row["caption"],
                    audio_path=clip_path,
                )
            )
        else:
            rows_to_fetch.append(row)

    def fetch(row: Dict[str, str]) -> Optional[MusicCapsSample]:
        path = download_audio(row["ytid"], row["start_s"])
        return MusicCapsSample(
            ytid=row["ytid"],
            start_s=row["start_s"],
            text=row["caption"],
            audio_path=path,
        )

    fetched_samples: List[MusicCapsSample] = []
    if rows_to_fetch:
        worker_count = max(1, YTDLP_MAX_WORKERS)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {executor.submit(fetch, row): row for row in rows_to_fetch}
            for future in as_completed(future_map):
                result = future.result()
                if result is not None:
                    fetched_samples.append(result)

    all_samples = existing_samples + fetched_samples
    all_samples.sort(key=lambda sample: order_index[(sample.ytid, float(sample.start_s))])
    return all_samples


# -----------------------------------------------------------------------------
# Torch dataset
# -----------------------------------------------------------------------------


class MusicCapsDataset(Dataset):
    def __init__(self, samples: Sequence[MusicCapsSample]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        audio, _ = librosa.load(
            sample.audio_path,
            sr=AUDIO_SAMPLING_RATE,
            mono=True,
            offset=0.0,
            duration=CLIP_SECONDS,
        )
        target_length = AUDIO_SAMPLING_RATE * CLIP_SECONDS
        if audio.shape[0] < target_length:
            pad_width = target_length - audio.shape[0]
            audio = np.pad(audio, (0, pad_width))
        if audio.shape[0] > target_length:
            audio = audio[:target_length]
        tensor = torch.from_numpy(audio.astype(np.float32))
        return {"audio": tensor, "text": sample.text}


# -----------------------------------------------------------------------------
# Training and evaluation
# -----------------------------------------------------------------------------


def collate_fn(batch: Sequence[Dict[str, torch.Tensor]], processor: AutoProcessor) -> Dict[str, torch.Tensor]:
    audios = [item["audio"].numpy() for item in batch]
    texts = [item["text"] for item in batch]
    inputs = processor(
        text=texts,
        audios=audios,
        sampling_rate=AUDIO_SAMPLING_RATE,
        return_tensors="pt",
        padding=True,
    )
    return inputs


def compute_accuracy(logits: torch.Tensor) -> float:
    predictions = logits.argmax(dim=-1)
    targets = torch.arange(predictions.shape[0], device=predictions.device)
    correct = (predictions == targets).float().mean().item()
    return correct


@dataclass
class Metrics:
    loss: float
    accuracy: float


def evaluate(model: ClapModel, loader: DataLoader, device: torch.device) -> Metrics:
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    steps = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch, return_loss=True)
            total_loss += outputs.loss.item()
            total_accuracy += compute_accuracy(outputs.logits_per_text)
            steps += 1
    return Metrics(loss=total_loss / steps, accuracy=total_accuracy / steps)


def train(model: ClapModel, processor: AutoProcessor, train_loader: DataLoader, val_loader: DataLoader, device: torch.device) -> None:
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    model.to(device)
    global_step = 0
    for epoch in range(NUM_EPOCHS):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch, return_loss=True)
            loss_value = outputs.loss.item()
            mlflow.log_metric("train_loss", loss_value, step=global_step + 1)
            loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            if (global_step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
            global_step += 1
            if global_step % VALIDATION_EVERY_STEPS == 0:
                metrics = evaluate(model, val_loader, device)
                print(f"step={global_step} loss={metrics.loss:.4f} acc={metrics.accuracy:.4f}")
                mlflow.log_metric("val_loss", metrics.loss, step=global_step)
                mlflow.log_metric("val_accuracy", metrics.accuracy, step=global_step)
            if global_step >= MAX_STEPS:
                return


# -----------------------------------------------------------------------------
# Workflow
# -----------------------------------------------------------------------------


def build_loaders(processor: AutoProcessor, sample_limit: Optional[int] = None) -> Dict[str, DataLoader]:
    samples = prepare_musiccaps_samples(sample_limit)
    pivot = max(1, int(len(samples) * 0.9))
    train_dataset = MusicCapsDataset(samples[:pivot])
    val_dataset = MusicCapsDataset(samples[pivot:])
    collate = lambda batch: collate_fn(batch, processor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate,
    )
    return {"train": train_loader, "val": val_loader, "samples": samples}


if __name__ == "__main__":
    set_seed(RANDOM_SEED)
    device = get_device()
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = ClapModel.from_pretrained(MODEL_NAME).to(device)

    if MODE == "train":
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        run = mlflow.start_run(run_name=MLFLOW_RUN_NAME)
        tracking_uri = mlflow.get_tracking_uri()
        if tracking_uri.startswith("file:"):
            run_base = Path(tracking_uri[5:]).resolve()
            run_url = f"file://{run_base}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}"
        else:
            run_url = f"{tracking_uri.rstrip('/')}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}"
        print(f"MLflow run started: {run_url}")
        mlflow.log_params(
            {
                "model_name": MODEL_NAME,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "batch_size": TRAIN_BATCH_SIZE,
                "max_steps": MAX_STEPS,
                "clip_seconds": CLIP_SECONDS,
                "sampling_rate": AUDIO_SAMPLING_RATE,
            }
        )
        try:
            loaders = build_loaders(processor)
            train(model, processor, loaders["train"], loaders["val"], device)
            metrics = evaluate(model, loaders["val"], device)
            print(f"final loss={metrics.loss:.4f} acc={metrics.accuracy:.4f}")
            mlflow.log_metric("final_val_loss", metrics.loss)
            mlflow.log_metric("final_val_accuracy", metrics.accuracy)
        finally:
            mlflow.end_run()
    elif MODE == "inference":
        samples = prepare_musiccaps_samples(sample_limit=INFERENCE_TOPK)
        audios: List[np.ndarray] = []
        for sample in samples:
            audio, _ = librosa.load(
                sample.audio_path,
                sr=AUDIO_SAMPLING_RATE,
                mono=True,
                offset=0.0,
                duration=CLIP_SECONDS,
            )
            target_length = AUDIO_SAMPLING_RATE * CLIP_SECONDS
            if audio.shape[0] < target_length:
                audio = np.pad(audio, (0, target_length - audio.shape[0]))
            if audio.shape[0] > target_length:
                audio = audio[:target_length]
            audios.append(audio.astype(np.float32))
        batch = processor(
            text=[INFERENCE_PROMPT],
            audios=audios,
            sampling_rate=AUDIO_SAMPLING_RATE,
            return_tensors="pt",
            padding=True,
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch, return_loss=False)
        probs = outputs.logits_per_text[0].softmax(dim=-1)
        ranking = sorted(
            zip(probs.tolist(), samples),
            key=lambda item: item[0],
            reverse=True,
        )
        print(f"Prompt: {INFERENCE_PROMPT}")
        for idx, (score, sample) in enumerate(ranking, start=1):
            snippet = sample.text.replace("\n", " ")
            if len(snippet) > 80:
                snippet = snippet[:77] + "..."
            print(f"{idx}. score={score:.4f} | ytid={sample.ytid} | start={sample.start_s}s | caption={snippet}")
    else:
        raise ValueError("MODE must be either 'train' or 'inference'")
