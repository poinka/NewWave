"""End-to-end CLAP training & evaluation harness for music retrieval.

This module orchestrates dataset ingestion, preprocessing, model fine-tuning,
validation, checkpointing, MLflow logging, and a simple inference example for
contrastive audio-text pretraining with CLAP.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from datasets import Audio, Dataset, DatasetDict, concatenate_datasets, load_dataset
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoProcessor, ClapModel

# ---------------------------------------------------------------------------
# Global configuration and hyperparameters
# ---------------------------------------------------------------------------

RANDOM_SEED: int = 42
MODEL_NAME: str = "laion/clap-htsat-unfused"
CONTEXT_LENGTH: int = 77
AUDIO_SAMPLING_RATE: int = 48_000
AUDIO_CHANNELS: int = 1
SYNTHETIC_AUDIO_SECONDS: float = 10.0
MAX_AUDIO_DURATION_SECONDS: float = 30.0
GRADIENT_ACCUMULATION_STEPS: int = 1
LEARNING_RATE: float = 1e-5
WEIGHT_DECAY: float = 1e-4
BETAS: Tuple[float, float] = (0.9, 0.98)
EPSILON: float = 1e-6
MAX_GRAD_NORM: float = 1.0
TRAIN_BATCH_SIZE: int = 2
EVAL_BATCH_SIZE: int = 4
NUM_EPOCHS: int = 1
MAX_TRAIN_STEPS: int = 50
VALIDATION_INTERVAL_STEPS: int = 10
CHECKPOINT_DIR: Path = Path("checkpoints")
CHECKPOINT_PREFIX: str = "clap"
MLFLOW_TRACKING_URI: str = "file:mlruns"
MLFLOW_EXPERIMENT_NAME: str = "music_by_description"
MLFLOW_RUN_NAME: str = "clap_finetune"
DATA_VERIFICATION_SAMPLES: int = 2
SMOKE_TEST_SAMPLES: int = 2
MUSICCAPS_DATASET: str = "google/musiccaps"
MUSICCAPS_TRAIN_SPLIT: str = "train"
SONG_DESCRIBER_DATASET: str = "renumics/song-describer-dataset"
SONG_DESCRIBER_TRAIN_SPLIT: str = "train"
JAMENDO_DATASET: str = "rkstgr/mtg-jamendo"
JAMENDO_TRAIN_SPLIT: str = "train"
JAMENDO_TRUST_REMOTE_CODE: bool = True
USE_SYNTHETIC_AUDIO_IF_MISSING: bool = True
SYNTHETIC_AUDIO_NOISE_SCALE: float = 0.01
TARGET_RECALL_K: Sequence[int] = (5, 10, 20)
TARGET_SCORE_THRESHOLD: float = 0.70
INFERENCE_TEXT_PROMPT: str = "a melancholic indie ballad with soft vocals and slow tempo"
INFERENCE_NUM_NEIGHBORS: int = 3
ENABLE_DATASET_AUDIO_CAST: bool = False

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def configure_logging() -> None:
    """Configure root logger for consistent output."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def set_seed(seed: int) -> None:
    """Seed random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return the preferred torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Dataset loading and preprocessing
# ---------------------------------------------------------------------------


def _select_subset(ds: Dataset, sample_limit: Optional[int]) -> Dataset:
    if sample_limit is None:
        return ds
    limit = min(sample_limit, len(ds))
    return ds.select(range(limit))


def load_musiccaps_dataset(split: str, sample_limit: Optional[int] = None) -> Dataset:
    """Load MusicCaps captions; audio is not bundled in the public set."""
    logging.info("Loading MusicCaps dataset split=%s", split)
    ds = load_dataset(MUSICCAPS_DATASET, split=split)
    ds = _select_subset(ds, sample_limit)
    logging.info("Loaded MusicCaps with %d rows and columns=%s", len(ds), list(ds.features))
    return ds


def load_song_describer_dataset(split: str, sample_limit: Optional[int] = None) -> Dataset:
    """Load Song Describer dataset with audio metadata."""
    logging.info("Loading Song Describer dataset split=%s", split)
    ds = load_dataset(SONG_DESCRIBER_DATASET, split=split)
    if ENABLE_DATASET_AUDIO_CAST:
        try:
            ds = ds.cast_column("path", Audio(sampling_rate=AUDIO_SAMPLING_RATE))
        except (ImportError, RuntimeError) as exc:
            logging.warning("Falling back to raw Song Describer audio paths: %s", exc)
    ds = _select_subset(ds, sample_limit)
    logging.info("Loaded Song Describer rows=%d columns=%s", len(ds), list(ds.features))
    return ds


def load_jamendo_dataset(split: str, sample_limit: Optional[int] = None) -> Dataset:
    """Load Jamendo dataset; requires trust in remote code."""
    logging.info("Loading Jamendo dataset split=%s", split)
    ds = load_dataset(
        JAMENDO_DATASET,
        split=split,
        trust_remote_code=JAMENDO_TRUST_REMOTE_CODE,
    )
    audio_column = None
    for key, feature in ds.features.items():
        if isinstance(feature, Audio):
            audio_column = key
            break
    if audio_column is None:
        raise ValueError("Jamendo dataset does not expose an Audio column.")
    if ENABLE_DATASET_AUDIO_CAST:
        try:
            ds = ds.cast_column(audio_column, Audio(sampling_rate=AUDIO_SAMPLING_RATE))
        except (ImportError, RuntimeError) as exc:
            logging.warning("Falling back to raw Jamendo audio paths: %s", exc)
    ds = _select_subset(ds, sample_limit)
    logging.info(
        "Loaded Jamendo rows=%d columns=%s audio_column=%s",
        len(ds),
        list(ds.features),
        audio_column,
    )
    return ds


def verify_dataset_structure(ds: Dataset, required_columns: Sequence[str], dataset_name: str) -> None:
    missing = [col for col in required_columns if col not in ds.column_names]
    if missing:
        raise ValueError(f"{dataset_name} missing columns: {missing}")
    sample = ds[0]
    logging.info("%s sample keys=%s", dataset_name, list(sample))


def _generate_synthetic_audio(duration_seconds: float, sampling_rate: int) -> np.ndarray:
    length = int(duration_seconds * sampling_rate)
    rng = np.random.default_rng(RANDOM_SEED)
    return (rng.normal(0.0, SYNTHETIC_AUDIO_NOISE_SCALE, length)).astype(np.float32)


def _clip_audio(array: np.ndarray, sampling_rate: int) -> np.ndarray:
    max_samples = int(MAX_AUDIO_DURATION_SECONDS * sampling_rate)
    if array.shape[-1] <= max_samples:
        return array
    return array[:max_samples]


def _load_audio_from_path(path: str, sampling_rate: int) -> Tuple[Optional[np.ndarray], int]:
    try:
        import soundfile as sf  # type: ignore[import]

        audio, file_rate = sf.read(path)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if file_rate != sampling_rate:
            try:
                import librosa  # type: ignore[import]

                audio = librosa.resample(np.asarray(audio, dtype=np.float32), orig_sr=file_rate, target_sr=sampling_rate)
                file_rate = sampling_rate
            except Exception as exc:  # pylint: disable=broad-except
                logging.warning("Resample failed for %s: %s", path, exc)
        return np.asarray(audio, dtype=np.float32), file_rate
    except Exception as exc:  # pylint: disable=broad-except
        logging.warning("Audio load failed for %s: %s", path, exc)
        return None, sampling_rate


def _load_audio_from_bytes(buffer: bytes, sampling_rate: int, file_hint: Optional[str] = None) -> Tuple[Optional[np.ndarray], int]:
    try:
        import soundfile as sf  # type: ignore[import]

        with sf.SoundFile(io.BytesIO(buffer)) as sound_file:
            audio = sound_file.read(always_2d=False)
            file_rate = sound_file.samplerate
        if isinstance(audio, tuple):
            audio = np.asarray(audio[0], dtype=np.float32)
        if np.asarray(audio).ndim > 1:
            audio = np.mean(audio, axis=1)
        if file_rate != sampling_rate:
            try:
                import librosa  # type: ignore[import]

                audio = librosa.resample(np.asarray(audio, dtype=np.float32), orig_sr=file_rate, target_sr=sampling_rate)
                file_rate = sampling_rate
            except Exception as exc:  # pylint: disable=broad-except
                logging.warning("Resample failed for %s (bytes): %s", file_hint or "buffer", exc)
        return np.asarray(audio, dtype=np.float32), file_rate
    except Exception as exc:  # pylint: disable=broad-except
        logging.warning("Audio load failed from bytes (%s): %s", file_hint or "buffer", exc)
        return None, sampling_rate


def preprocess_musiccaps(ds: Dataset) -> List[Dict[str, Any]]:
    processed: List[Dict[str, Any]] = []
    for row in ds:
        caption = row.get("caption") or row.get("caption_writing")
        if caption is None:
            continue
        audio = row.get("audio")
        sampling_rate = AUDIO_SAMPLING_RATE
        if audio is None and USE_SYNTHETIC_AUDIO_IF_MISSING:
            audio = _generate_synthetic_audio(SYNTHETIC_AUDIO_SECONDS, sampling_rate)
        elif isinstance(audio, dict) and "array" in audio:
            sampling_rate = audio.get("sampling_rate", sampling_rate)
            audio = audio["array"]
        if audio is None:
            continue
        audio = _clip_audio(np.asarray(audio, dtype=np.float32), sampling_rate)
        processed.append(
            {
                "audio": audio,
                "sampling_rate": sampling_rate,
                "text": caption,
                "dataset": "musiccaps",
                "track_id": row.get("ytid", "unknown"),
            }
        )
    logging.info("Preprocessed MusicCaps samples=%d", len(processed))
    return processed


def preprocess_song_describer(ds: Dataset) -> List[Dict[str, Any]]:
    processed: List[Dict[str, Any]] = []
    for row in ds:
        caption = row.get("caption")
        audio_info = row.get("path")
        if caption is None:
            continue
        audio: Optional[np.ndarray] = None
        sampling_rate = AUDIO_SAMPLING_RATE
        if isinstance(audio_info, dict):
            array = audio_info.get("array")
            sampling_rate = audio_info.get("sampling_rate", sampling_rate)
            if array is not None:
                audio = np.asarray(array, dtype=np.float32)
            elif "bytes" in audio_info:
                audio, sampling_rate = _load_audio_from_bytes(audio_info["bytes"], sampling_rate, audio_info.get("path"))
        elif isinstance(audio_info, str) and audio_info:
            audio, sampling_rate = _load_audio_from_path(audio_info, sampling_rate)
        if audio is None and USE_SYNTHETIC_AUDIO_IF_MISSING:
            audio = _generate_synthetic_audio(SYNTHETIC_AUDIO_SECONDS, sampling_rate)
        if audio is None:
            continue
        audio = _clip_audio(np.asarray(audio, dtype=np.float32), sampling_rate)
        processed.append(
            {
                "audio": audio,
                "sampling_rate": sampling_rate,
                "text": caption,
                "dataset": "song_describer",
                "track_id": row.get("track_id", "unknown"),
            }
        )
    logging.info("Preprocessed Song Describer samples=%d", len(processed))
    return processed


def preprocess_jamendo(ds: Dataset) -> List[Dict[str, Any]]:
    processed: List[Dict[str, Any]] = []
    audio_column = None
    text_column = None
    for key, feature in ds.features.items():
        if isinstance(feature, Audio):
            audio_column = key
        if feature.dtype == "string" and "text" in key.lower():
            text_column = key
    if audio_column is None:
        raise ValueError("Jamendo dataset lacks audio column post-load.")
    if text_column is None:
        text_column = "caption" if "caption" in ds.column_names else ds.column_names[0]
    for row in ds:
        audio_info = row[audio_column]
        caption = row.get(text_column)
        if caption is None:
            continue
        audio: Optional[np.ndarray] = None
        sampling_rate = AUDIO_SAMPLING_RATE
        if isinstance(audio_info, dict):
            array = audio_info.get("array")
            sampling_rate = audio_info.get("sampling_rate", sampling_rate)
            if array is not None:
                audio = np.asarray(array, dtype=np.float32)
            elif "bytes" in audio_info:
                audio, sampling_rate = _load_audio_from_bytes(audio_info["bytes"], sampling_rate, audio_info.get("path"))
        elif isinstance(audio_info, str) and audio_info:
            audio, sampling_rate = _load_audio_from_path(audio_info, sampling_rate)
        if audio is None and USE_SYNTHETIC_AUDIO_IF_MISSING:
            audio = _generate_synthetic_audio(SYNTHETIC_AUDIO_SECONDS, sampling_rate)
        if audio is None:
            continue
        audio = _clip_audio(np.asarray(audio, dtype=np.float32), sampling_rate)
        processed.append(
            {
                "audio": audio,
                "sampling_rate": sampling_rate,
                "text": caption,
                "dataset": "jamendo",
                "track_id": row.get("track_id", "unknown"),
            }
        )
    logging.info("Preprocessed Jamendo samples=%d", len(processed))
    return processed


def aggregate_datasets(sample_limit: Optional[int] = None) -> List[Dict[str, Any]]:
    musiccaps = preprocess_musiccaps(load_musiccaps_dataset(MUSICCAPS_TRAIN_SPLIT, sample_limit))
    song_describer = preprocess_song_describer(load_song_describer_dataset(SONG_DESCRIBER_TRAIN_SPLIT, sample_limit))
    jamendo: List[Dict[str, Any]] = []
    try:
        jamendo = preprocess_jamendo(load_jamendo_dataset(JAMENDO_TRAIN_SPLIT, sample_limit))
    except Exception as exc:  # pylint: disable=broad-except
        logging.warning("Jamendo dataset unavailable: %s", exc)
    aggregated = [*song_describer, *jamendo, *musiccaps]
    if not aggregated:
        raise RuntimeError("No usable samples across datasets.")
    logging.info("Aggregated dataset size=%d", len(aggregated))
    return aggregated


# ---------------------------------------------------------------------------
# Torch dataset and collate utilities
# ---------------------------------------------------------------------------


class RetrievalDataset(TorchDataset):
    """Thin wrapper to serve preprocessed audio-text pairs."""

    def __init__(self, data: Sequence[Dict[str, Any]]):
        self._data = list(data)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self._data[index]


def build_collate_fn(processor: AutoProcessor):
    def collate(batch: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audios = [item["audio"] for item in batch]
        sampling_rates = [item.get("sampling_rate", AUDIO_SAMPLING_RATE) for item in batch]
        texts = [item["text"] for item in batch]
        if len(set(sampling_rates)) > 1:
            logging.warning("Mixed sampling rates detected; normalizing to %d", AUDIO_SAMPLING_RATE)
            audios = [
                np.asarray(a[: int(MAX_AUDIO_DURATION_SECONDS * sr)], dtype=np.float32)
                if sr == AUDIO_SAMPLING_RATE
                else np.asarray(a, dtype=np.float32)
                for a, sr in zip(audios, sampling_rates)
            ]
        inputs = processor(
            text=texts,
            audios=audios,
            sampling_rate=AUDIO_SAMPLING_RATE,
            return_tensors="pt",
            padding=True,
        )
        return inputs

    return collate


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_recall_at_k(similarity: torch.Tensor, k: int) -> float:
    k = min(k, similarity.size(-1))
    if k <= 0:
        return 0.0
    topk = similarity.topk(k, dim=-1).indices
    matches = torch.arange(similarity.size(0), device=similarity.device).unsqueeze(1)
    hits = (topk == matches).any(dim=1).float()
    return hits.mean().item()


def compute_mrr(similarity: torch.Tensor) -> float:
    ranks = similarity.argsort(dim=-1, descending=True)
    targets = torch.arange(similarity.size(0), device=similarity.device)
    reciprocal_ranks = []
    for i in range(similarity.size(0)):
        rank = (ranks[i] == targets[i]).nonzero(as_tuple=False)
        if rank.numel() == 0:
            reciprocal_ranks.append(0.0)
        else:
            reciprocal_ranks.append(1.0 / (rank.item() + 1))
    return float(np.mean(reciprocal_ranks))


def compute_precision_at_k(similarity: torch.Tensor, k: int) -> float:
    k = min(k, similarity.size(-1))
    if k <= 0:
        return 0.0
    topk = similarity.topk(k, dim=-1).indices
    matches = torch.arange(similarity.size(0), device=similarity.device).unsqueeze(1)
    hits = (topk == matches).float()
    return hits.mean().item()


def compute_ndcg_at_k(similarity: torch.Tensor, k: int) -> float:
    k = min(k, similarity.size(-1))
    if k <= 0:
        return 0.0
    ranks = similarity.argsort(dim=-1, descending=True)[:, :k]
    gains = torch.zeros_like(ranks, dtype=torch.float32)
    targets = torch.arange(similarity.size(0), device=similarity.device)
    for i in range(similarity.size(0)):
        matches = (ranks[i] == targets[i]).float()
        gains[i] = matches / torch.log2(torch.arange(k, device=similarity.device, dtype=torch.float32) + 2)
    dcg = gains.sum(dim=1)
    idcg_denom = math.log2(2)
    idcg = torch.tensor([1.0 / idcg_denom] * similarity.size(0), device=similarity.device)
    ndcg = (dcg / idcg).clamp(max=1.0)
    return ndcg.mean().item()


def compute_retrieval_metrics(similarity: torch.Tensor) -> Dict[str, float]:
    metrics = {"mrr": compute_mrr(similarity)}
    for k in TARGET_RECALL_K:
        metrics[f"recall@{k}"] = compute_recall_at_k(similarity, k)
        metrics[f"precision@{k}"] = compute_precision_at_k(similarity, k)
        metrics[f"ndcg@{k}"] = compute_ndcg_at_k(similarity, k)
    return metrics


def sanitize_mlflow_key(key: str) -> str:
    return key.replace("@", "_at_")


# ---------------------------------------------------------------------------
# Training, evaluation, and inference
# ---------------------------------------------------------------------------


def initialize_model_and_processor() -> Tuple[ClapModel, AutoProcessor]:
    logging.info("Loading CLAP model %s", MODEL_NAME)
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = ClapModel.from_pretrained(MODEL_NAME)
    return model, processor


def setup_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    return AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=BETAS,
        eps=EPSILON,
    )


def evaluate_model(
    model: ClapModel,
    processor: AutoProcessor,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    audio_embeddings: List[torch.Tensor] = []
    text_embeddings: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(**batch, return_loss=False)
            audio_embeddings.append(outputs.audio_embeds)
            text_embeddings.append(outputs.text_embeds)
    audio_matrix = F.normalize(torch.cat(audio_embeddings, dim=0), dim=-1)
    text_matrix = F.normalize(torch.cat(text_embeddings, dim=0), dim=-1)
    similarity = text_matrix @ audio_matrix.t()
    return compute_retrieval_metrics(similarity)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    checkpoint_dir: Path,
    prefix: str,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"{prefix}_step-{step}.pt"
    torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict()}, path)
    logging.info("Saved checkpoint %s", path)
    return path


def log_metrics_to_mlflow(step: int, metrics: Dict[str, float]) -> None:
    for key, value in metrics.items():
        mlflow.log_metric(sanitize_mlflow_key(key), value, step=step)


def train(
    model: ClapModel,
    processor: AutoProcessor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> None:
    optimizer = setup_optimizer(model)
    global_step = 0
    model.to(device)
    mlflow.log_params(
        {
            "model_name": MODEL_NAME,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": TRAIN_BATCH_SIZE,
            "max_train_steps": MAX_TRAIN_STEPS,
            "num_epochs": NUM_EPOCHS,
            "validation_interval": VALIDATION_INTERVAL_STEPS,
        }
    )
    for epoch in range(NUM_EPOCHS):
        logging.info("Starting epoch %d", epoch + 1)
        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            model.train()
            outputs = model(**batch, return_loss=True)
            loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            if (global_step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                optimizer.zero_grad()
            global_step += 1
            if global_step % VALIDATION_INTERVAL_STEPS == 0:
                metrics = evaluate_model(model, processor, val_loader, device)
                logging.info("Validation metrics at step %d: %s", global_step, json.dumps(metrics, indent=2))
                log_metrics_to_mlflow(global_step, metrics)
                save_checkpoint(model, optimizer, global_step, CHECKPOINT_DIR, CHECKPOINT_PREFIX)
            mlflow.log_metric("train_loss", loss.item(), step=global_step)
            if global_step >= MAX_TRAIN_STEPS:
                logging.info("Reached max training steps (%d)", MAX_TRAIN_STEPS)
                return


def split_dataset(data: List[Dict[str, Any]], validation_fraction: float = 0.2) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    random.shuffle(data)
    split_index = int(len(data) * (1 - validation_fraction))
    return data[:split_index], data[split_index:]


def run_inference_example(
    model: ClapModel,
    processor: AutoProcessor,
    device: torch.device,
    reference_samples: Sequence[Dict[str, Any]],
    prompt: str = INFERENCE_TEXT_PROMPT,
    neighbors: int = INFERENCE_NUM_NEIGHBORS,
) -> Dict[str, Any]:
    if not reference_samples:
        raise ValueError("Reference samples required for inference example.")
    model.eval()
    audio_batch = [sample["audio"] for sample in reference_samples]
    texts = [prompt]
    with torch.no_grad():
        inputs = processor(
            text=texts,
            audios=audio_batch,
            sampling_rate=AUDIO_SAMPLING_RATE,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        outputs = model(**inputs, return_loss=False)
        query_embed = F.normalize(outputs.text_embeds[0:1], dim=-1)
        audio_embeds = F.normalize(outputs.audio_embeds, dim=-1)
        similarities = (query_embed @ audio_embeds.t()).squeeze(0)
        top_scores, top_indices = similarities.topk(min(neighbors, similarities.size(0)))
    results = []
    for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
        sample = reference_samples[idx]
        results.append({"score": score, "dataset": sample["dataset"], "track_id": sample["track_id"]})
    logging.info("Inference example prompt='%s' results=%s", prompt, results)
    return {"prompt": prompt, "results": results}


# ---------------------------------------------------------------------------
# MLflow setup
# ---------------------------------------------------------------------------


def initialize_mlflow() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------


def build_dataloaders(sample_limit: Optional[int] = None) -> Tuple[DataLoader, DataLoader, List[Dict[str, Any]]]:
    aggregated = aggregate_datasets(sample_limit=sample_limit)
    train_samples, val_samples = split_dataset(aggregated)
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    collate_fn = build_collate_fn(processor)
    train_loader = DataLoader(
        RetrievalDataset(train_samples),
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        RetrievalDataset(val_samples),
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader, aggregated


def run_workflow(mode: str) -> None:
    configure_logging()
    set_seed(RANDOM_SEED)
    initialize_mlflow()
    device = get_device()
    logging.info("Using device=%s", device)
    if mode == "smoke-test":
        train_loader, val_loader, aggregated = build_dataloaders(sample_limit=SMOKE_TEST_SAMPLES)
        model, processor = initialize_model_and_processor()
        with mlflow.start_run(run_name=f"{MLFLOW_RUN_NAME}_smoke_{int(time.time())}"):
            metrics = evaluate_model(model.to(device), processor, val_loader, device)
            logging.info("Smoke-test evaluation metrics=%s", json.dumps(metrics, indent=2))
            mlflow.log_metrics({f"smoke_{sanitize_mlflow_key(k)}": v for k, v in metrics.items()})
            run_inference_example(model, processor, device, aggregated[:SMOKE_TEST_SAMPLES])
        return
    train_loader, val_loader, aggregated = build_dataloaders(sample_limit=None)
    model, processor = initialize_model_and_processor()
    with mlflow.start_run(run_name=f"{MLFLOW_RUN_NAME}_{int(time.time())}"):
        train(model, processor, train_loader, val_loader, device)
        inference_payload = run_inference_example(model, processor, device, aggregated[:EVAL_BATCH_SIZE])
        mlflow.log_dict(inference_payload, "inference_example.json")
        logging.info("Workflow complete. Inference payload saved to MLflow artifact.")


# ---------------------------------------------------------------------------
# Testing helper
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="CLAP fine-tuning harness")
    parser.add_argument("--mode", choices=["train", "smoke-test"], default="train")
    args = parser.parse_args(argv)
    run_workflow(args.mode)


if __name__ == "__main__":
    main()
