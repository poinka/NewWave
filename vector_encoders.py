#!/usr/bin/env python3
"""Utility functions for lyric and audio embedding generation."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Sequence

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor, AutoTokenizer, ClapModel

TEXT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
CLAP_MODEL_NAME = "laion/clap-htsat-unfused"
AUDIO_SAMPLING_RATE = 48_000
MAX_AUDIO_SECONDS = 30.0

_TEXT_COMPONENTS: tuple[AutoTokenizer, AutoModel] | None = None
_CLAP_COMPONENTS: tuple[AutoProcessor, ClapModel] | None = None


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _mean_pooling(model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(model_output.size()).float()
    summed = torch.sum(model_output * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def _batch(iterable: Sequence, size: int) -> Iterable[Sequence]:
    for idx in range(0, len(iterable), size):
        yield iterable[idx : idx + size]


@lru_cache(maxsize=1)
def _load_text_components() -> tuple[AutoTokenizer, AutoModel]:
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    model = AutoModel.from_pretrained(TEXT_MODEL_NAME)
    model.to(_get_device())
    model.eval()
    return tokenizer, model


@lru_cache(maxsize=1)
def _load_clap_components() -> tuple[AutoProcessor, ClapModel]:
    processor = AutoProcessor.from_pretrained(CLAP_MODEL_NAME)
    model = ClapModel.from_pretrained(CLAP_MODEL_NAME)
    model.to(_get_device())
    model.eval()
    return processor, model


@torch.no_grad()
def encode_lyrics_biencoder(texts: List[str], batch_size: int = 16) -> np.ndarray:
    """Encode lyric strings with a sentence bi-encoder model."""
    tokenizer, model = _load_text_components()
    sanitized = [t if isinstance(t, str) and t.strip() else "" for t in texts]
    device = _get_device()
    outputs: List[np.ndarray] = []
    for chunk in _batch(sanitized, batch_size):
        encoded = tokenizer(
            list(chunk),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        model_out = model(**encoded)
        pooled = _mean_pooling(model_out.last_hidden_state, encoded["attention_mask"])
        normalized = F.normalize(pooled, p=2, dim=-1)
        outputs.append(normalized.cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


def _load_audio(path: Path) -> np.ndarray:
    audio, sr = librosa.load(path, sr=AUDIO_SAMPLING_RATE, mono=True)
    max_samples = int(MAX_AUDIO_SECONDS * AUDIO_SAMPLING_RATE)
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    return audio


@torch.no_grad()
def encode_audio_clap(paths: List[Path], batch_size: int = 2) -> np.ndarray:
    """Encode audio waveforms via CLAP's audio tower."""
    processor, model = _load_clap_components()
    device = _get_device()
    embeddings: List[np.ndarray] = []
    for chunk_paths in _batch(paths, batch_size):
        audio_arrays = [_load_audio(Path(p)) for p in chunk_paths]
        inputs = processor(
            audios=audio_arrays,
            sampling_rate=AUDIO_SAMPLING_RATE,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        features = model.get_audio_features(**inputs)
        normalized = F.normalize(features, p=2, dim=-1)
        embeddings.append(normalized.cpu().numpy().astype(np.float32))
    return np.concatenate(embeddings, axis=0)


@torch.no_grad()
def encode_text_clap(texts: List[str], batch_size: int = 8) -> np.ndarray:
    """Encode text prompts into the CLAP joint space."""
    processor, model = _load_clap_components()
    device = _get_device()
    sanitized = [t if isinstance(t, str) and t.strip() else "" for t in texts]
    embeddings: List[np.ndarray] = []
    for chunk in _batch(sanitized, batch_size):
        inputs = processor(text=list(chunk), return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        features = model.get_text_features(**inputs)
        normalized = F.normalize(features, p=2, dim=-1)
        embeddings.append(normalized.cpu().numpy().astype(np.float32))
    return np.concatenate(embeddings, axis=0)
