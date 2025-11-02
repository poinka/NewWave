from __future__ import annotations

import contextlib
import os
import random
import shutil
import subprocess
import sys
import json
from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
from types import MethodType
from typing import Dict, List, Optional, Tuple
import librosa
import numpy as np
import requests
import torch
import torch.nn.functional as F
from datasets import load_dataset, Audio
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, ClapModel
from transformers.modeling_outputs import BaseModelOutputWithPooling
from tqdm.auto import tqdm
import math


# -----------------------------------------------------------------------------
# Configuration (from clap.py)
# -----------------------------------------------------------------------------

MODEL_NAME = "laion/clap-htsat-unfused"
SONG_DESCRIBER_DATASET_ID = "renumics/song-describer-dataset"

RANDOM_SEED = 42
AUDIO_SAMPLING_RATE = 48_000
CLIP_SECONDS = 240
EVAL_BATCH_SIZE = 8

TEXT_START_MAX_LEN = 512
TEXT_TARGET_MAX_LEN_CAP = 512
AUDIO_TARGET_MAX_SECONDS_CAP = 300

CHECKPOINT_DIR = Path("checkpoints")
LATEST_CHECKPOINT_DIR = CHECKPOINT_DIR / "latest"

HF_ENDPOINT_OVERRIDE = "https://hf-mirror.com"
HF_REQUEST_TIMEOUT = "60"

HF_DATASETS_CACHE_DIR = Path("data/song_describer_cache")
AUDIO_CACHE_DIR = Path("data/audio_cache_test")

# Evaluation constants
EVAL_SPLIT_SIZE = 1000  # Number of samples to evaluate on
RETRIEVAL_K_VALUES = [1, 5, 10, 20]


# -----------------------------------------------------------------------------
# Data structures for metrics
# -----------------------------------------------------------------------------

@dataclass
class RetrievalMetrics:
    acc: float
    mrr: float
    recall_at_k: Dict[int, float]
    precision_at_k: Dict[int, float]


class MissingAudioError(Exception):
    """Raised when an audio asset cannot be resolved."""


# -----------------------------------------------------------------------------
# Utility functions (from clap.py)
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


def disable_fe_cap(processor: AutoProcessor) -> AutoProcessor:
    feature_extractor = getattr(processor, "feature_extractor", None)
    if feature_extractor is not None and hasattr(feature_extractor, "nb_max_samples"):
        if getattr(feature_extractor, "nb_max_samples") is not None:
            feature_extractor.nb_max_samples = None
    return processor


@torch.no_grad()
def _interp_1d_pos(pos: torch.Tensor, new_n: int) -> torch.Tensor:
    squeeze = False
    if pos.dim() == 2:
        pos = pos.unsqueeze(0)
        squeeze = True
    if pos.dim() != 3:
        raise ValueError(f"Expected position tensor with 2 or 3 dims, got shape {tuple(pos.shape)}")
    _, old_n, _ = pos.shape
    if new_n == old_n:
        return pos.squeeze(0) if squeeze else pos
    resized = F.interpolate(pos.permute(0, 2, 1), size=new_n, mode="linear", align_corners=False).permute(0, 2, 1)
    return resized.squeeze(0) if squeeze else resized


def enable_audio_dynamic_context(model: ClapModel) -> ClapModel:
    audio_model = getattr(model, "audio_model", None)
    audio_encoder = getattr(audio_model, "audio_encoder", None)
    if audio_encoder is not None:
        _enable_dynamic_audio_context(audio_encoder)
    return model


def _audio_encoder_dynamic_reshape_mel2img(self, normalized_input_features: torch.Tensor) -> torch.Tensor:
    batch, channels, time_length, freq_length = normalized_input_features.shape

    target_freq = max(freq_length, self.spec_size // self.freq_ratio)
    if freq_length != target_freq:
        normalized_input_features = F.interpolate(
            normalized_input_features, size=(time_length, target_freq), mode="bicubic", align_corners=True
        )
        freq_length = target_freq

    if time_length % self.freq_ratio != 0:
        pad = self.freq_ratio - (time_length % self.freq_ratio)
        normalized_input_features = F.pad(normalized_input_features, (0, 0, 0, pad))
        time_length += pad

    reshaped = normalized_input_features.reshape(
        batch, channels * self.freq_ratio, time_length // self.freq_ratio, freq_length
    )
    reshaped = reshaped.permute(0, 1, 3, 2).contiguous()
    reshaped = reshaped.reshape(batch, channels, freq_length * self.freq_ratio, time_length // self.freq_ratio)
    return reshaped


def _audio_encoder_dynamic_forward(
    self,
    input_features,
    is_longer: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    output_attentions: Optional[bool] = False,
    output_hidden_states: Optional[bool] = False,
    output_hidden_states_before_downsampling: Optional[bool] = False,
    always_partition: Optional[bool] = False,
    return_dict: Optional[bool] = True,
):
    input_features = input_features.transpose(1, 3)
    normalized_input_features = self.batch_norm(input_features)
    normalized_input_features = normalized_input_features.transpose(1, 3)

    is_longer_list_idx = None
    if self.enable_fusion and is_longer is not None:
        is_longer_list = is_longer.to(input_features.device)
        is_longer_list_idx = torch.where(is_longer_list == 1)[0]

    hidden_states_image = self.reshape_mel2img(normalized_input_features)

    img_height = hidden_states_image.shape[2]
    img_width = hidden_states_image.shape[3]

    pad_height = (self.patch_stride[0] - img_height % self.patch_stride[0]) % self.patch_stride[0]
    pad_width = (self.patch_stride[1] - img_width % self.patch_stride[1]) % self.patch_stride[1]
    if pad_height or pad_width:
        hidden_states_image = F.pad(hidden_states_image, (0, pad_width, 0, pad_height))
        img_height += pad_height
        img_width += pad_width

    current_resolution = (
        img_height // self.patch_stride[0],
        img_width // self.patch_stride[1],
    )
    self._current_hw = current_resolution
    try:
        pe = self.patch_embed
        if hasattr(pe, "img_size"):
            pe.img_size = (img_height, img_width)
        if hasattr(pe, "grid_size"):
            pe.grid_size = current_resolution
        if hasattr(pe, "num_patches"):
            pe.num_patches = int(current_resolution[0] * current_resolution[1])
    except Exception:
        pass

    hidden_states = self.patch_embed(hidden_states_image, is_longer_list_idx)

    all_hidden_states = () if output_hidden_states else None
    all_reshaped_hidden_states = () if output_hidden_states else None
    all_self_attentions = () if output_attentions else None

    if output_hidden_states:
        batch_size, _, hidden_size = hidden_states.shape
        reshaped_hidden_state = hidden_states.view(batch_size, current_resolution[0], current_resolution[1], hidden_size)
        reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
        all_hidden_states += (hidden_states,)
        all_reshaped_hidden_states += (reshaped_hidden_state,)

    for idx, layer_module in enumerate(self.layers):
        layer_head_mask = head_mask[idx] if head_mask is not None else None
        layer_outputs = layer_module(
            hidden_states, current_resolution, layer_head_mask, output_attentions, always_partition
        )

        hidden_states = layer_outputs[0]
        hidden_states_before_downsampling = layer_outputs[1]
        output_dimensions = layer_outputs[2]

        if output_hidden_states and output_hidden_states_before_downsampling:
            batch_size, _, hidden_size = hidden_states_before_downsampling.shape
            reshaped_hidden_state = hidden_states_before_downsampling.view(
                batch_size, output_dimensions[0], output_dimensions[1], hidden_size
            )
            reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
            all_hidden_states += (hidden_states_before_downsampling,)
            all_reshaped_hidden_states += (reshaped_hidden_state,)
        elif output_hidden_states and not output_hidden_states_before_downsampling:
            batch_size, _, hidden_size = hidden_states.shape
            reshaped_hidden_state = hidden_states.view(
                batch_size, output_dimensions[-2], output_dimensions[-1], hidden_size
            )
            reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
            all_hidden_states += (hidden_states,)
            all_reshaped_hidden_states += (reshaped_hidden_state,)

        if output_attentions:
            all_self_attentions += layer_outputs[3:]

        current_resolution = (output_dimensions[-2], output_dimensions[-1])

    last_hidden_state = self.norm(hidden_states)

    batch_size, _, n_channels = last_hidden_state.shape
    height, width = current_resolution

    last_hidden_state = last_hidden_state.permute(0, 2, 1).contiguous().reshape(batch_size, n_channels, height, width)

    batch_size, n_channels, n_frequencies, n_temp = last_hidden_state.shape
    c_freq_bin = max(1, n_frequencies // self.freq_ratio)
    last_hidden_state = last_hidden_state.reshape(
        batch_size, n_channels, n_frequencies // c_freq_bin, c_freq_bin, n_temp
    )
    last_hidden_state = (
        last_hidden_state.permute(0, 1, 3, 2, 4).contiguous().reshape(batch_size, n_channels, c_freq_bin, -1)
    )
    latent_output = self.avgpool(torch.flatten(last_hidden_state, 2))
    latent_output = torch.flatten(latent_output, 1)

    if not return_dict:
        return tuple(
            v
            for v in [
                last_hidden_state,
                latent_output,
                all_reshaped_hidden_states,
                all_self_attentions,
            ]
            if v is not None
        )

    return BaseModelOutputWithPooling(
        last_hidden_state=last_hidden_state,
        pooler_output=latent_output,
        hidden_states=all_reshaped_hidden_states,
        attentions=all_self_attentions,
    )


def _enable_dynamic_audio_context(audio_encoder) -> None:
    if getattr(audio_encoder, "_dynamic_context_enabled", False):
        return
    audio_encoder._dynamic_context_enabled = True
    audio_encoder._orig_reshape_mel2img = audio_encoder.reshape_mel2img
    audio_encoder._orig_forward = audio_encoder.forward
    audio_encoder.reshape_mel2img = MethodType(_audio_encoder_dynamic_reshape_mel2img, audio_encoder)
    audio_encoder.forward = MethodType(_audio_encoder_dynamic_forward, audio_encoder)


def enable_text_pos_interpolation(model: ClapModel) -> ClapModel:
    text_embedding_module = None

    for module in model.modules():
        embedding = getattr(module, "position_embeddings", None)
        if isinstance(embedding, torch.nn.Embedding):
            text_embedding_module = embedding
            break

    if text_embedding_module is None:
        return model

    original_weight = text_embedding_module.weight.detach().clone().cpu()
    padding_idx = text_embedding_module.padding_idx
    requires_grad = text_embedding_module.weight.requires_grad

    def hook(module: torch.nn.Embedding, inputs: tuple, kwargs: dict | None = None) -> None:
        if not inputs:
            return
        position_ids = inputs[0]
        if position_ids is None:
            return
        max_pos = int(position_ids.max().item()) + 1
        if padding_idx is not None:
            max_pos = max(max_pos, padding_idx + 1)
        if max_pos <= module.weight.shape[0]:
            return
        base = original_weight
        resized = _interp_1d_pos(base, max_pos)
        new_weight = resized.to(module.weight.device, dtype=module.weight.dtype)
        module.weight = torch.nn.Parameter(new_weight, requires_grad=requires_grad)
        module.num_embeddings = max_pos

    text_embedding_module.register_forward_pre_hook(hook, with_kwargs=False)
    return model


@contextlib.contextmanager
def _suppress_audio_backend_warnings() -> StringIO:
    buffer = StringIO()
    original_stderr = sys.stderr
    fd = None
    old_fd = None
    devnull = None
    try:
        try:
            fd = original_stderr.fileno()
        except (OSError, ValueError, AttributeError):
            fd = None
        if fd is not None:
            devnull = open(os.devnull, "w")
            old_fd = os.dup(fd)
            os.dup2(devnull.fileno(), fd)
        with contextlib.redirect_stderr(buffer):
            yield buffer
    finally:
        if fd is not None and old_fd is not None:
            try:
                os.dup2(old_fd, fd)
            finally:
                os.close(old_fd)
        if devnull is not None:
            devnull.close()


def _librosa_decode(source) -> tuple[np.ndarray, int]:
    if isinstance(source, (str, os.PathLike)):
        target = str(source)
        needs_reset = False
    else:
        target = source
        needs_reset = hasattr(target, "seek")

    if needs_reset:
        try:
            target.seek(0)
        except Exception:
            pass

    with _suppress_audio_backend_warnings() as stderr_buffer:
        try:
            arr, sr = librosa.load(target, sr=None, mono=True)
        except Exception as exc:
            details = stderr_buffer.getvalue().strip()
            note = f" ({details})" if details else ""
            raise MissingAudioError(f"Failed to decode audio{note}") from exc
    if arr.ndim > 1:
        arr = arr.mean(axis=-1)
    return arr.astype(np.float32), sr


def _load_audio_entry(audio_entry) -> tuple[np.ndarray, int]:
    if audio_entry is None:
        raise MissingAudioError("Audio entry is None")
    if isinstance(audio_entry, dict):
        cache_dir = audio_entry.get("_cache_dir")
        cache_key = audio_entry.get("_cache_key")

        # 1) Direct array
        if audio_entry.get("array") is not None:
            arr = np.asarray(audio_entry["array"], dtype=np.float32)
            sr = int(audio_entry.get("sampling_rate") or audio_entry.get("samplingRate") or AUDIO_SAMPLING_RATE)
            if arr.ndim > 1:
                arr = arr.mean(axis=-1)
            return arr, sr

        # 2) Bytes payload
        byte_data = audio_entry.get("bytes")
        if byte_data is not None:
            if isinstance(byte_data, memoryview):
                byte_data = byte_data.tobytes()
            elif isinstance(byte_data, np.ndarray):
                byte_data = byte_data.tobytes()
            elif isinstance(byte_data, str):
                byte_data = byte_data.encode("latin-1")
            elif not isinstance(byte_data, (bytes, bytearray)):
                byte_data = bytes(byte_data)

            p_hint = audio_entry.get("path") or audio_entry.get("filepath")
            suffix = Path(str(p_hint)).suffix if isinstance(p_hint, str) else ""

            if cache_dir and cache_key:
                cache_dir_path = Path(cache_dir)
                cache_dir_path.mkdir(parents=True, exist_ok=True)
                suffix = suffix or ".mp3"
                cache_path = cache_dir_path / f"{cache_key}{suffix}"
                if not cache_path.exists() or cache_path.stat().st_size == 0:
                    with open(cache_path, "wb") as f:
                        f.write(byte_data)
                source = cache_path
            else:
                source = BytesIO(byte_data)

            arr, sr = _librosa_decode(source)
            if arr.ndim > 1:
                arr = arr.mean(axis=-1)
            return arr.astype(np.float32), sr

        # 3) Path on disk
        p = audio_entry.get("path") or audio_entry.get("filepath")
        if p:
            source = str(p)
            arr, sr = _librosa_decode(source)
            if arr.ndim > 1:
                arr = arr.mean(axis=-1)
            return arr.astype(np.float32), sr

        raise MissingAudioError("Audio dict has no decodable data")

    elif isinstance(audio_entry, str):
        source = audio_entry
        arr, sr = _librosa_decode(source)
        if arr.ndim > 1:
            arr = arr.mean(axis=-1)
        return arr.astype(np.float32), sr

    # Fallbacks for already-decoded waveforms
    elif isinstance(audio_entry, np.ndarray):
        arr = audio_entry.astype(np.float32)
        return arr, AUDIO_SAMPLING_RATE
    elif isinstance(audio_entry, torch.Tensor):
        arr = audio_entry.detach().cpu().numpy().astype(np.float32)
        return arr, AUDIO_SAMPLING_RATE
    else:
        arr = np.asarray(audio_entry, dtype=np.float32)
        if arr.size == 0:
            raise MissingAudioError("Audio array is empty")
        return arr, AUDIO_SAMPLING_RATE


def _extract_caption(row: Dict) -> str:
    # Use description, caption, text, or fallback
    for key in ("description", "caption", "text", "title"):
        value = row.get(key)
        if value and isinstance(value, str) and value.strip():
            return value.strip()
    return "instrumental music track"


# -----------------------------------------------------------------------------
# Dataset class
# -----------------------------------------------------------------------------

class AudioTextDataset(Dataset):
    def __init__(self, hf_dataset, cache_dir: Optional[Path] = None):
        self.ds = hf_dataset
        self.cache_dir = cache_dir

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.ds[index]
        audio_entry = self._find_audio_field(row)
        if isinstance(audio_entry, dict):
            audio_entry = dict(audio_entry)
            if self.cache_dir is not None:
                audio_entry.setdefault("_cache_dir", str(self.cache_dir))
                audio_entry.setdefault("_cache_key", f"{index:06d}")
        text = _extract_caption(row)
        return {"audio": audio_entry, "text": text}

    @staticmethod
    def _find_audio_field(row: Dict) -> Optional[object]:
        audio = row.get("audio")
        if isinstance(audio, dict) or isinstance(audio, str):
            return audio
        for key, value in row.items():
            if isinstance(value, dict) and ("array" in value or "path" in value or "filepath" in value):
                return value
            if isinstance(value, str) and key.lower().endswith("audio"):
                return value
        return row.get("path") or row.get("filepath")


def collate_batch(batch: List[Dict[str, object]], processor: AutoProcessor, max_audio_seconds: int = CLIP_SECONDS, max_text_len: int = TEXT_TARGET_MAX_LEN_CAP) -> Optional[Dict[str, torch.Tensor]]:
    target_len = int(AUDIO_SAMPLING_RATE * max_audio_seconds)

    audios: List[np.ndarray] = []
    texts: List[str] = []

    for item in batch:
        audio_entry = item.get("audio")
        try:
            arr, sampling_rate = _load_audio_entry(audio_entry)
        except MissingAudioError:
            continue
        if arr.ndim > 1:
            arr = arr.mean(axis=-1)
        if sampling_rate != AUDIO_SAMPLING_RATE and arr.size > 0:
            arr = librosa.resample(arr, orig_sr=sampling_rate, target_sr=AUDIO_SAMPLING_RATE)
        if target_len > 0 and arr.shape[0] > target_len:
            # Take middle segment for consistency
            start = max(0, (arr.shape[0] - target_len) // 2)
            arr = arr[start : start + target_len]
        audios.append(arr.astype(np.float32))
        texts.append(str(item.get("text", "")))

    if not audios:
        return None

    feature_extractor_max_length = max(
        target_len, max(audio.shape[0] for audio in audios) if audios else 0
    )

    text_inputs = processor(
        text=texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_text_len,
    )
    audio_inputs = processor.feature_extractor(
        raw_speech=audios,
        sampling_rate=AUDIO_SAMPLING_RATE,
        return_tensors="pt",
        padding="longest",
        truncation=False,
        max_length=int(feature_extractor_max_length),
    )

    return {
        "input_ids": text_inputs["input_ids"],
        "attention_mask": text_inputs.get("attention_mask"),
        "input_features": audio_inputs["input_features"],
    }


# -----------------------------------------------------------------------------
# Metrics computation
# -----------------------------------------------------------------------------

def compute_retrieval_metrics(audio_embeddings: torch.Tensor, text_embeddings: torch.Tensor, k_values: List[int]) -> RetrievalMetrics:
    """
    Compute retrieval metrics for audio-text matching.

    Args:
        audio_embeddings: Audio embeddings of shape (N, D)
        text_embeddings: Text embeddings of shape (N, D)
        k_values: List of k values for Recall@k and Precision@k

    Returns:
        RetrievalMetrics object with computed metrics
    """
    # Normalize embeddings
    audio_embeddings = F.normalize(audio_embeddings, dim=-1)
    text_embeddings = F.normalize(text_embeddings, dim=-1)

    # Compute similarity matrix (audio-to-text and text-to-audio)
    audio2text_sim = torch.matmul(audio_embeddings, text_embeddings.T)  # (N, N)
    text2audio_sim = audio2text_sim.T  # (N, N)

    N = audio_embeddings.shape[0]

    # Accuracy (Recall@1)
    audio2text_acc = (torch.argmax(audio2text_sim, dim=1) == torch.arange(N, device=audio_embeddings.device)).float().mean()
    text2audio_acc = (torch.argmax(text2audio_sim, dim=1) == torch.arange(N, device=audio_embeddings.device)).float().mean()
    acc = (audio2text_acc + text2audio_acc) / 2

    # Mean Reciprocal Rank (MRR)
    audio2text_ranks = torch.argsort(audio2text_sim, dim=1, descending=True)
    text2audio_ranks = torch.argsort(text2audio_sim, dim=1, descending=True)

    audio2text_mrr = 0.0
    text2audio_mrr = 0.0

    for i in range(N):
        # Find rank of correct match
        audio2text_correct_rank = (audio2text_ranks[i] == i).nonzero(as_tuple=True)[0].item()
        text2audio_correct_rank = (text2audio_ranks[i] == i).nonzero(as_tuple=True)[0].item()

        audio2text_mrr += 1.0 / (audio2text_correct_rank + 1)
        text2audio_mrr += 1.0 / (text2audio_correct_rank + 1)

    mrr = (audio2text_mrr + text2audio_mrr) / (2 * N)

    # Recall@k and Precision@k
    recall_at_k = {}
    precision_at_k = {}

    for k in k_values:
        audio2text_recall = 0.0
        audio2text_precision = 0.0
        text2audio_recall = 0.0
        text2audio_precision = 0.0

        for i in range(N):
            # Get top-k predictions
            audio2text_top_k = audio2text_ranks[i, :k]
            text2audio_top_k = text2audio_ranks[i, :k]

            # Check if correct match is in top-k
            audio2text_correct_in_top_k = (audio2text_top_k == i).any().float().item()
            text2audio_correct_in_top_k = (text2audio_top_k == i).any().float().item()

            audio2text_recall += audio2text_correct_in_top_k
            text2audio_recall += text2audio_correct_in_top_k

            audio2text_precision += audio2text_correct_in_top_k / k
            text2audio_precision += text2audio_correct_in_top_k / k

        recall_at_k[k] = (audio2text_recall + text2audio_recall) / (2 * N)
        precision_at_k[k] = (audio2text_precision + text2audio_precision) / (2 * N)

    return RetrievalMetrics(
        acc=acc.item(),
        mrr=mrr,
        recall_at_k=recall_at_k,
        precision_at_k=precision_at_k
    )


def evaluate_model(
    model: ClapModel,
    processor: AutoProcessor,
    loader: DataLoader,
    device: torch.device,
    max_audio_seconds: int = CLIP_SECONDS,
    max_text_len: int = TEXT_TARGET_MAX_LEN_CAP
) -> RetrievalMetrics:
    """
    Evaluate the model on song-describer dataset with retrieval metrics.

    Args:
        model: CLAP model
        processor: CLAP processor
        loader: DataLoader for evaluation data
        device: Device to run evaluation on
        max_audio_seconds: Maximum audio length in seconds
        max_text_len: Maximum text length in tokens

    Returns:
        RetrievalMetrics object with evaluation results
    """
    model.eval()

    all_audio_embeddings = []
    all_text_embeddings = []

    progress_bar = tqdm(loader, desc="Evaluating", leave=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            if batch is None:
                continue

            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items() if v is not None}

            try:
                # Get embeddings
                outputs = model(**batch)

                # Extract audio and text embeddings
                audio_embeds = outputs.audio_embeds
                text_embeds = outputs.text_embeds

                # Store embeddings
                all_audio_embeddings.append(audio_embeds.cpu())
                all_text_embeddings.append(text_embeds.cpu())

                progress_bar.set_postfix({"processed": batch_idx * EVAL_BATCH_SIZE})

            except torch.cuda.OutOfMemoryError:
                print(f"OOM at batch {batch_idx}, skipping...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue

    if not all_audio_embeddings:
        print("No valid batches processed!")
        return RetrievalMetrics(
            acc=0.0,
            mrr=0.0,
            recall_at_k={k: 0.0 for k in RETRIEVAL_K_VALUES},
            precision_at_k={k: 0.0 for k in RETRIEVAL_K_VALUES}
        )

    # Concatenate all embeddings
    audio_embeddings = torch.cat(all_audio_embeddings, dim=0)
    text_embeddings = torch.cat(all_text_embeddings, dim=0)

    print(f"Computing retrieval metrics for {audio_embeddings.shape[0]} samples...")

    # Compute metrics
    metrics = compute_retrieval_metrics(audio_embeddings, text_embeddings, RETRIEVAL_K_VALUES)

    return metrics


def print_metrics(metrics: RetrievalMetrics) -> None:
    """Print evaluation metrics in a formatted way."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    print(f"Accuracy: {metrics.acc:.4f}")
    print(f"MRR: {metrics.mrr:.4f}")

    print("\nRecall@k:")
    for k in sorted(metrics.recall_at_k.keys()):
        print(f"  Recall@{k}:  {metrics.recall_at_k[k]:.4f}")

    print("\nPrecision@k:")
    for k in sorted(metrics.precision_at_k.keys()):
        print(f"  Precision@{k}: {metrics.precision_at_k[k]:.4f}")

    print("="*60)


def load_checkpoint_and_model(checkpoint_path: Path, device: torch.device) -> Tuple[ClapModel, AutoProcessor]:
    """
    Load model and processor from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint directory
        device: Device to load model on

    Returns:
        Tuple of (model, processor)
    """
    print(f"Loading checkpoint from: {checkpoint_path}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    # Load processor
    processor = AutoProcessor.from_pretrained(checkpoint_path)
    processor = disable_fe_cap(processor)

    # Configure tokenizer for longer context
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        try:
            tokenizer.model_max_length = max(int(TEXT_TARGET_MAX_LEN_CAP), int(getattr(tokenizer, "model_max_length", TEXT_START_MAX_LEN)))
        except Exception:
            tokenizer.model_max_length = TEXT_TARGET_MAX_LEN_CAP

    # Load model
    model = ClapModel.from_pretrained(checkpoint_path)

    # Apply custom modifications (same as in clap.py)
    model = enable_audio_dynamic_context(model)
    model = enable_text_pos_interpolation(model)
    model = model.to(device)

    print(f"Model loaded successfully on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, processor


def load_song_describer_dataset(split_size: int = EVAL_SPLIT_SIZE) -> Tuple[DataLoader, Dataset]:
    """
    Load song-describer dataset for evaluation.

    Args:
        split_size: Number of samples to use for evaluation

    Returns:
        Tuple of (dataloader, dataset)
    """
    print(f"Loading song-describer dataset (split_size={split_size})...")

    # Create cache directories
    HF_DATASETS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset
    try:
        raw_dataset = load_dataset(
            SONG_DESCRIBER_DATASET_ID,
            split="train",
            cache_dir=str(HF_DATASETS_CACHE_DIR),
        )

        # Take a subset for evaluation
        if split_size and len(raw_dataset) > split_size:
            raw_dataset = raw_dataset.select(range(split_size))
            print(f"Using subset of {split_size} samples for evaluation")
        else:
            print(f"Using full dataset of {len(raw_dataset)} samples")

        # Wrap in custom dataset
        dataset = AudioTextDataset(raw_dataset, cache_dir=AUDIO_CACHE_DIR)

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=EVAL_BATCH_SIZE,
            shuffle=False,
            collate_fn=lambda batch: collate_batch(batch, processor=AutoProcessor.from_pretrained(MODEL_NAME), max_audio_seconds=CLIP_SECONDS, max_text_len=TEXT_TARGET_MAX_LEN_CAP),
        )

        print(f"Dataset loaded successfully: {len(dataset)} samples")
        return dataloader, dataset

    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise


def main() -> None:
    """Main evaluation function."""
    print("Starting CLAP model evaluation on song-describer dataset")
    print("-" * 60)

    # Set environment variables
    if HF_ENDPOINT_OVERRIDE and not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = HF_ENDPOINT_OVERRIDE
        print(f"HF_ENDPOINT set to {HF_ENDPOINT_OVERRIDE}")
    if HF_REQUEST_TIMEOUT and not os.environ.get("HF_HUB_REQUESTS_TIMEOUT"):
        os.environ["HF_HUB_REQUESTS_TIMEOUT"] = HF_REQUEST_TIMEOUT
    if not os.environ.get("HF_HUB_DISABLE_HF_TRANSFER"):
        os.environ["HF_HUB_DISABLE_HF_TRANSFER"] = "1"
    if not os.environ.get("HF_HUB_ENABLE_HF_TRANSFER"):
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

    # Set random seed
    set_seed(RANDOM_SEED)

    # Get device
    device = get_device()
    print(f"Using device: {device}")

    # Load checkpoint
    try:
        model, processor = load_checkpoint_and_model(LATEST_CHECKPOINT_DIR, device)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure you have trained the model and have a checkpoint available.")
        return

    # Load dataset
    try:
        # We need to create the dataloader after we have the processor
        print("Loading song-describer dataset...")

        # Create cache directories
        HF_DATASETS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # Load dataset
        raw_dataset = load_dataset(
            SONG_DESCRIBER_DATASET_ID,
            split="train",
            cache_dir=str(HF_DATASETS_CACHE_DIR),
        )

        # Take a subset for evaluation
        if EVAL_SPLIT_SIZE and len(raw_dataset) > EVAL_SPLIT_SIZE:
            raw_dataset = raw_dataset.select(range(EVAL_SPLIT_SIZE))
            print(f"Using subset of {EVAL_SPLIT_SIZE} samples for evaluation")
        else:
            print(f"Using full dataset of {len(raw_dataset)} samples")

        # Wrap in custom dataset
        dataset = AudioTextDataset(raw_dataset, cache_dir=AUDIO_CACHE_DIR)

        # Create dataloader with the correct processor
        dataloader = DataLoader(
            dataset,
            batch_size=EVAL_BATCH_SIZE,
            shuffle=False,
            collate_fn=lambda batch: collate_batch(batch, processor=processor, max_audio_seconds=CLIP_SECONDS, max_text_len=TEXT_TARGET_MAX_LEN_CAP),
        )

        print(f"Dataset loaded successfully: {len(dataset)} samples")

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Run evaluation
    print("\nStarting evaluation...")
    try:
        metrics = evaluate_model(model, processor, dataloader, device, CLIP_SECONDS, TEXT_TARGET_MAX_LEN_CAP)
        print_metrics(metrics)

        # Save results to file
        results_file = Path("evaluation_results.json")
        results = {
            "accuracy": metrics.acc,
            "mrr": metrics.mrr,
            "recall_at_k": metrics.recall_at_k,
            "precision_at_k": metrics.precision_at_k,
            "num_samples": len(dataset),
            "model_checkpoint": str(LATEST_CHECKPOINT_DIR),
            "max_audio_seconds": CLIP_SECONDS,
            "max_text_length": TEXT_TARGET_MAX_LEN_CAP,
        }

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {results_file}")

    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()