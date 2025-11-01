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
from typing import Dict, List, Optional
import librosa
import mlflow
import numpy as np
import requests
import torch
import torch.nn.functional as F
from datasets import load_dataset, Audio
from huggingface_hub import hf_hub_download
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, ClapModel
from transformers.modeling_outputs import BaseModelOutputWithPooling
from tqdm.auto import tqdm


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

MODEL_NAME = "laion/clap-htsat-unfused"

JAMENDO_DATASET_ID = "amaai-lab/JamendoMaxCaps"
SONG_DESCRIBER_DATASET_ID = "renumics/song-describer-dataset"

JAMENDO_TOTAL_SHARDS = 2272
JAMENDO_START_SHARD = 11 * 46 - 1 # Start from validation shard
JAMENDO_SHARD_PAD = 5
TRAIN_SHARDS_PER_CYCLE = 10
EVAL_SHARDS_PER_CYCLE = 1

RANDOM_SEED = 42
AUDIO_SAMPLING_RATE = 48_000
CLIP_SECONDS = 240

TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 1
GRADIENT_ACCUMULATION_STEPS = 2

CONTEXT_GROWTH_FRACTION = 0.9
TEXT_START_MAX_LEN = 512
TEXT_TARGET_MAX_LEN_CAP = 512
AUDIO_TARGET_MAX_SECONDS_CAP = 300
SCHEDULE_SCAN_LIMIT = 1024

MLFLOW_TRACKING_URI = "file:mlruns"
MLFLOW_EXPERIMENT_NAME = "clap_jamendo"
MLFLOW_RUN_NAME = "clap_jamendo_run"

CHECKPOINT_DIR = Path("checkpoints")
BEST_CHECKPOINT_DIR = CHECKPOINT_DIR / "best"
LATEST_CHECKPOINT_DIR = CHECKPOINT_DIR / "latest"
RESUME_TRAINING = True

SHARD_CACHE_DIR = Path("data/jamendo_shards")
AUDIO_CACHE_DIR = Path("data/audio_cache")
HF_ENDPOINT_OVERRIDE = "https://hf-mirror.com"
HF_REQUEST_TIMEOUT = "60"

HF_DATASETS_CACHE_DIR = SHARD_CACHE_DIR / "hf_datasets_cache"

# -----------------------------------------------------------------------------
# Caption index for Jamendo
# -----------------------------------------------------------------------------
_CAPTION_BY_ID: Optional[Dict[str, str]] = None


# -----------------------------------------------------------------------------
# Context schedule state
# -----------------------------------------------------------------------------

CURRENT_AUDIO_SECONDS = CLIP_SECONDS
CURRENT_TEXT_MAX_LEN = TEXT_START_MAX_LEN
_SCHEDULE_COLLATE_STEP = 0
_SCHEDULE_ACTIVE = True
SCHEDULE: Optional["ContextSchedule"] = None


@dataclass
class ContextSchedule:
    audio_start_s: int
    audio_target_s: int
    text_start_tok: int
    text_target_tok: int
    growth_steps: int

    def value_at(self, step: int) -> tuple[int, int]:
        if self.growth_steps <= 0:
            frac = 1.0
        else:
            frac = min(1.0, step / float(self.growth_steps))
        audio_seconds = int(
            round(self.audio_start_s + frac * (self.audio_target_s - self.audio_start_s))
        )
        text_tokens = int(
            round(self.text_start_tok + frac * (self.text_target_tok - self.text_start_tok))
        )
        return max(1, audio_seconds), max(8, text_tokens)


@dataclass
class Metrics:
    loss: float
    accuracy: float


class MissingAudioError(Exception):
    """Raised when an audio asset cannot be resolved."""


# -----------------------------------------------------------------------------
# Utilities
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



# --- Minimal helper to enable dynamic audio context for audio tower ---
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
    # Keep ClapAudioPatchEmbed's bookkeeping in sync with dynamic H×W
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


def launch_mlflow_ui(port: int = 5050) -> Optional[subprocess.Popen]:
    tracking_uri = Path(mlflow.get_tracking_uri()[5:]).resolve()
    tracking_uri.mkdir(parents=True, exist_ok=True)
    cmd = [
        "mlflow",
        "ui",
        "--backend-store-uri",
        str(tracking_uri),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"MLflow UI running at http://127.0.0.1:{port}")
        return proc
    except FileNotFoundError:
        print("mlflow CLI not found; UI not launched.")
    return None


def _download_to_cache(url: str) -> str:
    AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    target = AUDIO_CACHE_DIR / os.path.basename(url.split("?")[0])
    if target.exists() and target.stat().st_size > 0:
        return str(target)
    last_exc: Optional[Exception] = None
    for attempt in range(1, 4):
        try:
            with requests.get(url, timeout=60, stream=True) as r:
                r.raise_for_status()
                with open(target, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            return str(target)
        except Exception as exc:
            last_exc = exc
            if target.exists():
                try:
                    target.unlink()
                except OSError:
                    pass
    raise MissingAudioError(f"Failed to download audio from {url}: {last_exc}")


def _jamendo_shard_filename(index: int) -> str:
    if index < 0 or index >= JAMENDO_TOTAL_SHARDS:
        raise ValueError(f"Shard index {index} out of range (0-{JAMENDO_TOTAL_SHARDS-1})")
    return f"train-{index:0{JAMENDO_SHARD_PAD}d}-of-{JAMENDO_TOTAL_SHARDS:0{JAMENDO_SHARD_PAD}d}.parquet"


def _download_shard(filename: str, base_dir: Path) -> str:
    base_dir.mkdir(parents=True, exist_ok=True)
    return hf_hub_download(
        repo_id=JAMENDO_DATASET_ID,
        filename=f"data/{filename}",
        repo_type="dataset",
        local_dir=str(base_dir),
    )


# --- Jamendo captions index helpers ---
def _download_captions_jsonl() -> str:
    # Try both possible paths inside the HF dataset repo
    caps_dir = SHARD_CACHE_DIR / "caps"
    caps_dir.mkdir(parents=True, exist_ok=True)
    print("Fetching Jamendo captions JSONL index...")
    try:
        return hf_hub_download(repo_id=JAMENDO_DATASET_ID, filename="final_caption30sec.jsonl", repo_type="dataset", local_dir=str(caps_dir))
    except Exception:
        return hf_hub_download(repo_id=JAMENDO_DATASET_ID, filename="data/final_caption30sec.jsonl", repo_type="dataset", local_dir=str(caps_dir))


def _ensure_captions_index() -> None:
    global _CAPTION_BY_ID
    if _CAPTION_BY_ID is not None:
        return
    path = _download_captions_jsonl()
    idx: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                cid = str(obj.get("id") or "").strip()
                cap = obj.get("caption")
                if cid and isinstance(cap, str) and cid not in idx:
                    idx[cid] = cap
            except Exception:
                continue
    _CAPTION_BY_ID = idx


def _hf_cache_dir(*parts: str) -> Path:
    path = HF_DATASETS_CACHE_DIR.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_jamendo_shard(index: int, cache_subdir: str) -> tuple[AudioTextDataset, str, Path, Path]:
    filename = _jamendo_shard_filename(index)
    local_path = _download_shard(filename, SHARD_CACHE_DIR / cache_subdir)
    hf_cache = _hf_cache_dir(cache_subdir, Path(filename).stem)
    ds_dict = load_dataset(
        "parquet",
        data_files={"train": [local_path]},
        streaming=False,
        cache_dir=str(hf_cache),
    )
    ds_dict = ds_dict.cast_column("audio", Audio(decode=False))
    cache_dir = AUDIO_CACHE_DIR / cache_subdir / Path(filename).stem
    dataset = AudioTextDataset(ds_dict["train"], cache_dir=cache_dir)
    return dataset, local_path, cache_dir, hf_cache


def _resolve_audio_path(path_str: str) -> str:
    candidate = path_str.strip()
    if os.path.exists(candidate):
        return candidate
    if candidate.startswith("http://") or candidate.startswith("https://"):
        return _download_to_cache(candidate)
    raise MissingAudioError(f"Audio path not accessible: {path_str}")


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

        # 2) Bytes payload (prefer bytes over path: paths may be remote/unavailable)
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

        # 3) Path on disk or URL
        p = audio_entry.get("path") or audio_entry.get("filepath")
        if p:
            source = _resolve_audio_path(str(p))
            arr, sr = _librosa_decode(source)
            if arr.ndim > 1:
                arr = arr.mean(axis=-1)
            return arr.astype(np.float32), sr

        raise MissingAudioError("Audio dict has no decodable data")

    elif isinstance(audio_entry, str):
        source = _resolve_audio_path(audio_entry)
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


def _infer_audio_duration(audio_entry) -> Optional[float]:
    try:
        arr, sr = _load_audio_entry(audio_entry)
        if arr.size == 0:
            return None
        return arr.shape[0] / float(sr)
    except MissingAudioError:
        return None


def _extract_caption(row: Dict) -> str:
    # 1) Try Jamendo captions indexed by track id (from final_caption30sec.jsonl)
    try:
        _ensure_captions_index()
        audio = row.get("audio")
        track_id = None
        if isinstance(audio, dict):
            p = audio.get("path") or audio.get("filepath")
            if isinstance(p, str):
                base = os.path.basename(p)
                track_id = base.split(".")[0]
        if not track_id:
            rid = row.get("id")
            if isinstance(rid, (str, int)):
                track_id = str(rid)
        if track_id and _CAPTION_BY_ID:
            cap = _CAPTION_BY_ID.get(str(track_id))
            if isinstance(cap, str) and cap.strip():
                return cap
    except Exception:
        pass
    # 2) Fallbacks present in some rows
    for key in ("pseudo_caption", "description", "caption", "text"):
        value = row.get(key)
        if value:
            return str(value)
    # 3) Final fallback
    return "instrumental music track"


def _schedule_targets_from_samples(
    durations: List[float],
    texts: List[str],
    tokenizer,
) -> tuple[int, int]:
    if durations:
        audio_target = int(
            min(
                AUDIO_TARGET_MAX_SECONDS_CAP,
                max(CLIP_SECONDS, round(float(np.percentile(durations, 95)))),
            )
        )
    else:
        audio_target = CLIP_SECONDS

    text_target = TEXT_START_MAX_LEN
    if tokenizer is not None and texts:
        lengths: List[int] = []
        for text in texts[:SCHEDULE_SCAN_LIMIT]:
            try:
                encoded = tokenizer(
                    text,
                    add_special_tokens=True,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )
                lengths.append(len(encoded["input_ids"]))
            except Exception:
                continue
        if lengths:
            text_target = int(
                min(
                    TEXT_TARGET_MAX_LEN_CAP,
                    max(TEXT_START_MAX_LEN, max(lengths)),
                )
            )
    return audio_target, text_target


def _schedule_next_limits() -> None:
    global _SCHEDULE_COLLATE_STEP, CURRENT_AUDIO_SECONDS, CURRENT_TEXT_MAX_LEN
    if not _SCHEDULE_ACTIVE or SCHEDULE is None:
        return
    _SCHEDULE_COLLATE_STEP += 1
    audio_seconds, text_tokens = SCHEDULE.value_at(_SCHEDULE_COLLATE_STEP)
    CURRENT_AUDIO_SECONDS = min(audio_seconds, AUDIO_TARGET_MAX_SECONDS_CAP)
    CURRENT_TEXT_MAX_LEN = min(text_tokens, TEXT_TARGET_MAX_LEN_CAP)


def _enter_eval_limits() -> tuple[bool, int, int, int]:
    global _SCHEDULE_ACTIVE, CURRENT_AUDIO_SECONDS, CURRENT_TEXT_MAX_LEN, _SCHEDULE_COLLATE_STEP
    prev_state = (_SCHEDULE_ACTIVE, CURRENT_AUDIO_SECONDS, CURRENT_TEXT_MAX_LEN, _SCHEDULE_COLLATE_STEP)
    _SCHEDULE_ACTIVE = False
    if SCHEDULE is not None:
        CURRENT_AUDIO_SECONDS = min(SCHEDULE.audio_target_s, AUDIO_TARGET_MAX_SECONDS_CAP)
        CURRENT_TEXT_MAX_LEN = min(SCHEDULE.text_target_tok, TEXT_TARGET_MAX_LEN_CAP)
    return prev_state


def _exit_eval_limits(state: tuple[bool, int, int, int]) -> None:
    global _SCHEDULE_ACTIVE, CURRENT_AUDIO_SECONDS, CURRENT_TEXT_MAX_LEN, _SCHEDULE_COLLATE_STEP
    active, audio_seconds, text_tokens, collate_step = state
    _SCHEDULE_ACTIVE = active
    CURRENT_AUDIO_SECONDS = audio_seconds
    CURRENT_TEXT_MAX_LEN = text_tokens
    _SCHEDULE_COLLATE_STEP = collate_step


# -----------------------------------------------------------------------------
# Dataset wrappers
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


def collate_batch(batch: List[Dict[str, object]], processor: AutoProcessor) -> Optional[Dict[str, torch.Tensor]]:
    _schedule_next_limits()
    target_len = int(AUDIO_SAMPLING_RATE * CURRENT_AUDIO_SECONDS)

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
            max_start = arr.shape[0] - target_len
            if _SCHEDULE_ACTIVE and max_start > 0:
                start = np.random.randint(0, max_start + 1)
            else:
                start = max(0, max_start // 2)
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
        max_length=CURRENT_TEXT_MAX_LEN,
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
# Data preparation
# -----------------------------------------------------------------------------


def _estimate_schedule_targets_from_shards(indices: List[int], processor: AutoProcessor) -> tuple[int, int, int]:
    if not indices:
        tokenizer = getattr(processor, "tokenizer", None)
        audio_target, text_target = _schedule_targets_from_samples([], [], tokenizer)
        return audio_target, text_target, 1

    scan_dir = SHARD_CACHE_DIR / "scan"
    local_path = _download_shard(_jamendo_shard_filename(indices[0]), scan_dir)
    hf_cache = _hf_cache_dir("scan", Path(local_path).stem)
    ds_dict = load_dataset(
        "parquet",
        data_files={"train": [local_path]},
        streaming=True,
        cache_dir=str(hf_cache),
    )

    ds_dict = ds_dict.cast_column("audio", Audio(decode=False))

    stream = ds_dict["train"]
    durations: List[float] = []
    texts: List[str] = []
    sample_count = 0
    for row in stream:
        sample_count += 1
        audio_entry = row.get("audio")
        duration = _infer_audio_duration(audio_entry)
        if duration:
            durations.append(duration)
        texts.append(_extract_caption(row))
        if sample_count >= SCHEDULE_SCAN_LIMIT:
            break
    try:
        os.remove(local_path)
    except OSError:
        pass
    shutil.rmtree(hf_cache, ignore_errors=True)

    tokenizer = getattr(processor, "tokenizer", None)
    audio_target, text_target = _schedule_targets_from_samples(durations, texts, tokenizer)
    return audio_target, text_target, max(1, sample_count)


def prepare_training(processor: AutoProcessor) -> Dict[str, object]:
    global SCHEDULE, CURRENT_AUDIO_SECONDS, CURRENT_TEXT_MAX_LEN, _SCHEDULE_COLLATE_STEP, _SCHEDULE_ACTIVE

    _ensure_captions_index()

    start_shard = min(max(JAMENDO_START_SHARD, 0), max(0, JAMENDO_TOTAL_SHARDS - 1))
    audio_target, text_target, sample_count = _estimate_schedule_targets_from_shards([start_shard], processor)
    approx_batches_per_shard = max(1, sample_count // max(1, TRAIN_BATCH_SIZE))
    planned_total = max(1, approx_batches_per_shard * max(1, TRAIN_SHARDS_PER_CYCLE) * NUM_EPOCHS)

    SCHEDULE = ContextSchedule(
        audio_start_s=CLIP_SECONDS,
        audio_target_s=int(audio_target),
        text_start_tok=TEXT_START_MAX_LEN,
        text_target_tok=int(text_target),
        growth_steps=planned_total * 2,
    )
    CURRENT_AUDIO_SECONDS = CLIP_SECONDS
    CURRENT_TEXT_MAX_LEN = TEXT_START_MAX_LEN
    _SCHEDULE_COLLATE_STEP = 0
    _SCHEDULE_ACTIVE = True

    song_cache = _hf_cache_dir("song-describer")
    song_eval_raw = load_dataset(
        SONG_DESCRIBER_DATASET_ID,
        split="train",
        cache_dir=str(song_cache),
    )
    song_eval_dataset = AudioTextDataset(song_eval_raw)
    song_eval_loader = DataLoader(
        song_eval_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda batch: collate_batch(batch, processor),
    )

    return {
        "song_eval_loader": song_eval_loader,
    }


# -----------------------------------------------------------------------------
# Training & evaluation
# -----------------------------------------------------------------------------


def compute_accuracy(logits: torch.Tensor) -> float:
    predictions = logits.argmax(dim=-1)
    targets = torch.arange(predictions.shape[0], device=predictions.device)
    return (predictions == targets).float().mean().item()

def evaluate(
    model: ClapModel,
    loader: Optional[DataLoader],
    device: torch.device,
    desc: str,
) -> Metrics:
    if loader is None:
        return Metrics(loss=0.0, accuracy=0.0)

    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    steps = 0
    progress = tqdm(loader, desc=desc, leave=False)
    with torch.no_grad():
        for batch in progress:
            if batch is None:
                continue
            batch = {k: v.to(device) for k, v in batch.items() if v is not None}
            try:
                outputs = model(**batch, return_loss=True)
            except torch.cuda.OutOfMemoryError:
                print(f"OOM during evaluation '{desc}'; skipping batch")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            total_loss += outputs.loss.item()
            total_accuracy += compute_accuracy(outputs.logits_per_text)
            steps += 1
            progress.set_postfix(
                loss=f"{total_loss / max(1, steps):.4f}",
                acc=f"{total_accuracy / max(1, steps):.4f}",
            )
    progress.close()
    if steps == 0:
        return Metrics(loss=0.0, accuracy=0.0)
    return Metrics(loss=total_loss / steps, accuracy=total_accuracy / steps)


def save_checkpoint(model: ClapModel, processor: AutoProcessor, target_dir: Path) -> None:
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(target_dir)
    processor.save_pretrained(target_dir)
    print(f"Saved checkpoint: {target_dir}")


def _evaluate_jamendo_shards(
    model: ClapModel,
    processor: AutoProcessor,
    indices: List[int],
    device: torch.device,
) -> Metrics:
    total_loss = 0.0
    total_accuracy = 0.0
    steps = 0
    for idx in indices:
        dataset, local_path, cache_dir, hf_cache = _load_jamendo_shard(idx, "eval")
        loader = DataLoader(
            dataset,
            batch_size=EVAL_BATCH_SIZE,
            shuffle=False,
            collate_fn=lambda batch: collate_batch(batch, processor),
        )
        for batch in loader:
            if batch is None:
                continue
            batch = {k: v.to(device) for k, v in batch.items() if v is not None}
            try:
                outputs = model(**batch, return_loss=True)
            except torch.cuda.OutOfMemoryError:
                print(f"OOM during Jamendo eval shard {idx:05d}; skipping batch")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            total_loss += outputs.loss.item()
            total_accuracy += compute_accuracy(outputs.logits_per_text)
            steps += 1
        print(f"Evaluated shard {idx:05d}; removing {local_path}")
        try:
            os.remove(local_path)
        except OSError:
            pass
        if cache_dir is not None:
            shutil.rmtree(cache_dir, ignore_errors=True)
        shutil.rmtree(hf_cache, ignore_errors=True)
        loader = None
        del dataset
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if steps == 0:
        return Metrics(loss=0.0, accuracy=0.0)
    return Metrics(loss=total_loss / steps, accuracy=total_accuracy / steps)


def train_model(
    model: ClapModel,
    processor: AutoProcessor,
    context: Dict[str, object],
    device: torch.device,
    start_shard: int = 0,
) -> Metrics:
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    best_metrics = Metrics(loss=float("inf"), accuracy=0.0)
    global_step = 0
    processed_train_shards = 0

    song_eval_loader: Optional[DataLoader] = context["song_eval_loader"]

    safe_start_shard = min(max(start_shard, 0), JAMENDO_TOTAL_SHARDS)
    if safe_start_shard > 0:
        print(f"Starting training from shard index {safe_start_shard}")

    cycle_len = TRAIN_SHARDS_PER_CYCLE + EVAL_SHARDS_PER_CYCLE
    cycle_offset = safe_start_shard % cycle_len if cycle_len > 0 else 0

    def run_eval_cycle(eval_indices: List[int]) -> Metrics:
        nonlocal best_metrics, global_step
        fixed_eval_indices = [JAMENDO_TOTAL_SHARDS - 1] if JAMENDO_TOTAL_SHARDS > 0 else []
        if not fixed_eval_indices:
            return Metrics(0.0, 0.0)
        prev_state = _enter_eval_limits()
        jamendo_metrics = _evaluate_jamendo_shards(model, processor, fixed_eval_indices, device)
        song_metrics = (
            evaluate(model, song_eval_loader, device, desc="Song-Describer eval")
            if song_eval_loader is not None
            else Metrics(0.0, 0.0)
        )
        _exit_eval_limits(prev_state)

        mlflow.log_metric("jamendo_val_loss", jamendo_metrics.loss, step=global_step)
        mlflow.log_metric("jamendo_val_accuracy", jamendo_metrics.accuracy, step=global_step)
        mlflow.log_metric("song_describer_val_loss", song_metrics.loss, step=global_step)
        mlflow.log_metric("song_describer_val_accuracy", song_metrics.accuracy, step=global_step)

        target_metrics = jamendo_metrics if fixed_eval_indices else song_metrics
        if target_metrics.loss < best_metrics.loss:
            best_metrics = target_metrics
            save_checkpoint(model, processor, BEST_CHECKPOINT_DIR)
        return jamendo_metrics

    if (
        EVAL_SHARDS_PER_CYCLE
        and cycle_len > 0
        and cycle_offset >= TRAIN_SHARDS_PER_CYCLE
        and safe_start_shard < JAMENDO_TOTAL_SHARDS
    ):
        eval_indices = list(
            range(
                safe_start_shard,
                min(safe_start_shard + EVAL_SHARDS_PER_CYCLE, JAMENDO_TOTAL_SHARDS),
            )
        )
        print(f"Resuming inside evaluation phase; running eval on shards {eval_indices}")
        run_eval_cycle(eval_indices)
        safe_start_shard = min(JAMENDO_TOTAL_SHARDS, eval_indices[-1] + 1)
        cycle_offset = 0

    for epoch in range(NUM_EPOCHS):
        shard_index = safe_start_shard if epoch == 0 else 0
        cycle = 0
        while shard_index < JAMENDO_TOTAL_SHARDS:
            train_indices = list(range(shard_index, min(shard_index + TRAIN_SHARDS_PER_CYCLE, JAMENDO_TOTAL_SHARDS)))
            eval_start = min(JAMENDO_TOTAL_SHARDS, train_indices[-1] + 1) if train_indices else shard_index
            eval_indices = list(range(eval_start, min(eval_start + EVAL_SHARDS_PER_CYCLE, JAMENDO_TOTAL_SHARDS))) if EVAL_SHARDS_PER_CYCLE else []

            shard_bar = tqdm(total=len(train_indices), desc=f"Epoch {epoch + 1} | Cycle {cycle + 1} Training shards", leave=True)
            for idx in train_indices:
                model.train()
                dataset, local_path, cache_dir, hf_cache = _load_jamendo_shard(idx, "train")
                loader = DataLoader(
                    dataset,
                    batch_size=TRAIN_BATCH_SIZE,
                    shuffle=True,
                    collate_fn=lambda batch: collate_batch(batch, processor),
                )
                batch_progress = tqdm(loader, desc=f"Shard {idx:05d}", leave=True)
                running_loss = 0.0
                steps = 0
                for batch in batch_progress:
                    if batch is None:
                        continue
                    batch = {k: v.to(device) for k, v in batch.items() if v is not None}
                    with torch.no_grad():
                        if hasattr(model, "logit_scale"):
                            model.logit_scale.clamp_(0.0, 4.6052)
                    try:
                        outputs = model(**batch, return_loss=True)
                    except torch.cuda.OutOfMemoryError:
                        print(f"OOM during training shard {idx:05d} forward; skipping batch")
                        optimizer.zero_grad(set_to_none=True)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS
                    try:
                        loss.backward()
                    except torch.cuda.OutOfMemoryError:
                        print(f"OOM during training shard {idx:05d} backward; skipping batch")
                        optimizer.zero_grad(set_to_none=True)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    if (steps + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                    running_loss += outputs.loss.item()
                    steps += 1
                    global_step += 1
                    mlflow.log_metric("train_loss_step", outputs.loss.item(), step=global_step)
                    mlflow.log_metric("allowed_audio_seconds", CURRENT_AUDIO_SECONDS, step=global_step)
                    mlflow.log_metric("allowed_text_tokens", CURRENT_TEXT_MAX_LEN, step=global_step)
                    batch_progress.set_postfix(loss=f"{running_loss / max(1, steps):.4f}")
                batch_progress.close()
                if steps > 0:
                    shard_loss = running_loss / steps
                    mlflow.log_metric("train_loss_shard", shard_loss, step=global_step)
                print(f"Processed shard {idx:05d}; removing {local_path}")
                try:
                    os.remove(local_path)
                except OSError:
                    pass
                if cache_dir is not None:
                    shutil.rmtree(cache_dir, ignore_errors=True)
                shutil.rmtree(hf_cache, ignore_errors=True)
                loader = None
                del dataset
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                processed_train_shards += 1
                if processed_train_shards % 5 == 0:
                    save_checkpoint(model, processor, LATEST_CHECKPOINT_DIR)
                shard_bar.update(1)
            shard_bar.close()

            if eval_indices:
                run_eval_cycle(eval_indices)
                save_checkpoint(model, processor, LATEST_CHECKPOINT_DIR)

            shard_index = eval_indices[-1] + 1 if eval_indices else train_indices[-1] + 1
            cycle += 1

    # Final evaluation on remaining eval shards (if any) and Song-Describer
    prev_state = _enter_eval_limits()
    jamendo_metrics = Metrics(0.0, 0.0)
    song_metrics = evaluate(model, song_eval_loader, device, desc="Song-Describer eval (final)") if song_eval_loader is not None else Metrics(0.0, 0.0)
    _exit_eval_limits(prev_state)

    mlflow.log_metric("jamendo_val_loss_final", jamendo_metrics.loss, step=NUM_EPOCHS)
    mlflow.log_metric("jamendo_val_accuracy_final", jamendo_metrics.accuracy, step=NUM_EPOCHS)
    mlflow.log_metric("song_describer_val_loss_final", song_metrics.loss, step=NUM_EPOCHS)
    mlflow.log_metric("song_describer_val_accuracy_final", song_metrics.accuracy, step=NUM_EPOCHS)

    target_metrics = song_metrics
    save_checkpoint(model, processor, LATEST_CHECKPOINT_DIR)
    if target_metrics.loss < best_metrics.loss:
        best_metrics = target_metrics
        save_checkpoint(model, processor, BEST_CHECKPOINT_DIR)

    return best_metrics


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


def main() -> None:
    if HF_ENDPOINT_OVERRIDE and not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = HF_ENDPOINT_OVERRIDE
        print(f"HF_ENDPOINT set to {HF_ENDPOINT_OVERRIDE}")
    if HF_REQUEST_TIMEOUT and not os.environ.get("HF_HUB_REQUESTS_TIMEOUT"):
        os.environ["HF_HUB_REQUESTS_TIMEOUT"] = HF_REQUEST_TIMEOUT
    if not os.environ.get("HF_HUB_DISABLE_HF_TRANSFER"):
        os.environ["HF_HUB_DISABLE_HF_TRANSFER"] = "1"
    if not os.environ.get("HF_HUB_ENABLE_HF_TRANSFER"):
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

    set_seed(RANDOM_SEED)
    device = get_device()

    HF_DATASETS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_source: str | Path = MODEL_NAME
    resumed_from: Optional[Path] = None
    if RESUME_TRAINING:
        if LATEST_CHECKPOINT_DIR.exists():
            checkpoint_source = LATEST_CHECKPOINT_DIR
            resumed_from = LATEST_CHECKPOINT_DIR
        elif BEST_CHECKPOINT_DIR.exists():
            checkpoint_source = BEST_CHECKPOINT_DIR
            resumed_from = BEST_CHECKPOINT_DIR
    if resumed_from is not None:
        print(f"Resuming from checkpoint: {resumed_from}")

    processor = AutoProcessor.from_pretrained(checkpoint_source)
    processor = disable_fe_cap(processor)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        try:
            tokenizer.model_max_length = max(int(TEXT_TARGET_MAX_LEN_CAP), int(getattr(tokenizer, "model_max_length", TEXT_START_MAX_LEN)))
        except Exception:
            tokenizer.model_max_length = TEXT_TARGET_MAX_LEN_CAP

    model = ClapModel.from_pretrained(checkpoint_source)
    model = enable_audio_dynamic_context(model)
    model = enable_text_pos_interpolation(model)
    model = model.to(device)

    _ensure_captions_index()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    launch_mlflow_ui()

    with mlflow.start_run(run_name=MLFLOW_RUN_NAME):
        context = prepare_training(processor)
        mlflow.log_params(
            {
                "model_name": MODEL_NAME,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "train_batch_size": TRAIN_BATCH_SIZE,
                "eval_batch_size": EVAL_BATCH_SIZE,
                "context_growth_fraction": CONTEXT_GROWTH_FRACTION,
                "train_shards_per_cycle": TRAIN_SHARDS_PER_CYCLE,
                "eval_shards_per_cycle": EVAL_SHARDS_PER_CYCLE,
            }
        )
        best_metrics = train_model(model, processor, context, device, start_shard=JAMENDO_START_SHARD)
        mlflow.log_metric("best_val_loss", best_metrics.loss, step=NUM_EPOCHS)
        mlflow.log_metric("best_val_accuracy", best_metrics.accuracy, step=NUM_EPOCHS)

if __name__ == "__main__":
    main()
