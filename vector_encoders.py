import os
from pathlib import Path
from typing import List

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor, AutoTokenizer, ClapModel

device = torch.device(os.getenv("VEC_DEVICE", "cpu"))
clap_checkpoint_dir = Path("checkpoints/clap")
biencoder_checkpoint_dir = Path("checkpoints/biencoder")
fusion_checkpoint_path = Path("checkpoints/fusion/fusion_best.pth")

biencoder_tokenizer = None
biencoder_model = None
clap_processor = None
clap_model = None
fusion_model = None


class FusionEncoder(torch.nn.Module):
    def __init__(self, in_dim: int, fused_dim: int = 256, hidden: int = 1024, p_drop: float = 0.2):
        super().__init__()
        self.in_dim = in_dim
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p_drop),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p_drop),
            torch.nn.Linear(hidden, fused_dim),
        )
        self.norm = torch.nn.LayerNorm(fused_dim)

    def forward(self, audio_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([audio_emb, text_emb], dim=-1)
        if x.shape[-1] != self.in_dim:
            if x.shape[-1] < self.in_dim:
                pad = self.in_dim - x.shape[-1]
                x = torch.nn.functional.pad(x, (0, pad))
            else:
                x = x[..., : self.in_dim]
        x = self.mlp(x)
        x = self.norm(x)
        return F.normalize(x, p=2, dim=-1)


def load_biencoder():
    global biencoder_tokenizer, biencoder_model

    if biencoder_tokenizer is not None and biencoder_model is not None:
        return biencoder_tokenizer, biencoder_model

    if not biencoder_checkpoint_dir.exists():
        raise FileNotFoundError(f"Bi-encoder checkpoint not found at {biencoder_checkpoint_dir}")

    biencoder_tokenizer = AutoTokenizer.from_pretrained(biencoder_checkpoint_dir)
    biencoder_model = AutoModel.from_pretrained(biencoder_checkpoint_dir).to(device)
    biencoder_model.eval()
    return biencoder_tokenizer, biencoder_model


def load_clap():
    global clap_processor, clap_model

    if clap_processor is not None and clap_model is not None:
        return clap_processor, clap_model

    if not clap_checkpoint_dir.exists():
        raise FileNotFoundError(f"CLAP checkpoint not found at {clap_checkpoint_dir}")

    clap_processor = AutoProcessor.from_pretrained(clap_checkpoint_dir)
    clap_model = ClapModel.from_pretrained(clap_checkpoint_dir).to(device)
    clap_model.eval()
    return clap_processor, clap_model


def load_fusion() -> FusionEncoder:
    global fusion_model

    if fusion_model is not None:
        return fusion_model

    if not fusion_checkpoint_path.exists():
        raise FileNotFoundError(f"Fusion checkpoint not found at {fusion_checkpoint_path}")

    state_dict = torch.load(fusion_checkpoint_path, map_location="cpu")
    first_w = state_dict.get("mlp.0.weight")
    last_w = state_dict.get("mlp.6.weight")
    if first_w is None or last_w is None:
        raise RuntimeError("Unexpected fusion checkpoint format: missing mlp weights.")
    in_dim = first_w.shape[1]
    hidden = first_w.shape[0]
    fused_dim = last_w.shape[0]

    fusion_model = FusionEncoder(in_dim=in_dim, fused_dim=fused_dim, hidden=hidden, p_drop=0.2)
    fusion_model.load_state_dict(state_dict)
    fusion_model.to(device).eval()
    return fusion_model


@torch.no_grad()
def encode_text_biencoder(texts: List[str]) -> np.ndarray:
    tokenizer, model = load_biencoder()
    texts = [str(t) for t in texts]

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    outputs = model(**encoded)
    cls = outputs.last_hidden_state[:, 0, :]  # CLS-токен
    cls = F.normalize(cls, p=2, dim=-1)
    return cls.cpu().numpy().astype(np.float32)


@torch.no_grad()
def encode_audio_clap(path: Path) -> np.ndarray:
    processor, model = load_clap()

    audio, _ = librosa.load(str(path), sr=48_000, mono=True)
    max_samples = int(30.0 * 48_000)
    if len(audio) > max_samples:
        audio = audio[:max_samples]

    inputs = processor(
        audios=[audio],
        sampling_rate=48_000,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    features = model.get_audio_features(**inputs)
    features = F.normalize(features, p=2, dim=-1)
    return features[0].cpu().numpy().astype(np.float32)


@torch.no_grad()
def encode_text_clap(texts: List[str]) -> np.ndarray:
    processor, model = load_clap()
    texts = [str(t) for t in texts]
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    features = model.get_text_features(**inputs)
    features = F.normalize(features, p=2, dim=-1)
    return features.cpu().numpy().astype(np.float32)


@torch.no_grad()
def encode_fusion_query(texts: List[str]) -> np.ndarray:
    fusion = load_fusion()
    text_biencoder = encode_text_biencoder(texts)
    text_clap = encode_text_clap(texts)

    text_biencoder_t = torch.from_numpy(text_biencoder).to(device)
    text_clap_t = torch.from_numpy(text_clap).to(device)
    fused = fusion(text_clap_t, text_biencoder_t)
    return fused.cpu().numpy().astype(np.float32)


@torch.no_grad()
def encode_fusion_tracks(audio_vectors: np.ndarray, lyric_vectors: np.ndarray) -> np.ndarray:
    fusion = load_fusion()
    audio_np = np.asarray(audio_vectors, dtype=np.float32)
    lyrics_np = np.asarray(lyric_vectors, dtype=np.float32)

    if audio_np.shape[0] != lyrics_np.shape[0]:
        raise ValueError("audio_vectors and lyric_vectors must have the same length")

    fused_chunks = []
    batch_size = 256
    for start in range(0, audio_np.shape[0], batch_size):
        end = start + batch_size
        audio_t = torch.from_numpy(audio_np[start:end]).to(device)
        lyrics_t = torch.from_numpy(lyrics_np[start:end]).to(device)
        fused = fusion(audio_t, lyrics_t)
        fused_chunks.append(fused.cpu())

    fused_np = torch.cat(fused_chunks, dim=0).cpu().numpy().astype(np.float32)
    return np.ascontiguousarray(fused_np)
