from pathlib import Path
from typing import List

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor, AutoTokenizer, ClapModel

device = torch.device("mps")  # поменяешь на то, что нужно
clap_checkpoint_dir = Path("checkpoints/clap")
biencoder_checkpoint_dir = Path("checkpoints/biencoder")

biencoder_tokenizer = None
biencoder_model = None
clap_processor = None
clap_model = None


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
