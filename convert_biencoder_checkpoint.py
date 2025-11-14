#!/usr/bin/env python3
"""
Convert `checkpoints/biencoder/bi_encoder_best.pth` into a Hugging Face checkpoint
so that `AutoModel.from_pretrained("checkpoints/biencoder")` works without code changes.

The script:
1. Loads the saved bi-encoder state dict from the notebook training run.
2. Extracts only the lyrics encoder weights.
3. Initializes the base Longformer architecture from config and loads those weights.
4. Saves the resulting model + tokenizer into `checkpoints/biencoder`.

Run this once:
    python convert_biencoder_checkpoint.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert bi_encoder_best.pth into a HF AutoModel checkpoint.")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/biencoder"),
        help="Directory that contains bi_encoder_best.pth (outputs are written here as well).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="xlm-roberta-large",
        help="Base pretrained model used during training (used to fetch tokenizer/config template).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_dir: Path = args.checkpoint_dir
    pth_path = checkpoint_dir / "bi_encoder_best.pth"
    if not pth_path.exists():
        raise FileNotFoundError(f"{pth_path} not found. Put the original training checkpoint there.")

    print(f"Loading checkpoint from {pth_path} ...")
    state_dict = torch.load(pth_path, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]

    prefixes = ["lyrics_encoder.", "backbone."]
    base_state = {}
    used_prefix = None
    for prefix in prefixes:
        filtered = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
        if filtered:
            base_state = filtered
            used_prefix = prefix
            break

    if not base_state:
        raise RuntimeError(
            "Could not find expected keys (lyrics_encoder.* or backbone.*) in the checkpoint. Nothing to convert."
        )

    print("Deriving model dimensions from checkpoint ...")
    def shape_of(name: str) -> torch.Size:
        if name not in base_state:
            raise KeyError(f"{name} not found in checkpoint.")
        return base_state[name].shape

    vocab_size, hidden_size = shape_of("embeddings.word_embeddings.weight")
    max_position_embeddings, _ = shape_of("embeddings.position_embeddings.weight")
    type_vocab_size, _ = shape_of("embeddings.token_type_embeddings.weight")
    intermediate_size, _ = shape_of("encoder.layer.0.intermediate.dense.weight")
    num_layers = max(
        int(k.split(".")[2]) for k in base_state.keys() if k.startswith("encoder.layer.")
    ) + 1

    config = AutoConfig.from_pretrained(args.model_name)
    config.vocab_size = vocab_size
    config.hidden_size = hidden_size
    config.max_position_embeddings = max_position_embeddings
    config.type_vocab_size = type_vocab_size
    config.intermediate_size = intermediate_size
    config.num_hidden_layers = num_layers
    if hidden_size % config.num_attention_heads != 0:
        guessed_heads = hidden_size // 64
        if hidden_size % guessed_heads == 0:
            config.num_attention_heads = guessed_heads
        else:
            raise RuntimeError(
                f"Cannot derive num_attention_heads: hidden_size={hidden_size}, "
                f"current heads={config.num_attention_heads}"
            )

    print("Initializing base model from config ...")
    model = AutoModel.from_config(config)
    missing, unexpected = model.load_state_dict(base_state, strict=False)
    if missing:
        raise RuntimeError(f"Missing parameters when loading into base model: {missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected parameters after removing prefix: {unexpected}")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving converted model into {checkpoint_dir} ...")
    model.save_pretrained(checkpoint_dir)

    print("Saving tokenizer into checkpoint directory ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.save_pretrained(checkpoint_dir)

    print(f"Conversion complete (used prefix '{used_prefix}').")


if __name__ == "__main__":
    main()
