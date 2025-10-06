from __future__ import annotations

import random
import shutil
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
from yt_dlp.utils import DownloadError
from tqdm.auto import tqdm

try:
    import browser_cookie3
except ImportError:
    browser_cookie3 = None

MODEL_NAME: str = "laion/clap-htsat-unfused"
RANDOM_SEED: int = 42
AUDIO_SAMPLING_RATE: int = 48_000
CLIP_SECONDS: int = 10
MODE: str = "train"  # or "inference"
TRAIN_BATCH_SIZE: int = 22
EVAL_BATCH_SIZE: int = 2
LEARNING_RATE: float = 1e-5
WEIGHT_DECAY: float = 1e-4
NUM_EPOCHS: int = 10
TRAIN_LOG_EVERY_STEPS: int = 10
GRADIENT_ACCUMULATION_STEPS: int = 1
AUDIO_CACHE_DIR: Path = Path("data/musiccaps/audio")
MUSICCAPS_DATASET: str = "google/musiccaps"
MUSICCAPS_SPLIT: str = "train"
INFERENCE_PROMPT: str = "a melancholic indie ballad with soft vocals and slow tempo"
INFERENCE_TOPK: int = 3
MLFLOW_TRACKING_URI: str = "file:mlruns"
MLFLOW_EXPERIMENT_NAME: str = "musiccaps_clap"
MLFLOW_RUN_NAME: str = "clap_musiccaps_run"
MLFLOW_AUTO_LAUNCH_UI: bool = True
MLFLOW_UI_PORT: int = 5050

CHECKPOINT_DIR: Path = Path("checkpoints")
BEST_CHECKPOINT_DIR: Path = CHECKPOINT_DIR / "best"
RESUME_TRAINING: bool = True

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


def launch_mlflow_ui(tracking_uri: str) -> Optional[subprocess.Popen]:
    if not MLFLOW_AUTO_LAUNCH_UI:
        return None
    if not tracking_uri.startswith("file:"):
        print("MLflow UI launch skipped: tracking URI is not a local file store.")
        return None
    backend_root = Path(tracking_uri[5:]).resolve()
    backend_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        "mlflow",
        "ui",
        "--backend-store-uri",
        str(backend_root),
        "--host",
        "127.0.0.1",
        "--port",
        str(MLFLOW_UI_PORT),
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"MLflow UI available at http://127.0.0.1:{MLFLOW_UI_PORT}")
        return proc
    except FileNotFoundError:
        print("mlflow CLI not found in PATH; UI launch skipped.")
    except OSError as exc:
        print(f"Failed to launch MLflow UI: {exc}")
    return None


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


def prepare_musiccaps_samples(sample_limit: Optional[int] = None) -> List[MusicCapsSample]:
    metadata = load_musiccaps_metadata(sample_limit)
    ordered_rows: List[Dict[str, str]] = list(metadata)
    order_index = {
        (row["ytid"], float(row["start_s"])): idx for idx, row in enumerate(ordered_rows)
    }

    all_samples: List[MusicCapsSample] = []
    for row in ordered_rows:
        clip_path = AUDIO_CACHE_DIR / f"{row['ytid']}_{int(row['start_s'])}.wav"
        if clip_path.exists():
            all_samples.append(
                MusicCapsSample(
                    ytid=row["ytid"],
                    start_s=row["start_s"],
                    text=row["caption"],
                    audio_path=clip_path,
                )
            )
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


def evaluate(
    model: ClapModel,
    loader: DataLoader,
    device: torch.device,
    *,
    desc: Optional[str] = None,
) -> Metrics:
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    steps = 0
    progress_desc = desc or "Validation"
    try:
        total_batches: Optional[int] = len(loader)
    except TypeError:
        total_batches = None
    progress_bar = tqdm(
        loader,
        desc=progress_desc,
        unit="batch",
        leave=False,
        total=total_batches,
    )
    with torch.no_grad():
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch, return_loss=True)
            total_loss += outputs.loss.item()
            total_accuracy += compute_accuracy(outputs.logits_per_text)
            steps += 1
            avg_loss = total_loss / steps
            avg_accuracy = total_accuracy / steps
            progress_bar.set_postfix(
                loss=f"{avg_loss:.4f}",
                acc=f"{avg_accuracy:.4f}",
            )
    progress_bar.close()
    if steps == 0:
        return Metrics(loss=0.0, accuracy=0.0)
    return Metrics(loss=total_loss / steps, accuracy=total_accuracy / steps)


def save_checkpoint(
    model: ClapModel,
    processor: AutoProcessor,
    target_dir: Path,
) -> None:
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(target_dir)
    processor.save_pretrained(target_dir)
    print(f"Saved checkpoint: {target_dir}")


def train(
    model: ClapModel,
    processor: AutoProcessor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    initial_best_val_accuracy: Optional[float] = None,
    initial_global_step: int = 0,
) -> float:
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    model.to(device)
    best_val_accuracy = (
        initial_best_val_accuracy if initial_best_val_accuracy is not None else -float("inf")
    )
    global_step = initial_global_step

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    epoch_bar = tqdm(
        range(NUM_EPOCHS),
        desc="Epochs",
        unit="epoch",
    )
    for epoch_index in epoch_bar:
        epoch_number = epoch_index + 1
        model.train()
        running_loss = 0.0
        step_count = 0
        try:
            total_train_batches: Optional[int] = len(train_loader)
        except TypeError:
            total_train_batches = None
        train_bar = tqdm(
            train_loader,
            desc=f"Train {epoch_number}/{NUM_EPOCHS}",
            unit="batch",
            leave=False,
            total=total_train_batches,
        )
        for batch in train_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch, return_loss=True)
            loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            if (step_count + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
            batch_loss = outputs.loss.item()
            running_loss += batch_loss
            step_count += 1
            global_step += 1

            avg_loss = running_loss / max(1, step_count)
            train_bar.set_postfix(
                loss=f"{batch_loss:.4f}",
                avg_loss=f"{avg_loss:.4f}",
            )

            if TRAIN_LOG_EVERY_STEPS and global_step % TRAIN_LOG_EVERY_STEPS == 0:
                train_accuracy = compute_accuracy(outputs.logits_per_text.detach())
                mlflow.log_metric("train_loss_step", batch_loss, step=global_step)
                mlflow.log_metric("train_accuracy_step", train_accuracy, step=global_step)

        if step_count % GRADIENT_ACCUMULATION_STEPS != 0:
            optimizer.step()
            optimizer.zero_grad()

        avg_train_loss = running_loss / max(1, step_count)
        mlflow.log_metric("train_loss", avg_train_loss, step=epoch_number)
        metrics = evaluate(
            model,
            val_loader,
            device,
            desc=f"Val {epoch_number}/{NUM_EPOCHS}",
        )
        train_bar.close()
        tqdm.write(
            f"epoch={epoch_number}/{NUM_EPOCHS} train_loss={avg_train_loss:.4f} "
            f"val_loss={metrics.loss:.4f} val_acc={metrics.accuracy:.4f}"
        )
        epoch_bar.set_postfix(
            train_loss=f"{avg_train_loss:.4f}",
            val_acc=f"{metrics.accuracy:.4f}",
        )
        mlflow.log_metric("val_loss", metrics.loss, step=epoch_number)
        mlflow.log_metric("val_accuracy", metrics.accuracy, step=epoch_number)

        if metrics.accuracy > best_val_accuracy:
            best_val_accuracy = metrics.accuracy
            save_checkpoint(model, processor, BEST_CHECKPOINT_DIR)

    epoch_bar.close()
    return best_val_accuracy

def build_loaders(processor: AutoProcessor, sample_limit: Optional[int] = None) -> Dict[str, DataLoader]:
    samples = prepare_musiccaps_samples(sample_limit)
    if not samples:
        raise RuntimeError(
            "No cached audio found. Ensure data/musiccaps/audio contains clips "
            "matching the MusicCaps metadata."
        )
    pivot = max(1, int(len(samples) * 0.9))
    pivot = min(pivot, len(samples))
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
    checkpoint_source: str | Path = MODEL_NAME
    loaded_from_checkpoint = False
    if RESUME_TRAINING and BEST_CHECKPOINT_DIR.exists():
        checkpoint_source = BEST_CHECKPOINT_DIR
        loaded_from_checkpoint = True
        print(f"Loading weights from checkpoint: {BEST_CHECKPOINT_DIR}")

    processor = AutoProcessor.from_pretrained(checkpoint_source)
    model = ClapModel.from_pretrained(checkpoint_source).to(device)

    if MODE == "train":
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        launch_mlflow_ui(MLFLOW_TRACKING_URI)
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
                "clip_seconds": CLIP_SECONDS,
                "sampling_rate": AUDIO_SAMPLING_RATE,
                "num_epochs": NUM_EPOCHS,
            }
        )
        try:
            loaders = build_loaders(processor)
            initial_best_accuracy: Optional[float] = None
            if loaded_from_checkpoint:
                baseline_metrics = evaluate(
                    model,
                    loaders["val"],
                    device,
                    desc="Validation (resume)",
                )
                print(
                    "Resumed checkpoint validation: "
                    f"loss={baseline_metrics.loss:.4f} acc={baseline_metrics.accuracy:.4f}"
                )
                mlflow.log_metric("resume_val_loss", baseline_metrics.loss, step=0)
                mlflow.log_metric("resume_val_accuracy", baseline_metrics.accuracy, step=0)
                initial_best_accuracy = baseline_metrics.accuracy

            best_val_accuracy = train(
                model,
                processor,
                loaders["train"],
                loaders["val"],
                device,
                initial_best_val_accuracy=initial_best_accuracy,
            )
            metrics = evaluate(
                model,
                loaders["val"],
                device,
                desc="Validation (final)",
            )
            print(f"final loss={metrics.loss:.4f} acc={metrics.accuracy:.4f}")
            mlflow.log_metric("final_val_loss", metrics.loss, step=NUM_EPOCHS)
            mlflow.log_metric("final_val_accuracy", metrics.accuracy, step=NUM_EPOCHS)
            mlflow.log_metric("best_val_accuracy", best_val_accuracy, step=NUM_EPOCHS)
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
