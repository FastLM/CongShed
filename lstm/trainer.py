"""LSTM training pipeline with online Adagrad updates."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adagrad

from hics.lstm.lstm_model import CongestionLSTM, HISTORY_LEN, PREDICT_AHEAD_MS


class UtilizationDataset:
    """
    Sliding window dataset from telemetry traces.
    Input: K=16 consecutive 1 ms samples
    Target: utilization 5 ms ahead
    """

    def __init__(
        self,
        trace: np.ndarray,
        history_len: int = HISTORY_LEN,
        predict_ahead: int = PREDICT_AHEAD_MS,
    ):
        self.trace = trace.astype(np.float32)
        self.history_len = history_len
        self.predict_ahead = predict_ahead

    def __len__(self) -> int:
        return max(0, len(self.trace) - self.history_len - self.predict_ahead)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.trace[idx : idx + self.history_len]
        y = self.trace[idx + self.history_len + self.predict_ahead - 1]
        return (
            torch.tensor(x, dtype=torch.float32).unsqueeze(-1),
            torch.tensor([y], dtype=torch.float32),
        )


def generate_synthetic_trace(
    length_ms: int = 100_000,
    seed: int = 42,
) -> np.ndarray:
    """
    Synthetic link utilization trace mimicking KV migration bursts
    and bimodal utilization patterns.
    """
    rng = np.random.default_rng(seed)
    trace = rng.uniform(0.3, 0.6, size=length_ms).astype(np.float32)

    # Inject KV migration bursts
    for _ in range(length_ms // 500):
        start = rng.integers(0, length_ms - 50)
        duration = rng.integers(5, 20)
        peak = rng.uniform(0.85, 0.99)
        trace[start : start + duration] = peak

    # Diurnal utilization pattern
    for t in range(length_ms):
        hour = (t // 3600) % 24
        if 10 <= hour <= 18:
            trace[t] = min(trace[t] * 1.3, 0.99)

    return trace


class CongestionTrainer:
    def __init__(
        self,
        model: Optional[CongestionLSTM] = None,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        self.model = model or CongestionLSTM()
        self.device = device
        self.model.to(device)
        self.optimizer = Adagrad(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train_epoch(
        self, dataset: UtilizationDataset, batch_size: int = 64
    ) -> float:
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        self.model.train()
        total_loss = 0.0
        n = 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = self.criterion(pred, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            n += 1
        return total_loss / max(n, 1)

    def evaluate(self, dataset: UtilizationDataset) -> dict:
        self.model.eval()
        errors = []
        with torch.no_grad():
            for i in range(len(dataset)):
                x, y = dataset[i]
                pred = self.model(x.unsqueeze(0))
                errors.append(abs(pred.item() - y.item()))
        errors = np.array(errors)
        return {
            "mae_pct": float(errors.mean() * 100),
            "p95_err_pct": float(np.percentile(errors, 95) * 100),
        }

    def online_update(
        self, history: list[float], target: float
    ) -> float:
        """Single-step online update every 50 ms."""
        self.model.train()
        x = torch.tensor(history, dtype=torch.float32).view(1, HISTORY_LEN, 1)
        y = torch.tensor([[target]], dtype=torch.float32)
        self.optimizer.zero_grad()
        pred = self.model(x)
        loss = self.criterion(pred, y)
        loss.backward()
        self.optimizer.step()
        return pred.item()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def export_runtime_weights(self, path: str | Path) -> None:
        self.model.export_int8_weights(path)

    @classmethod
    def train_and_evaluate(
        cls,
        trace: Optional[np.ndarray] = None,
        epochs: int = 10,
        export_path: Optional[str] = None,
    ) -> "CongestionTrainer":
        trace = trace or generate_synthetic_trace()
        split = int(len(trace) * 0.8)
        train_ds = UtilizationDataset(trace[:split])
        val_ds = UtilizationDataset(trace[split:])

        trainer = cls()
        for epoch in range(epochs):
            loss = trainer.train_epoch(train_ds)
            metrics = trainer.evaluate(val_ds)
            print(
                f"Epoch {epoch+1}/{epochs}  loss={loss:.6f}  "
                f"MAE={metrics['mae_pct']:.1f}%  "
                f"P95={metrics['p95_err_pct']:.1f}%"
            )

        if export_path:
            trainer.export_runtime_weights(export_path)
            trainer.save(export_path + ".pt")
        return trainer
