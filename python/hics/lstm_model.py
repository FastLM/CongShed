"""2-layer LSTM congestion predictor (§V-C, Table III)."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

HIDDEN_SIZE = 32
HISTORY_LEN = 16
PREDICT_AHEAD_MS = 5
SAMPLE_INTERVAL_MS = 1


class CongestionLSTM(nn.Module):
    """
    Predicts link utilization 5 ms ahead from K=16 samples at 1 ms intervals.
    Architecture: 2-layer LSTM, 32 hidden units (paper Table III).
    """

    def __init__(self, hidden_size: int = HIDDEN_SIZE, num_layers: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, HISTORY_LEN, 1) utilization history
        Returns:
            (batch, 1) predicted utilization ∈ [0, 1]
        """
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.sigmoid(self.head(last))

    def predict_single(self, history: list[float]) -> float:
        assert len(history) == HISTORY_LEN
        x = torch.tensor(history, dtype=torch.float32).view(1, HISTORY_LEN, 1)
        with torch.no_grad():
            return self.forward(x).item()

    def export_int8_weights(self, path: str | Path) -> None:
        """Export INT8-quantized weights for C++/Rust runtime (§VI)."""
        path = Path(path)
        state = self.state_dict()

        def quantize(tensor: torch.Tensor) -> tuple[bytes, bytes]:
            scale = tensor.abs().max().item() / 127.0 or 1e-8
            q = (tensor / scale).round().clamp(-127, 127).to(torch.int8)
            return q.numpy().tobytes(), struct.pack("f", scale)

        with open(path, "wb") as f:
            for key in ["lstm.weight_ih_l0", "lstm.weight_hh_l0"]:
                if key in state:
                    qbytes, scale = quantize(state[key])
                    f.write(qbytes)
                    f.write(scale)

    @classmethod
    def load_from_checkpoint(cls, path: str | Path) -> "CongestionLSTM":
        model = cls()
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model


class PredictorCache:
    """Per-link prediction cache with effective utilization blending."""

    def __init__(self, model: CongestionLSTM):
        self.model = model
        self._predictions: dict[int, float] = {}

    def update(self, edge_idx: int, history: list[float]) -> float:
        pred = self.model.predict_single(history)
        self._predictions[edge_idx] = pred
        return pred

    def effective_util(
        self, edge_idx: int, current: float, transfer_duration_ms: float
    ) -> float:
        predicted = self._predictions.get(edge_idx, current)
        weight = min(transfer_duration_ms / PREDICT_AHEAD_MS, 1.0)
        blended = predicted * weight + current * (1.0 - weight)
        return max(current, blended)
