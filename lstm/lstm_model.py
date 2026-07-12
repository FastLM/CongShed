"""2-layer LSTM congestion predictor."""

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
    Architecture: 2-layer LSTM, 32 hidden units.
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
        """Export INT8-quantized weights (HICSLSTM v2) for C++/Rust runtime."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = self.state_dict()

        def quantize(tensor: torch.Tensor) -> tuple[bytes, float]:
            scale = float(tensor.abs().max().item() / 127.0 or 1e-8)
            q = (tensor / scale).round().clamp(-127, 127).to(torch.int8)
            return q.contiguous().numpy().tobytes(), scale

        with open(path, "wb") as f:
            f.write(b"HICSLSTM")
            f.write(struct.pack("<IIII", 2, HIDDEN_SIZE, 2, HISTORY_LEN))

            for layer in range(2):
                w_ih = state[f"lstm.weight_ih_l{layer}"]
                w_hh = state[f"lstm.weight_hh_l{layer}"]
                b_ih = state[f"lstm.bias_ih_l{layer}"].detach().cpu().float().numpy()
                b_hh = state[f"lstm.bias_hh_l{layer}"].detach().cpu().float().numpy()
                input_size = 1 if layer == 0 else HIDDEN_SIZE
                rows = 4 * HIDDEN_SIZE
                f.write(struct.pack("<II", input_size, rows))
                q_ih, s_ih = quantize(w_ih)
                f.write(q_ih)
                f.write(struct.pack("<f", s_ih))
                q_hh, s_hh = quantize(w_hh)
                f.write(q_hh)
                f.write(struct.pack("<f", s_hh))
                f.write(b_ih.astype("<f4").tobytes())
                f.write(b_hh.astype("<f4").tobytes())

            head_w = state["head.weight"].view(-1)
            head_b = float(state["head.bias"].item())
            q_h, s_h = quantize(head_w)
            f.write(struct.pack("<I", HIDDEN_SIZE))
            f.write(q_h)
            f.write(struct.pack("<ff", s_h, head_b))

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
