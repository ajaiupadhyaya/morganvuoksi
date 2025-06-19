"""Utility to load and run ML models for signal generation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

try:
    from ..models.lstm import LSTM
    from ..models.xgboost import XGBoost
    from ..models.transformer import TransformerModel
except Exception:  # pragma: no cover - optional deps
    LSTM = XGBoost = TransformerModel = None

try:  # optional deps for N-BEATS
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - optional
    torch = None
    nn = None


if nn is not None:
    class NBeatsModel(nn.Module):
        """Simple N-BEATS style model for forecasting."""

        def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple
            return self.fc(x)
else:
    class NBeatsModel:
        pass


@dataclass
class LoadedModel:
    name: str
    model: Any


class ModelLoader:
    """Load and run various ML models."""

    def __init__(self, model_dir: str = "models") -> None:
        self.model_dir = Path(model_dir)
        self.models: Dict[str, LoadedModel] = {}

    def load_lstm(self, path: Optional[str] = None) -> LSTM:
        if LSTM is None:
            raise ImportError("LSTM model requires torch")
        model = LSTM()
        model_path = Path(path or self.model_dir / "lstm.pth")
        if model_path.exists():
            model.load(str(model_path))
        self.models["lstm"] = LoadedModel("lstm", model)
        return model

    def load_xgboost(self, path: Optional[str] = None) -> XGBoost:
        model = XGBoost()
        model_path = Path(path or self.model_dir / "xgb.json")
        if model_path.exists():
            model.load(str(model_path))
        self.models["xgboost"] = LoadedModel("xgboost", model)
        return model

    def load_transformer(self, path: Optional[str] = None) -> TransformerModel:
        if TransformerModel is None:
            raise ImportError("Transformer model requires torch")
        model = TransformerModel({})
        model_path = Path(path or self.model_dir / "transformer.pth")
        if model_path.exists():
            model.load(str(model_path))
        self.models["transformer"] = LoadedModel("transformer", model)
        return model

    def load_nbeats(self, path: Optional[str] = None) -> NBeatsModel:
        if torch is None:
            raise ImportError("PyTorch required for N-BEATS")
        model = NBeatsModel(input_dim=10, hidden_dim=64, output_dim=1)
        model_path = Path(path or self.model_dir / "nbeats.pth")
        if model_path.exists():
            model.load_state_dict(torch.load(model_path))
        self.models["nbeats"] = LoadedModel("nbeats", model)
        return model

    def get(self, name: str) -> Any:
        if name not in self.models:
            loader = getattr(self, f"load_{name}", None)
            if callable(loader):
                loader()
            else:
                raise ValueError(f"Unknown model: {name}")
        return self.models[name].model

    def predict(self, name: str, df: pd.DataFrame) -> np.ndarray:
        model = self.get(name)
        if hasattr(model, "predict"):
            return model.predict(df)
        if torch and isinstance(model, nn.Module):
            model.eval()
            with torch.no_grad():
                tensor = torch.FloatTensor(df.values)
                return model(tensor).cpu().numpy()
        raise ValueError(f"Model {name} cannot make predictions")


__all__ = ["ModelLoader", "LoadedModel"]
