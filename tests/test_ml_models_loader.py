import pandas as pd
import pytest
from src.signals.ml_models import ModelLoader

torch = pytest.importorskip("torch", reason="requires torch")


def test_loader_predict_dummy():
    loader = ModelLoader()
    loader.load_nbeats()  # uses simple model
    df = pd.DataFrame([[0.1]*10])
    preds = loader.predict('nbeats', df)
    assert preds.shape[0] == 1
