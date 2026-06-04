import numpy as np

from nod_encoding.eeg import make_lagged_design
from nod_encoding.metrics import pearsonr_columns, r2_columns


def test_lagged_design_shape():
    x = np.ones((2, 3), dtype=np.float32)
    onsets = np.array([2, 5])
    design, lags = make_lagged_design(x, onsets, n_times=10, sfreq=10, tmin=0.0, tmax=0.2)
    assert design.shape == (10, 9)
    assert len(lags) == 3
    assert design.sum() == 18


def test_metrics_perfect_prediction():
    y = np.random.default_rng(0).normal(size=(20, 4))
    assert np.allclose(pearsonr_columns(y, y), 1.0, atol=1e-6)
    assert np.allclose(r2_columns(y, y), 1.0, atol=1e-6)
