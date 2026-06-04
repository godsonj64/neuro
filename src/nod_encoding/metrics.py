from __future__ import annotations

import numpy as np


def pearsonr_columns(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = y_true - y_true.mean(axis=0, keepdims=True)
    y_pred = y_pred - y_pred.mean(axis=0, keepdims=True)
    num = np.sum(y_true * y_pred, axis=0)
    den = np.sqrt(np.sum(y_true**2, axis=0) * np.sum(y_pred**2, axis=0)) + eps
    return num / den


def r2_columns(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2, axis=0) + eps
    return 1.0 - ss_res / ss_tot
