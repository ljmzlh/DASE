"""Embedding-space diagnostics and transforms used by SemJI robustness tests."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

EPSILON = 1e-8


def shrinkage_covariance(values: np.ndarray, shrinkage: float) -> np.ndarray:
    centered = values - values.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(1, values.shape[0] - 1)
    if shrinkage > 0 and covariance.size:
        eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
        target = float(np.median(eigenvalues)) if eigenvalues.size else 1.0
        covariance = (
            (1.0 - shrinkage) * covariance
            + shrinkage * target * np.eye(covariance.shape[0])
        )
    return covariance


def participation_ratio(values: np.ndarray) -> float:
    eigenvalues = np.clip(np.asarray(values, dtype=float), 0.0, None)
    total = float(eigenvalues.sum())
    if total <= EPSILON:
        return 1.0
    return float(total * total / np.sum(eigenvalues * eigenvalues))


def spectral_diagnostics(values: np.ndarray) -> dict[str, float | int]:
    if values.shape[0] < 2 or values.shape[1] < 1:
        return {
            "participation_ratio": 1.0,
            "condition_number": float("inf"),
            "dimensions": int(values.shape[1]),
        }
    eigenvalues = np.clip(
        np.linalg.eigvalsh(shrinkage_covariance(values, 0.0))[::-1], 0.0, None
    )
    condition = (
        float(eigenvalues[0] / eigenvalues[-1])
        if eigenvalues[-1] > EPSILON
        else float("inf")
    )
    return {
        "participation_ratio": participation_ratio(eigenvalues),
        "condition_number": condition,
        "dimensions": int(values.shape[1]),
    }


@dataclass(frozen=True)
class EmbeddingTransform:
    method: str
    mean: np.ndarray
    components: np.ndarray
    scale: Optional[np.ndarray]
    renormalize: bool = False


def fit_transform(
    values: np.ndarray,
    *,
    method: str,
    shrinkage: float = 0.1,
    top_components: int = 1,
    renormalize: bool = False,
) -> EmbeddingTransform:
    if method not in {"pca", "abtt"}:
        raise ValueError(f"unknown embedding transform: {method!r}")
    mean = values.mean(axis=0)
    covariance = shrinkage_covariance(values, shrinkage if method == "pca" else 0.0)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    if method == "pca":
        scale = 1.0 / np.sqrt(np.clip(eigenvalues, EPSILON, None))
        return EmbeddingTransform(method, mean, eigenvectors, scale, renormalize)
    count = max(0, min(int(top_components), eigenvectors.shape[1]))
    return EmbeddingTransform(method, mean, eigenvectors[:, :count], None, renormalize)


def apply_transform(values: np.ndarray, transform: EmbeddingTransform) -> np.ndarray:
    centered = values - transform.mean
    if transform.method == "pca":
        output = (centered @ transform.components) * transform.scale
    elif transform.components.shape[1] == 0:
        output = centered
    else:
        projection = centered @ transform.components
        output = centered - projection @ transform.components.T
    if transform.renormalize:
        output = output / np.clip(
            np.linalg.norm(output, axis=1, keepdims=True), EPSILON, None
        )
    return output


def squared_l2_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = np.sum(left * left, axis=1)
    right_norm = np.sum(right * right, axis=1)
    squared = left_norm[:, None] + right_norm[None, :] - 2.0 * left @ right.T
    return np.maximum(squared, 0.0)


def l2_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.sqrt(squared_l2_matrix(left, right))