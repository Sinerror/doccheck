"""2D PCA via numpy SVD. The basis is fitted once per run and reused to project queries."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PCABasis:
    mean: np.ndarray        # (d,)
    components: np.ndarray  # (2, d)
    scale: np.ndarray       # (2,) max |coord| over the fitted points, for [-1, 1] normalisation

    def project(self, vectors: np.ndarray | list[list[float]]) -> np.ndarray:
        x = np.atleast_2d(np.asarray(vectors, dtype=np.float64))
        coords = (x - self.mean) @ self.components.T
        return np.clip(coords / self.scale, -1.0, 1.0)

    def project_one(self, vector: list[float]) -> list[float]:
        return [round(float(v), 5) for v in self.project([vector])[0]]


def fit_pca(vectors: np.ndarray | list[list[float]]) -> PCABasis:
    x = np.asarray(vectors, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("need a non-empty 2D array")
    mean = x.mean(axis=0)
    xc = x - mean
    d = x.shape[1]
    if x.shape[0] == 1:
        comps = np.eye(2, d)
    else:
        _, _, vt = np.linalg.svd(xc, full_matrices=False)
        comps = vt[:2]
        if comps.shape[0] < 2:
            comps = np.vstack([comps, np.zeros((2 - comps.shape[0], d))])
        # deterministic sign: largest-magnitude loading positive
        for i in range(2):
            j = int(np.argmax(np.abs(comps[i])))
            if comps[i, j] < 0:
                comps[i] = -comps[i]
    coords = xc @ comps.T
    scale = np.abs(coords).max(axis=0)
    scale[scale == 0] = 1.0
    return PCABasis(mean=mean, components=comps, scale=scale)
