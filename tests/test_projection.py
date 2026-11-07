import numpy as np
import pytest

from doccheck.rag.projection import fit_pca


def test_pca_recovers_main_axis_and_normalises():
    rng = np.random.default_rng(0)
    t = rng.normal(size=200)
    x = np.zeros((200, 16))
    x[:, 3] = 10 * t  # dominant direction
    x[:, 7] = 2 * rng.normal(size=200)
    basis = fit_pca(x)
    assert abs(basis.components[0, 3]) > 0.99
    xy = basis.project(x)
    assert xy.shape == (200, 2)
    assert np.all(xy <= 1.0) and np.all(xy >= -1.0)
    assert np.isclose(np.abs(xy).max(axis=0), 1.0).all()


def test_projection_is_deterministic_and_query_consistent():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(50, 384))
    b1, b2 = fit_pca(x), fit_pca(x.copy())
    assert np.allclose(b1.project(x), b2.project(x))
    q = x[5].tolist()
    assert np.allclose(b1.project_one(q), b1.project(x)[5], atol=1e-4)


def test_degenerate_inputs():
    single = fit_pca(np.ones((1, 8)))
    assert single.project(np.ones((1, 8))).tolist() == [[0.0, 0.0]]
    with pytest.raises(ValueError):
        fit_pca(np.zeros((0, 8)))
