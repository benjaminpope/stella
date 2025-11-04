import pytest
import os
import sys


def _run_minimal_model(expect_backend: str):
    import keras
    import numpy as np

    assert keras.backend.backend() == expect_backend
    # Minimal dense network
    m = keras.Sequential(
        [
            keras.layers.Input((20,)),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    x = np.random.RandomState(0).randn(4, 20).astype("float32")
    y = m(x, training=False)
    assert y.shape == (4, 1)


def test_minimal_jax():
    pytest.importorskip("jax")
    # Only run if keras is either not imported or already using jax
    if "keras" in sys.modules:
        import keras

        if keras.backend.backend() != "jax":
            return  # skip if another backend is already active in this process
    else:
        os.environ["KERAS_BACKEND"] = "jax"
    _run_minimal_model("jax")


def test_minimal_torch():
    torch = pytest.importorskip("torch")
    if hasattr(torch.backends, "mps"):
        if not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
            pytest.skip("MPS backend not available or not built.")
    # Only run if keras is either not imported or already using torch
    if "keras" in sys.modules:
        import keras
        if keras.backend.backend() != "torch":
            return  # skip if another backend is already active
    else:
        os.environ["KERAS_BACKEND"] = "torch"
    _run_minimal_model("torch")