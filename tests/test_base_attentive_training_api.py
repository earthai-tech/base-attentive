from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


PKG_PREFIX = "base_attentive"


def _purge_base_attentive_modules() -> None:
    """Remove loaded package modules so import-time class bases refresh.

    BaseAttentiveV2 binds `Model = KERAS_DEPS.Model` at module import time.
    If the package was first imported while the fallback runtime was active,
    later importing `keras` will not retroactively change the already-defined
    class base. Purging lets us re-import against the real Keras runtime.
    """
    doomed = [
        name
        for name in sys.modules
        if name == PKG_PREFIX or name.startswith(f"{PKG_PREFIX}.")
    ]
    for name in doomed:
        sys.modules.pop(name, None)





def _run_python(code: str, monkeypatch=None):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    env.setdefault("BASE_ATTENTIVE_EAGER_RUNTIME", "1")
    env.setdefault("BASE_ATTENTIVE_BACKEND", "tensorflow")
    env.setdefault("KERAS_BACKEND", "tensorflow")
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
def _import_package_with_real_keras(monkeypatch):
    """Reload the package in a state where real Keras is available.

    This mirrors how notebooks/examples should import the package if users
    expect the full Keras training API (`compile`, `fit`, `predict`, etc.).
    """
    keras = pytest.importorskip("keras")

    monkeypatch.setenv("BASE_ATTENTIVE_EAGER_RUNTIME", "1")
    monkeypatch.setenv("BASE_ATTENTIVE_BACKEND", "tensorflow")
    monkeypatch.setenv("KERAS_BACKEND", "tensorflow")

    _purge_base_attentive_modules()

    # Import standalone Keras first so KERAS_DEPS.Model resolves to a real
    # Keras Model instead of the package fallback model.
    importlib.import_module("keras")

    package = importlib.import_module("base_attentive")
    core_module = importlib.import_module(
        "base_attentive.core.base_attentive"
    )
    v2_module = importlib.import_module(
        "base_attentive.experimental.base_attentive_v2"
    )
    return keras, package, core_module, v2_module



def _make_supervised_data(
    *,
    n_samples: int = 24,
    lookback: int = 6,
    static_dim: int = 3,
    dynamic_dim: int = 4,
    future_dim: int = 2,
    horizon: int = 2,
    output_dim: int = 1,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    x_static = rng.normal(size=(n_samples, static_dim)).astype("float32")
    x_dynamic = rng.normal(
        size=(n_samples, lookback, dynamic_dim)
    ).astype("float32")
    x_future = rng.normal(
        size=(n_samples, horizon, future_dim)
    ).astype("float32")
    y_point = rng.normal(
        size=(n_samples, horizon, output_dim)
    ).astype("float32")
    return (x_static, x_dynamic, x_future), y_point


@pytest.mark.filterwarnings("ignore:.*oneDNN custom operations.*")
def test_base_attentive_uses_real_keras_model_when_keras_is_preloaded(
    monkeypatch,
):
    keras, package, core_module, v2_module = _import_package_with_real_keras(
        monkeypatch
    )

    BaseAttentive = package.BaseAttentive
    BaseAttentiveV2 = v2_module.BaseAttentiveV2

    assert issubclass(BaseAttentiveV2, keras.Model)
    assert issubclass(BaseAttentive, BaseAttentiveV2)
    assert hasattr(BaseAttentive, "compile")
    assert hasattr(BaseAttentive, "fit")
    assert hasattr(BaseAttentive, "predict")


@pytest.mark.filterwarnings("ignore:.*oneDNN custom operations.*")
def test_base_attentive_point_compile_fit_predict_roundtrip(monkeypatch):
    keras, package, _, _ = _import_package_with_real_keras(monkeypatch)
    BaseAttentive = package.BaseAttentive

    inputs, targets = _make_supervised_data()

    model = BaseAttentive(
        static_dim=inputs[0].shape[-1],
        dynamic_dim=inputs[1].shape[-1],
        future_dim=inputs[2].shape[-1],
        output_dim=targets.shape[-1],
        forecast_horizon=targets.shape[1],
        embed_dim=16,
        hidden_units=32,
        attention_units=16,
        num_heads=2,
        num_encoder_layers=1,
        lookback_window=inputs[1].shape[1],
        use_residuals=False,
        use_batch_norm=False,
        use_vsn=False,
        objective="hybrid",
        name="ba_point_train",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError()],
    )

    history = model.fit(
        list(inputs),
        targets,
        epochs=1,
        batch_size=8,
        verbose=0,
    )
    pred = model.predict(list(inputs), verbose=0)

    assert "loss" in history.history
    assert pred.shape == targets.shape
    assert np.isfinite(pred).all()

    cfg = model.get_config()
    clone = BaseAttentive.from_config(cfg)
    clone.compile(optimizer="adam", loss="mse")
    clone_pred = clone.predict(list(inputs), verbose=0)
    assert clone_pred.shape == targets.shape


@pytest.mark.filterwarnings("ignore:.*oneDNN custom operations.*")
def test_base_attentive_quantile_predict_and_train_step(monkeypatch):
    keras, package, _, _ = _import_package_with_real_keras(monkeypatch)
    BaseAttentive = package.BaseAttentive

    quantiles = (0.1, 0.5, 0.9)
    inputs, y_point = _make_supervised_data(horizon=3)
    y_quant = np.repeat(y_point[:, :, None, :], len(quantiles), axis=2)

    model = BaseAttentive(
        static_dim=inputs[0].shape[-1],
        dynamic_dim=inputs[1].shape[-1],
        future_dim=inputs[2].shape[-1],
        output_dim=y_point.shape[-1],
        forecast_horizon=y_point.shape[1],
        quantiles=quantiles,
        embed_dim=16,
        hidden_units=32,
        attention_units=16,
        num_heads=2,
        num_encoder_layers=1,
        lookback_window=inputs[1].shape[1],
        use_residuals=False,
        use_batch_norm=False,
        use_vsn=False,
        objective="hybrid",
        name="ba_quantile_train",
    )

    model.compile(optimizer="adam", loss="mse")
    history = model.fit(
        list(inputs),
        y_quant,
        epochs=1,
        batch_size=8,
        verbose=0,
    )
    pred = model.predict(list(inputs), verbose=0)

    assert "loss" in history.history
    # Contract: batch, horizon, quantiles, output_dim
    assert pred.shape == y_quant.shape
    assert pred.shape[2] == len(quantiles)


@pytest.mark.filterwarnings("ignore:.*oneDNN custom operations.*")
def test_base_attentive_save_and_reload_after_fit(tmp_path):
    """Run save/reload in a subprocess to isolate native TF runtime state.

    On Windows, repeated TensorFlow/Keras initialization across a long pytest
    session can trigger fatal access violations in native threads. This test
    still verifies the real save/reload path, but in a clean subprocess.
    """
    model_path = tmp_path / "base_attentive.keras"
    code = f"""
import importlib
import numpy as np
import os
import sys
os.environ['BASE_ATTENTIVE_EAGER_RUNTIME']='1'
os.environ['BASE_ATTENTIVE_BACKEND']='tensorflow'
os.environ['KERAS_BACKEND']='tensorflow'
import keras
import base_attentive as package
BaseAttentive = package.BaseAttentive
rng = np.random.default_rng(7)
inputs = (
    rng.normal(size=(12, 3)).astype('float32'),
    rng.normal(size=(12, 5, 4)).astype('float32'),
    rng.normal(size=(12, 2, 2)).astype('float32'),
)
targets = rng.normal(size=(12, 2, 1)).astype('float32')
model = BaseAttentive(
    static_dim=3,
    dynamic_dim=4,
    future_dim=2,
    output_dim=1,
    forecast_horizon=2,
    embed_dim=8,
    hidden_units=16,
    attention_units=8,
    num_heads=2,
    num_encoder_layers=1,
    lookback_window=5,
    use_residuals=False,
    use_batch_norm=False,
    use_vsn=False,
    objective='hybrid',
    name='ba_save_reload',
)
model.compile(optimizer='adam', loss='mse')
model.fit(list(inputs), targets, epochs=1, batch_size=4, verbose=0)
model.save(r'{str(model_path)}')
restored = keras.models.load_model(r'{str(model_path)}')
pred = restored.predict(list(inputs), verbose=0)
assert pred.shape == targets.shape
"""
    completed = _run_python(code)
    assert completed.returncode == 0, completed.stderr
    assert model_path.exists()


def test_explains_why_compile_can_be_missing_under_fallback(monkeypatch):
    """Document the import-order pitfall behind the missing `compile` API.

    If `base_attentive` is imported before real standalone Keras is loaded and
    eager runtime imports are disabled, `BaseAttentiveV2` binds against the
    package fallback `Model`, which intentionally does not implement the Keras
    training API.
    """
    monkeypatch.setenv("BASE_ATTENTIVE_EAGER_RUNTIME", "0")
    monkeypatch.setenv("BASE_ATTENTIVE_BACKEND", "tensorflow")
    monkeypatch.setenv("KERAS_BACKEND", "tensorflow")
    sys.modules.pop("keras", None)
    _purge_base_attentive_modules()

    exp_mod = importlib.import_module(
        "base_attentive.experimental.base_attentive_v2"
    )
    fallback_mod = importlib.import_module("base_attentive._keras_fallback")

    assert exp_mod.Model is fallback_mod.Model
    assert not hasattr(exp_mod.BaseAttentiveV2, "compile")
