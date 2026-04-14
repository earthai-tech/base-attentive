from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


SRC_DIR = Path(__file__).resolve().parents[1] / "src"


def _run_backend_script(
    backend: str, code: str, *, timeout: int = 120
):
    env = dict(os.environ)
    env["BASE_ATTENTIVE_BACKEND"] = backend
    env["KERAS_BACKEND"] = backend
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(SRC_DIR), env.get("PYTHONPATH")])
    )
    if backend == "jax":
        env.setdefault("JAX_DISABLE_JIT", "1")
    return subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
        env=env,
        timeout=timeout,
    )


def test_torch_full_matrix_smoke():
    code = textwrap.dedent(
        """
        import json
        import os
        import tempfile
        import numpy as np
        import keras
        from base_attentive.experimental.base_attentive_v2 import BaseAttentiveV2

        keras.utils.set_random_seed(7)

        def to_numpy(value):
            if hasattr(value, "detach"):
                value = value.detach()
            if hasattr(value, "cpu"):
                value = value.cpu()
            return np.asarray(value)

        model = BaseAttentiveV2(
            static_input_dim=2,
            dynamic_input_dim=3,
            future_input_dim=1,
            output_dim=2,
            forecast_horizon=4,
            quantiles=(0.1, 0.5, 0.9),
            head_type="quantile",
            spec={
                "architecture": {
                    "sequence_pooling": "pool.last",
                    "fusion": "fusion.concat",
                },
                "runtime": {"final_agg": "average"},
            },
        )
        inputs = [
            np.ones((2, 2), dtype="float32"),
            np.ones((2, 5, 3), dtype="float32"),
            np.ones((2, 4, 1), dtype="float32"),
        ]
        outputs = model(inputs)
        payload = model.to_json()
        clone = keras.models.model_from_json(payload)
        _ = clone(inputs)
        clone.set_weights(model.get_weights())
        clone_outputs = clone(inputs)
        path = os.path.join(tempfile.mkdtemp(), "model.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        loaded_outputs = loaded(inputs)
        summary = {
            "shape": list(getattr(outputs, "shape", ())),
            "clone_delta": float(
                np.max(np.abs(to_numpy(outputs) - to_numpy(clone_outputs)))
            ),
            "load_delta": float(
                np.max(np.abs(to_numpy(outputs) - to_numpy(loaded_outputs)))
            ),
        }
        print(json.dumps(summary))
        """
    )
    completed = _run_backend_script("torch", code)
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(
        completed.stdout.strip().splitlines()[-1]
    )
    assert payload["shape"] == [2, 4, 2, 3]
    assert payload["clone_delta"] < 1e-6
    assert payload["load_delta"] < 1e-6


@pytest.mark.skipif(
    importlib.util.find_spec("jax") is None,
    reason="JAX is not installed",
)
def test_jax_decoder_stack_smoke_and_config_roundtrip():
    code = textwrap.dedent(
        """
        import json
        import numpy as np
        import keras
        from base_attentive.experimental.base_attentive_v2 import BaseAttentiveV2, _invoke

        keras.utils.set_random_seed(7)
        model = BaseAttentiveV2(
            static_input_dim=2,
            dynamic_input_dim=3,
            future_input_dim=1,
            output_dim=2,
            forecast_horizon=4,
            quantiles=(0.1, 0.5, 0.9),
            head_type="quantile",
            spec={
                "architecture": {
                    "sequence_pooling": "pool.last",
                    "fusion": "fusion.concat",
                },
                "runtime": {"final_agg": "average"},
            },
        )
        static_x = np.ones((2, 2), dtype="float32")
        dynamic_x = np.ones((2, 5, 3), dtype="float32")
        future_x = np.ones((2, 4, 1), dtype="float32")

        sp = model._resolve_processor("static_processor", "static_projection")
        dp = model._resolve_processor("dynamic_processor", "dynamic_projection")
        fp = model._resolve_processor("future_processor", "future_projection")
        static_context = _invoke(sp, static_x, training=False)
        dynamic_processed = _invoke(dp, dynamic_x, training=False)
        future_processed = _invoke(fp, future_x, training=False)
        encoder_input = model.backend_context.concat([dynamic_processed], axis=-1)
        encoder_input = _invoke(model._assembly.encoder_positional_encoding, encoder_input, training=False)
        encoder_sequences = _invoke(model._assembly.dynamic_encoder, encoder_input, training=False)
        encoder_sequences = _invoke(model._assembly.dynamic_window, encoder_sequences, training=False)
        static_expanded = model.backend_context.expand_dims(static_context, axis=1)
        static_expanded = model.backend_context.tile(
            static_expanded,
            [1, model.spec.forecast_horizon, 1],
        )
        decoder_future = _invoke(
            model._assembly.future_positional_encoding,
            future_processed,
            training=False,
        )
        raw_decoder_input = model.backend_context.concat(
            [static_expanded, decoder_future],
            axis=-1,
        )
        projected_decoder_input = _invoke(
            model._assembly.decoder_input_projection,
            raw_decoder_input,
            training=False,
        )
        final_sequence = model._apply_decoder_stack(
            projected_decoder_input,
            encoder_sequences,
            training=False,
        )
        payload = json.loads(model.to_json())
        clone = keras.models.model_from_json(model.to_json())
        summary = {
            "final_sequence_shape": list(getattr(final_sequence, "shape", ())),
            "sequence_pooling": payload["config"]["spec"]["components"]["sequence_pooling"],
            "clone_backend": clone.spec.backend_name,
            "cross_class": type(model._assembly.decoder_cross_attention).__name__,
            "fusion_class": type(model._assembly.decoder_fusion).__name__,
        }
        print(json.dumps(summary))
        """
    )
    completed = _run_backend_script("jax", code)
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(
        completed.stdout.strip().splitlines()[-1]
    )
    assert payload["final_sequence_shape"] == [2, 4, 32]
    assert payload["sequence_pooling"] == "pool.last"
    assert payload["clone_backend"] == "jax"
    assert payload["cross_class"].startswith("_Jax")
    assert payload["fusion_class"].startswith("_Jax")


def test_tensorflow_import_matrix_smoke():
    code = textwrap.dedent(
        """
        import json
        import importlib.util
        import base_attentive as batt
        from base_attentive import BaseAttentive

        print(
            json.dumps(
                {
                    "backend": batt.KERAS_BACKEND,
                    "tensorflow_installed": importlib.util.find_spec("tensorflow") is not None,
                    "base_name": BaseAttentive.__name__,
                }
            )
        )
        """
    )
    completed = _run_backend_script("tensorflow", code)
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(
        completed.stdout.strip().splitlines()[-1]
    )
    assert payload["backend"] == "tensorflow"
    assert payload["base_name"] == "BaseAttentive"
