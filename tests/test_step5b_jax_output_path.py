from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


SRC_DIR = Path(__file__).resolve().parents[1] / 'src'


def _run_backend_script(backend: str, code: str, *, timeout: int = 240):
    env = dict(os.environ)
    env['BASE_ATTENTIVE_BACKEND'] = backend
    env['KERAS_BACKEND'] = backend
    env['PYTHONPATH'] = os.pathsep.join(
        filter(None, [str(SRC_DIR), env.get('PYTHONPATH')])
    )
    if backend == 'jax':
        env.setdefault('JAX_DISABLE_JIT', '1')
    return subprocess.run(
        [sys.executable, '-c', code],
        text=True,
        capture_output=True,
        env=env,
        timeout=timeout,
    )


@pytest.mark.skipif(
    importlib.util.find_spec('jax') is None,
    reason='JAX is not installed',
)
def test_jax_full_forward_clone_and_reload_path():
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
            return np.asarray(value)

        model = BaseAttentiveV2(
            static_input_dim=2,
            dynamic_input_dim=3,
            future_input_dim=1,
            output_dim=2,
            forecast_horizon=4,
            quantiles=(0.1, 0.5, 0.9),
            head_type='quantile',
            spec={
                'architecture': {
                    'sequence_pooling': 'pool.last',
                    'fusion': 'fusion.concat',
                },
                'runtime': {'final_agg': 'average'},
            },
        )
        inputs = [
            np.ones((2, 2), dtype='float32'),
            np.ones((2, 5, 3), dtype='float32'),
            np.ones((2, 4, 1), dtype='float32'),
        ]
        outputs = model(inputs, training=False)

        clone = keras.models.model_from_json(model.to_json())
        _ = clone(inputs, training=False)
        clone.set_weights(model.get_weights())
        clone_outputs = clone(inputs, training=False)

        path = os.path.join(tempfile.mkdtemp(), 'model.keras')
        model.save(path)
        loaded = keras.models.load_model(path)
        loaded_outputs = loaded.call(inputs, training=False)

        model_weights = model.get_weights()
        loaded_weights = loaded.get_weights()
        weight_delta = max(
            float(np.max(np.abs(a - b)))
            for a, b in zip(model_weights, loaded_weights)
        ) if model_weights and loaded_weights else 0.0

        summary = {
            'shape': list(getattr(outputs, 'shape', ())),
            'clone_delta': float(
                np.max(np.abs(to_numpy(outputs) - to_numpy(clone_outputs)))
            ),
            'load_delta': float(
                np.max(np.abs(to_numpy(outputs) - to_numpy(loaded_outputs)))
            ),
            'weight_delta': weight_delta,
            'loaded_built': bool(getattr(loaded, 'built', False)),
        }
        print(json.dumps(summary))
        """
    )
    completed = _run_backend_script('jax', code)
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload['shape'] == [2, 4, 2, 3]
    assert payload['clone_delta'] < 1e-6
    assert payload['load_delta'] < 1e-6
    assert payload['weight_delta'] < 1e-6
    assert payload['loaded_built'] is True
