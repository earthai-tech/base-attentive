"""
base_attentive: A foundational blueprint for sequence-to-sequence
time series forecasting models with attention mechanisms.
"""

__version__ = "0.1.0"
__author__ = "LKouadio"
__email__ = "etanoyau@gmail.com"
__license__ = "Apache-2.0"

try:
    from base_attentive.core import BaseAttentive
    __all__ = ["BaseAttentive"]
except ImportError:
    __all__ = []
    import warnings
    warnings.warn(
        "Failed to import BaseAttentive. Ensure TensorFlow/Keras is installed.",
        RuntimeWarning
    )
