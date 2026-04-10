"""Utilities for input validation."""

try:
    import tensorflow as tf

    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False


def validate_model_inputs(
    inputs,
    static_input_dim,
    dynamic_input_dim,
    future_covariate_dim,
    forecast_horizon,
    mode="strict",
    verbose=0,
):
    """
    Validates and unpacks model inputs.
    
    Parameters
    ----------
    inputs : tuple or list
        The input tuple/list containing [static, dynamic, future]
    static_input_dim : int
        Expected static feature dimension
    dynamic_input_dim : int
        Expected dynamic feature dimension
    future_covariate_dim : int
        Expected future feature dimension
    forecast_horizon : int
        Expected forecast horizon
    mode : str, default "strict"
        Validation mode
    verbose : int, default 0
        Verbosity level
        
    Returns
    -------
    tuple
        (static_input, dynamic_input, future_input)
    """
    if not HAS_TENSORFLOW:
        raise ImportError("TensorFlow is required for model validation")
    
    if not isinstance(inputs, (list, tuple)) or len(inputs) != 3:
        raise ValueError(
            f"Expected inputs to be a list/tuple of 3 tensors "
            f"[static, dynamic, future], got {len(inputs)} tensors"
        )
    
    static_p, dynamic_p, future_p = inputs
    
    # Basic shape validation
    if len(static_p.shape) != 2:
        raise ValueError(
            f"Static input must be 2D (batch, features), "
            f"got shape {static_p.shape}"
        )
    
    if len(dynamic_p.shape) != 3:
        raise ValueError(
            f"Dynamic input must be 3D (batch, time, features), "
            f"got shape {dynamic_p.shape}"
        )
    
    if len(future_p.shape) != 3:
        raise ValueError(
            f"Future input must be 3D (batch, time, features), "
            f"got shape {future_p.shape}"
        )
    
    # Check feature dimensions
    if static_p.shape[-1] != static_input_dim and static_input_dim > 0:
        raise ValueError(
            f"Static features mismatch: expected {static_input_dim}, "
            f"got {static_p.shape[-1]}"
        )
    
    if dynamic_p.shape[-1] != dynamic_input_dim:
        raise ValueError(
            f"Dynamic features mismatch: expected {dynamic_input_dim}, "
            f"got {dynamic_p.shape[-1]}"
        )
    
    if future_p.shape[-1] != future_covariate_dim:
        raise ValueError(
            f"Future features mismatch: expected {future_covariate_dim}, "
            f"got {future_p.shape[-1]}"
        )
    
    if verbose > 0:
        print(f"✓ Input validation passed")
        print(f"  Static: {static_p.shape}")
        print(f"  Dynamic: {dynamic_p.shape}")
        print(f"  Future: {future_p.shape}")
    
    return static_p, dynamic_p, future_p


__all__ = ["validate_model_inputs"]
