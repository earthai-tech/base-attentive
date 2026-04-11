"""Implementation packages for V2 backend-specific builders.

This package provides backend-specific implementations which register
their components with the registry when explicitly imported.

Note: Backend modules are NOT auto-imported to avoid circular dependencies.
Instead, they're loaded on-demand when the resolver needs them.
"""

__all__ = []
