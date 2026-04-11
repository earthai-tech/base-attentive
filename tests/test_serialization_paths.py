"""Static tests for Keras serialization package paths."""

from __future__ import annotations

import ast
from pathlib import Path

SRC_ROOT = Path(__file__).parent.parent / "src" / "base_attentive"
SERIALIZABLE_MODULES = [
    SRC_ROOT / "core" / "base_attentive.py",
    SRC_ROOT / "components" / "attention.py",
    SRC_ROOT / "components" / "encoder_decoder.py",
    SRC_ROOT / "components" / "gating_norm.py",
    SRC_ROOT / "components" / "heads.py",
    SRC_ROOT / "components" / "layer_utils.py",
    SRC_ROOT / "components" / "losses.py",
    SRC_ROOT / "components" / "misc.py",
    SRC_ROOT / "components" / "temporal.py",
    SRC_ROOT / "components" / "_loss_utils.py",
    SRC_ROOT / "components" / "_temporal_utils.py",
]


def _parse_module(path: Path) -> ast.Module:
    source = path.read_text(encoding="utf-8", errors="ignore")
    return ast.parse(source, filename=str(path))


def _has_serialization_package_alias(tree: ast.Module) -> bool:
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if target.id != "SERIALIZATION_PACKAGE":
            continue
        return isinstance(node.value, ast.Name) and node.value.id == "__name__"
    return False


def _register_decorators(tree: ast.Module):
    for node in ast.walk(tree):
        if not isinstance(
            node,
            (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            func = decorator.func
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            else:
                name = None
            if name == "register_keras_serializable":
                yield decorator


def test_register_keras_serializable_uses_real_module_paths():
    """Each serializable module should register objects under its own path."""
    for path in SERIALIZABLE_MODULES:
        tree = _parse_module(path)

        assert _has_serialization_package_alias(tree), (
            f"{path} should define SERIALIZATION_PACKAGE = __name__"
        )

        decorators = list(_register_decorators(tree))
        assert decorators, f"{path} should contain serializable decorators"

        for decorator in decorators:
            assert decorator.args, (
                f"{path} has a register_keras_serializable decorator without "
                "a package argument"
            )
            first_arg = decorator.args[0]
            assert isinstance(first_arg, ast.Name), (
                f"{path} should pass SERIALIZATION_PACKAGE as the first "
                "decorator argument"
            )
            assert first_arg.id == "SERIALIZATION_PACKAGE", (
                f"{path} should register under its real module path"
            )
