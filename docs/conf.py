"""Sphinx configuration for BaseAttentive."""

from __future__ import annotations

import importlib.util
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PACKAGE_INIT = SRC / "base_attentive" / "__init__.py"

sys.path.insert(0, str(SRC))


def _read_version() -> str:
    match = re.search(
        r'^__version__\s*=\s*["\']([^"\']+)["\']',
        PACKAGE_INIT.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    if match:
        return match.group(1)
    return "0.0.0"


project = "BaseAttentive"
author = "Laurent Kouadio"
copyright = f"{date.today().year}, {author}"
version = _read_version()
release = version

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.coverage",
]

root_doc = "index"
templates_path: list[str] = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True
autoclass_content = "both"
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "undoc-members": False,
}
autodoc_mock_imports = [
    "tensorflow",
    "keras",
    "jax",
    "jaxlib",
    "torch",
]

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

html_theme = (
    "sphinx_rtd_theme"
    if importlib.util.find_spec("sphinx_rtd_theme") is not None
    else "alabaster"
)
html_title = f"{project} {release} documentation"
html_static_path: list[str] = []
html_theme_options = {}

# Sidebars configuration
html_sidebars = {
    "**": ["localtoc.html", "relations.html", "sourcelink.html", "searchbox.html"]
}
