"""Sphinx configuration for BaseAttentive."""

from __future__ import annotations

import importlib.util
import re
import shutil
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PACKAGE_INIT = SRC / "base_attentive" / "__init__.py"

sys.path.insert(0, str(SRC))

# ---------------------------------------------------------------------------
# Auto-copy notebooks from examples/ into docs/notebooks/ before Sphinx runs.
# This keeps examples/ as the single source of truth while making the
# notebooks reachable by nbsphinx (which requires files inside the source dir).
# ---------------------------------------------------------------------------
_EXAMPLES_DIR = ROOT / "examples"
_NOTEBOOKS_DIR = Path(__file__).parent / "notebooks"
_NOTEBOOKS_DIR.mkdir(exist_ok=True)
for _nb in _EXAMPLES_DIR.glob("*.ipynb"):
    shutil.copy(_nb, _NOTEBOOKS_DIR / _nb.name)


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
    "sphinx_copybutton",
    "nbsphinx",
]

# nbsphinx — never re-execute notebooks during the doc build.
# Cells are rendered as-is (with any saved outputs).
# On ReadTheDocs there is no ML runtime available, so execution must be off.
nbsphinx_execute = "never"

# Allow nbsphinx to process notebooks even when they have no outputs.
nbsphinx_allow_errors = False

root_doc = "index"
templates_path: list[str] = ["_templates"]
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
    if importlib.util.find_spec("sphinx_rtd_theme")
    is not None
    else "alabaster"
)
html_title = f"{project} {release} documentation"
html_logo = "_static/base-attentive-logo.svg"
html_favicon = "_static/base-attentive.ico"
html_static_path: list[str] = ["_static"]
html_css_files: list[str] = ["custom.css"]
html_theme_options = {}
html_context = {
    "author_name": author,
    "author_url": "https://lkouadio.com/",
    "copyright_year": str(date.today().year),
}

# Sidebars configuration
# Sidebars configuration
html_sidebars = {
    "**": [
        "localtoc.html",
        "relations.html",
        "sourcelink.html",
        "searchbox.html",
    ]
}

# Badge and substitution definitions for reStructuredText
rst_prolog = """
.. |Feature| image:: https://img.shields.io/badge/type-feature-brightgreen.svg
   :alt: Feature

.. |Fix| image:: https://img.shields.io/badge/type-fix-blue.svg
   :alt: Bug Fix

.. |Bug| image:: https://img.shields.io/badge/type-bug_fix-blue.svg
   :alt: Bug Fix

.. |Bugfix| image:: https://img.shields.io/badge/type-bugfix-blue.svg
   :alt: Bug Fix

.. |Dependencies| image:: https://img.shields.io/badge/type-dependencies-orange.svg
   :alt: Dependencies

.. |Internal| image:: https://img.shields.io/badge/type-internal-gray.svg
   :alt: Internal

.. |MAJOR| image:: https://img.shields.io/badge/version-MAJOR-red.svg
   :alt: MAJOR Version Change

.. |MINOR| image:: https://img.shields.io/badge/version-MINOR-yellow.svg
   :alt: MINOR Version Change

.. |PATCH| image:: https://img.shields.io/badge/version-PATCH-green.svg
   :alt: PATCH Version Change

.. |Breaking| image:: https://img.shields.io/badge/breaking-yes-red.svg
   :alt: Breaking Change

.. |Deprecated| image:: https://img.shields.io/badge/status-deprecated-yellow.svg
   :alt: Deprecated

.. |Security| image:: https://img.shields.io/badge/type-security-critical.svg
   :alt: Security
"""
