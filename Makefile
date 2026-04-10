.PHONY: help bootstrap install install-dev install-tensorflow install-jax install-torch install-all test test-fast test-core test-backend lint format type-check check build release-check clean docs

PYTHON ?= python
PIP := $(PYTHON) -m pip
PACKAGE := base_attentive
SRC := src tests

.DEFAULT_GOAL := help

help:
	@echo "Base-Attentive development commands"
	@echo ""
	@echo "Environment"
	@echo "  make bootstrap          Upgrade pip/build tooling"
	@echo "  make install            Install package in editable mode"
	@echo "  make install-dev        Install editable package with dev extras"
	@echo "  make install-tensorflow Install editable package with TensorFlow runtime"
	@echo "  make install-jax        Install editable package with JAX runtime"
	@echo "  make install-torch      Install editable package with Torch runtime"
	@echo "  make install-all        Install editable package with all runtimes"
	@echo ""
	@echo "Quality"
	@echo "  make test               Run full pytest suite"
	@echo "  make test-fast          Run a faster local pytest pass"
	@echo "  make test-core          Run core/unit-focused tests"
	@echo "  make test-backend       Run backend/runtime focused tests"
	@echo "  make lint               Run Ruff lint checks"
	@echo "  make format             Apply Ruff formatting and import fixes"
	@echo "  make type-check         Run mypy"
	@echo "  make check              Run lint, type-check, and fast tests"
	@echo ""
	@echo "Packaging"
	@echo "  make build              Build sdist and wheel"
	@echo "  make release-check      Validate built distributions with twine"
	@echo "  make clean              Remove common build/test artifacts"
	@echo "  make docs               Build docs if a docs/ directory exists"

bootstrap:
	$(PIP) install --upgrade pip setuptools wheel build twine

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"

install-tensorflow:
	$(PIP) install -e ".[dev,tensorflow]"

install-jax:
	$(PIP) install -e ".[dev,jax]"

install-torch:
	$(PIP) install -e ".[dev,torch]"

install-all:
	$(PIP) install -e ".[dev,all-backends]"

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

test-fast:
	$(PYTHON) -m pytest tests/ -q -x --disable-warnings

test-core:
	$(PYTHON) -m pytest tests/test_core.py tests/test_base_attentive_main.py -q

test-backend:
	$(PYTHON) -m pytest tests/test_backend.py tests/test_runtime_shim.py tests/test_imports.py -k "backend or validation or runtime" -q

lint:
	$(PYTHON) -m ruff check $(SRC)
	$(PYTHON) -m ruff format --check $(SRC)

format:
	$(PYTHON) -m ruff check --fix $(SRC)
	$(PYTHON) -m ruff format $(SRC)

type-check:
	$(PYTHON) -m mypy src/$(PACKAGE) --ignore-missing-imports

check: lint type-check test-fast

build:
	$(PYTHON) -m build

release-check: build
	$(PYTHON) -m twine check dist/*

clean:
	$(PYTHON) -c "from pathlib import Path; import shutil; targets=['build','dist','.pytest_cache','.mypy_cache','.ruff_cache','htmlcov']; files=['coverage.xml','tests.log']; [shutil.rmtree(Path(t), ignore_errors=True) for t in targets]; [Path(f).unlink(missing_ok=True) for f in files]"
	$(PYTHON) -c "from pathlib import Path; import shutil; [shutil.rmtree(p, ignore_errors=True) for p in Path('.').rglob('__pycache__')]"
	$(PYTHON) -c "from pathlib import Path; [p.unlink(missing_ok=True) for p in Path('.').rglob('*.pyc')]"

docs:
	@if [ -d docs ]; then \
		$(PYTHON) -m sphinx -W -b html -d docs/_build/doctrees docs docs/_build/html; \
	else \
		echo "docs/ directory is missing. Add documentation sources before using 'make docs'."; \
	fi
