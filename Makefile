# MSRCF task runner. Run `make help` for the list of targets.
# Works with GNU make. On Windows, use `make` from Git Bash / WSL, or run
# the underlying python commands directly (see README).

PYTHON ?= python

.DEFAULT_GOAL := help

.PHONY: help install run tune test cov lint format typecheck experiments ablation all clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install runtime + dev dependencies
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e ".[dev]"

run:  ## Run the full pipeline (500 components)
	$(PYTHON) src/main.py

tune:  ## Run the pipeline with GridSearchCV refinement
	$(PYTHON) src/main.py --tune-best

experiments:  ## Multi-seed statistical benchmark + Friedman/Nemenyi
	$(PYTHON) src/experiments.py --n-seeds 15

ablation:  ## Feature + design-choice ablation study
	$(PYTHON) src/ablation.py --n-seeds 8

test:  ## Run the test suite
	$(PYTHON) -m pytest

cov:  ## Run tests with coverage report
	$(PYTHON) -m pytest --cov=src --cov-report=term-missing

lint:  ## Ruff lint + import-order check
	$(PYTHON) -m ruff check src tests

format:  ## Auto-format with ruff
	$(PYTHON) -m ruff format src tests

typecheck:  ## Static type check with mypy
	$(PYTHON) -m mypy src

all: lint typecheck test  ## Lint, type-check, and test

clean:  ## Remove generated artifacts and caches
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
