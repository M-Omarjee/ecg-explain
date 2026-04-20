.PHONY: install sync data smoke train eval explain demo test lint typecheck format clean help

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	uv sync --all-extras

sync: install  ## Alias for install

data:  ## Download PTB-XL dataset (~1GB, ~30 min)
	uv run python scripts/download_data.py

smoke:  ## Train a tiny model for 2 epochs (sanity check)
	uv run python scripts/train.py configs/smoke.yaml

train:  ## Train the baseline model
	uv run python scripts/train.py configs/baseline.yaml

eval:  ## Evaluate the trained model on the test set
	uv run python scripts/evaluate.py \
		--config configs/baseline.yaml \
		--checkpoint checkpoints/baseline/best.pt \
		--output results/baseline_test_metrics.json

explain:  ## Generate a Grad-CAM explanation for the first test record
	uv run python scripts/explain.py \
		--config configs/baseline.yaml \
		--checkpoint checkpoints/baseline/best.pt \
		--record-idx 0 --target-class MI \
		--output figures/example_explanation.png

case-studies:  ## Generate case study figures for the README
	uv run python scripts/build_case_studies.py \
		--config configs/baseline.yaml \
		--checkpoint checkpoints/baseline/best.pt

demo:  ## Run the Gradio demo locally
	uv run python app/app.py

test:  ## Run the full test suite
	uv run pytest -v

lint:  ## Run ruff lint check
	uv run ruff check .

typecheck:  ## Run mypy
	uv run mypy

format:  ## Auto-format with ruff
	uv run ruff format .
	uv run ruff check --fix .

clean:  ## Remove caches and build artefacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache __pycache__ build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
