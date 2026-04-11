CONFIG ?= config/3-layer/exact/double.json
EPOCH ?= 100

.PHONY: sync train format

sync:
	uv sync

train:
	uv run python train_main.py $(CONFIG) $(EPOCH)

format:
	uvx ruff format .
