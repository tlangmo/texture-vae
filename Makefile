.PHONY: all
SHELL := /bin/bash

DIRS = texture_vae


tb::
	tensorboard --logdir ./tb/runs

setup:
	uv venv --system-site-packages
	source  ./.venv/bin/activate  && uv sync

http:
	npx light-server -s . -p 8080

fmt:: isort black

isort::
	python3 -m isort --profile black $(DIRS)

black::
	python3 -m black  $(DIRS)

