.PHONY: all
SHELL := /bin/bash

DIRS = texture_vae


tb::
	tensorboard --logdir ./tb/runs

setup:
	./create_venv.sh
	source  ./.venv/bin/activate && poetry install
	echo "use 'poetry shell' to activate virtual environment"

http:
	npx light-server -s . -p 8080

fmt:: isort black

isort::
	python3 -m isort --profile black $(DIRS)

black::
	python3 -m black  $(DIRS)

