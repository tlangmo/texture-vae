.PHONY: all

tb::
	tensorboard --logdir tensorboard/runs

setup:
	poetry build
	tar -xzf dist/texture_vae-*.tar.gz -C dist/
	mv dist/texture_vae-*/setup.py .
	pip3 install -e .

http:
	npx light-server -s . -p 8080