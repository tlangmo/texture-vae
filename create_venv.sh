#!/usr/bin/env bash

set -euf -o pipefail  # make it strict

cd "$(dirname "$0")"

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Use Linux installer."
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Use MacOS installer."
else
    >&2 echo "Unsupported OSTYPE: ${OSTYPE}"
    exit 1
fi

if [[ "$(python3 --version)" == "Python 3.8."* ]] ; then
    echo "Use $(python3 --version)"
else
    >&2 echo "Unsupported version: $(python3 --version)"
    >&2 echo "Expected version: Python 3.8.x"
    exit 1
fi

2> /dev/null deactivate || true

if [ -d .venv ] && [ ! -z "${KEEP_VENV:-}" ]; then
    echo "Update existing venv."
else
    echo "Create new venv."
    rm -rf .venv
    mkdir .venv
    python3 -m venv .venv
fi

source .venv/bin/activate

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    export TORCH_VARIANT="+cu111"
fi

set -o xtrace # show commands

# due to https://github.com/jazzband/pip-tools/issues/1558, we need pin pip to an older version!

pip3 install --upgrade isort pur==5.4.0 black==21.12b0 flake8 shapely==1.8.2 rtree==1.0.0

pip3 install --no-index --no-deps torch==1.10.2${TORCH_VARIANT} torchvision==0.11.3${TORCH_VARIANT} \
    -f https://download.pytorch.org/whl/torch_stable.html
set +o xtrace # hide commands
set +o xtrace # hide commands

source .venv/bin/activate

echo "Done."
