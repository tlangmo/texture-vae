#!/usr/bin/env bash

set -euf -o pipefail  # make it strict

cd "$(dirname "$0")"

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

pip3 install --upgrade pip
pip3 install --no-index --no-deps torch==1.10.2${TORCH_VARIANT} torchvision==0.11.3${TORCH_VARIANT} \
    -f https://download.pytorch.org/whl/torch_stable.html
set +o xtrace # hide commands

source .venv/bin/activate

echo "Done."
