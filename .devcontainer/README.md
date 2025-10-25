# Dev Container Setup

This dev container is configured for aarch64/arm64 systems and uses `uv` for fast Python package management.

## Features

- Python 3.10 (compatible with aarch64)
- uv package manager (much faster than pip)
- VSCode extensions: Python, Pylance, Ruff, Jupyter
- Non-root user (vscode)
- TensorBoard port forwarding (6006)

## Usage

### Opening in VSCode

1. Install the "Dev Containers" extension in VSCode
2. Open this folder in VSCode
3. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
4. Select "Dev Containers: Reopen in Container"
5. Wait for the container to build and dependencies to install

### Using uv

The container automatically runs `uv sync` after creation to install all dependencies.

Common uv commands:
```bash
# Install dependencies
uv sync

# Install dev dependencies
uv sync --extra dev

# Add a new dependency
uv add <package-name>

# Add a dev dependency
uv add --dev <package-name>

# Run a script
uv run python -m texture_vae

# Run installed CLI tools
uv run vae-crop-gen
uv run vae-onnx
```

## Migration from Poetry

This project has been converted from Poetry to use the standard PEP 621 format with uv:

- `poetry.lock` → `uv.lock` (generated on first sync)
- `poetry install` → `uv sync`
- `poetry add` → `uv add`
- `poetry run` → `uv run`

Note: PyTorch and torchvision have been added to the dependencies as they were missing from the original setup.
