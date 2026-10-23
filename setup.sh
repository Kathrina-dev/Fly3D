#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${FLY3D_ENV_NAME:-fly3d}"

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda is required. Install Miniconda, restart your shell, then run this again." >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda env create --file environment.yml --name "$ENV_NAME" || conda env update --file environment.yml --name "$ENV_NAME"
conda activate "$ENV_NAME"
python -m pip install --upgrade pip
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu118 \
  torch==2.1.2+cu118 torchvision==0.16.2+cu118
python -m pip install -e .

echo "Fly3D is ready. Activate it with: conda activate $ENV_NAME"
echo "Check the installation with: fly3d doctor"