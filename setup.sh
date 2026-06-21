#!/usr/bin/env bash
# One-shot environment setup for TokenC.
set -euo pipefail
cd "$(dirname "$0")"

python3 -m venv .venv
./.venv/bin/python -m pip install -q --upgrade pip
./.venv/bin/python -m pip install -q -r requirements.txt
./.venv/bin/python -m ipykernel install --user --name tokenc --display-name "TokenC (.venv)"

echo
echo "Setup done. Next:"
echo "  1) echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env"
echo "  2) ./.venv/bin/python distill.py --n 240        # distill labels from Claude"
echo "  3) ./.venv/bin/python train_compressor.py        # train the keep/drop model"
echo "  4) ./.venv/bin/python build_notebook.py && ./.venv/bin/jupyter notebook demo.ipynb"
