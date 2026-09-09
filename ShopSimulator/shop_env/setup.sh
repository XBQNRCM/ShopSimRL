#!/usr/bin/env bash
set -euo pipefail

SHOP_ENV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SHOP_ENV_ROOT}"

python -m pip install -r requirements.txt
if ! python -c "import zh_core_web_sm" >/dev/null 2>&1; then
  python -m spacy download zh_core_web_sm
fi
python scripts/prepare_data.py

echo "ShopSimulator data and search index are ready."
