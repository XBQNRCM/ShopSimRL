"""
Thin source-tree entry point; equivalent to ``python -m shopsimrl.cli``.

# start ShopSimulator service
cd C:\project\ShopSimRL\ShopSimulator\shop_env
$env:SHOPSIM_ENV_SLOTS = "16"
.\start.ps1

Invoke-RestMethod http://127.0.0.1:5700/healthz

# run experiment
cd C:\project\ShopSimRL
python scripts\run_shopsimrl.py run configs\qwen35_4b_test.yaml

# replay trajectories
http://127.0.0.1:5700/replay-ui
"""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shopsimrl.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
