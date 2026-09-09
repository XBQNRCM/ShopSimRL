$ErrorActionPreference = "Stop"
$shopEnvRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$defaultIndex = Join-Path $shopEnvRoot "search_engine\products.sqlite3"
$indexPath = if ($env:SHOP_SEARCH_INDEX) { $env:SHOP_SEARCH_INDEX } else { $defaultIndex }

if (-not (Test-Path -LiteralPath $indexPath -PathType Leaf)) {
    throw "ShopSimulator index is missing: $indexPath. Run: python scripts/prepare_data.py"
}

if (-not $env:SHOP_ENV_CONFIG) {
    $env:SHOP_ENV_CONFIG = Join-Path $shopEnvRoot "configs\environment.json"
}
$env:SHOP_SEARCH_INDEX = $indexPath
if (-not $env:SHOPSIM_ENV_SLOTS) { $env:SHOPSIM_ENV_SLOTS = "8" }
if (-not $env:SHOPSIM_PORT) { $env:SHOPSIM_PORT = "5700" }

Set-Location -LiteralPath $shopEnvRoot
python -m shop_env.pack_api
