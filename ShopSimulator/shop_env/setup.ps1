$ErrorActionPreference = "Stop"
$shopEnvRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $shopEnvRoot

python -m pip install -r requirements.txt
python -c "import zh_core_web_sm" 2>$null
if ($LASTEXITCODE -ne 0) {
    python -m spacy download zh_core_web_sm
}
python scripts/prepare_data.py

Write-Host "ShopSimulator data and search index are ready."
