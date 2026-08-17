# Grid Up Datathon - ortam kurulumu (Windows PowerShell)
#
# Calistirma:
#   powershell -ExecutionPolicy Bypass -File setup.ps1
#
# NOT: Bu betik Windows PowerShell 5.1 ile uyumlu yazilmistir.
# 5.1'de '&&', '||', '??' ve ternary operatorleri PARSER HATASI verir -- kullanilmaz.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host ""
Write-Host "=== Grid Up Datathon - kurulum ===" -ForegroundColor Cyan
Write-Host ""

# --- 1. Python kontrolu -----------------------------------------------------
# 'python3' bu platformda Microsoft Store kisayoluna gider ve calismaz. 'python' kullan.
$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
if ($null -eq $pythonCommand) {
    Write-Host "HATA: 'python' bulunamadi. Python 3.10+ kurun ve PATH'e ekleyin." -ForegroundColor Red
    exit 1
}

$version = & python --version
Write-Host "Python: $version"

# --- 2. Kilitli bagimliliklar -----------------------------------------------
Write-Host ""
Write-Host "Bagimliliklar kuruluyor..." -ForegroundColor Yellow
$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
if ($null -eq $uvCommand) {
    Write-Host "uv 0.12.5 kuruluyor..." -ForegroundColor Yellow
    & python -m pip install --require-hashes -r requirements/uv-bootstrap.txt --quiet
    if ($LASTEXITCODE -ne 0) {
        Write-Host "HATA: uv kurulumu basarisiz." -ForegroundColor Red
        exit 1
    }
}
& uv sync --locked --extra full --extra dev
if ($LASTEXITCODE -ne 0) {
    Write-Host "HATA: uv.lock ile tam eslesen ortam kurulamadi." -ForegroundColor Red
    exit 1
}
Write-Host "Kilitli bagimliliklar tamam." -ForegroundColor Green

# --- 3. Dizinler ------------------------------------------------------------
$directories = @(
    "data\raw", "data\interim", "data\external", "data\features",
    "models", "submissions", "experiments", "reports\figures"
)
foreach ($directory in $directories) {
    if (-not (Test-Path $directory)) {
        New-Item -ItemType Directory -Path $directory -Force | Out-Null
    }
}
Write-Host "Dizinler hazir." -ForegroundColor Green

# --- 4. Notebook cikti temizleyici ------------------------------------------
# Notebook ciktilarinin commit'e girmesi hem diff'i okunamaz kilar hem de
# yarisma verisini kazara repoya sizdirabilir.
if (Test-Path ".git") {
    & uv run python -m nbstripout --install 2>$null
    if ($?) {
        Write-Host "nbstripout kuruldu (notebook ciktilari commit'e girmeyecek)." -ForegroundColor Green
    }
}

# --- 5. Dogrulama -----------------------------------------------------------
Write-Host ""
Write-Host "Testler calistiriliyor..." -ForegroundColor Yellow
$env:PYTHONIOENCODING = "utf-8"
& uv run python -m pytest -q
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "UYARI: bazi testler basarisiz. Devam etmeden once bakin." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=== Kurulum tamam ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Sonraki adimlar:"
Write-Host "  python scripts\smoke_test.py                    # uctan uca dogrulama (~60 sn)"
Write-Host "  python scripts\fetch_weather.py                 # hava durumu verisi indir"
Write-Host "  jupyter lab notebooks\01_kesif.ipynb            # kesif notebook'u"
Write-Host ""
