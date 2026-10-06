# Windows PowerShell: set up, seed and start the app locally (no Docker, no keys).
#   .\scripts\run_local.ps1            # first run creates .venv, installs deps, seeds, starts on :8000
#   .\scripts\run_local.ps1 -Reset     # wipe local.db and .qdrant first
#   .\scripts\run_local.ps1 -NoHF     # skip importing the Hugging Face dataset (offline / quicker first start)
#   .\scripts\run_local.ps1 -Port 8001
#   .\scripts\run_local.ps1 -Embeddings sentence-transformers -Reset   # real MiniLM embeddings (downloads ~90 MB once)
# If scripts are blocked:  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
param([switch]$Reset, [switch]$NoHF, [int]$Port = 8000,
      [ValidateSet("hashing", "sentence-transformers")][string]$Embeddings = "hashing")
# Not "Stop": Windows PowerShell 5.1 turns harmless stderr warnings of native commands into fatal errors.
# Failures are caught explicitly through exit codes instead.
function Check($what) { if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $what" -ForegroundColor Red; exit 1 } }
Set-Location (Split-Path -Parent $PSScriptRoot)

if (-not (Test-Path .venv)) {
    Write-Host "Creating virtual environment (.venv) ..."
    py -3.12 -m venv .venv 2>$null
    if ($LASTEXITCODE -ne 0) { python -m venv .venv }
}
& .\.venv\Scripts\Activate.ps1
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements-dev.txt; Check "pip install -r requirements-dev.txt"
if ($Embeddings -eq "sentence-transformers") {
    python -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
    python -m pip install --quiet -r requirements-ml.txt; Check "pip install -r requirements-ml.txt"
}

# environment (read by pydantic-settings; a .env file would also work)
$env:APP_ENV = "dev"; $env:AUTH_MODE = "jwt"; $env:JWT_SECRET = "local-dev-secret"
$env:DATABASE_URL = "sqlite:///./local.db"; $env:QDRANT_URL = "path:./.qdrant"
$env:EMBEDDING_BACKEND = $Embeddings; $env:CELERY_EAGER = "true"
$env:PYTHONUTF8 = "1"   # UTF-8 for files and console output on Windows

if ($Embeddings -ne "hashing" -and (Test-Path local.db) -and -not $Reset) {
    Write-Host "Note: switching embedder on an existing index? Re-run with -Reset so vectors are rebuilt." -ForegroundColor Yellow
}
if ($Reset) { Remove-Item -Recurse -Force local.db, .qdrant -ErrorAction SilentlyContinue }
python -m scripts.seed; Check "seeding"

# Knowledge base from the Hugging Face dataset too (tickets + derived KB articles). Needs internet once; failures only warn.
if (-not $NoHF) {
    $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
    python -m pip install --quiet -r requirements-hf.txt
    if ($LASTEXITCODE -eq 0) {
        python -m data.load_hf
        if ($LASTEXITCODE -ne 0) { Write-Host "Hugging Face import skipped (see message above). The app still works with the bundled telecom data; retry later with: python -m data.load_hf" -ForegroundColor Yellow }
    } else { Write-Host "Could not install the 'datasets' package; skipping the Hugging Face import." -ForegroundColor Yellow }
}
Write-Host "`nOpen http://localhost:$Port/  (sign in as Admin; API docs at /docs)`n"
# embedded Qdrant is single-process: no --reload, one worker
python -m uvicorn app.api.main:app --port $Port
