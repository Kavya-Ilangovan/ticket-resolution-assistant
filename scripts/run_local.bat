@echo off
REM Windows cmd.exe equivalent of run_local.ps1 (no Docker, no keys).
REM   scripts\run_local.bat                         -> offline hashing embedder
REM   scripts\run_local.bat hashing nohf                  -> skip the Hugging Face import
REM   scripts\run_local.bat sentence-transformers    -> real embeddings (pip install torch + requirements-ml.txt first)
cd /d "%~dp0\.."
if not exist .venv ( py -3.12 -m venv .venv || python -m venv .venv )
call .venv\Scripts\activate.bat
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements-dev.txt
set APP_ENV=dev
set AUTH_MODE=jwt
set JWT_SECRET=local-dev-secret
set DATABASE_URL=sqlite:///./local.db
set QDRANT_URL=path:./.qdrant
if "%1"=="" ( set EMBEDDING_BACKEND=hashing ) else ( set EMBEDDING_BACKEND=%1 )
set CELERY_EAGER=true
set PYTHONUTF8=1
python -m scripts.seed
if /I not "%2"=="nohf" (
  set HF_HUB_DISABLE_SYMLINKS_WARNING=1
  python -m pip install --quiet -r requirements-hf.txt && python -m data.load_hf || echo Hugging Face import skipped - the app still works; retry later with: python -m data.load_hf
)
echo.
echo Open http://localhost:8000/  (sign in as Admin; API docs at /docs)
python -m uvicorn app.api.main:app --port 8000
