@echo off
echo ========================================================
echo        Starting Signova AI (Backend + Frontend)
echo ========================================================
cd /d "%~dp0"

echo [1/2] Starting FastAPI Backend on http://127.0.0.1:8000 ...
start "Signova AI - Backend" cmd /k ".venv\Scripts\activate && uvicorn backend.app.main:app --host 127.0.0.1 --port 8000"

echo [2/2] Starting Vite Frontend on http://127.0.0.1:5173 ...
start "Signova AI - Frontend" cmd /k "cd frontend && npx vite --host 127.0.0.1 --port 5173"

echo.
echo Signova AI is running!
echo   - App:      http://127.0.0.1:5173
echo   - API Docs: http://127.0.0.1:8000/docs
echo ========================================================
