@echo off
cd /d "%~dp0"
echo ========================================================
echo   INICIANDO JUEGO DE GATO Y RATON (VISION POR COMPUTADORA)
echo ========================================================
".venv\Scripts\python.exe" main.py
if errorlevel 1 (
    echo.
    echo Ocurrio un error al ejecutar. Presiona cualquier tecla para cerrar.
    pause >nul
)
