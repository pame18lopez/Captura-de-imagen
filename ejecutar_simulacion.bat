@echo off
cd /d "%~dp0"
echo ========================================================
echo   INICIANDO MODO SIMULACION (SIN CAMARA FISICA)
echo ========================================================
".venv\Scripts\python.exe" main.py --simulate
if errorlevel 1 (
    echo.
    echo Ocurrio un error al ejecutar. Presiona cualquier tecla para cerrar.
    pause >nul
)
