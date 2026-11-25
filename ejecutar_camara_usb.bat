@echo off
cd /d "%~dp0"
echo ========================================================
echo   INICIANDO JUEGO DE GATO Y RATON CON CAMARA USB
echo ========================================================
echo Buscando y conectando con la camara web USB...
echo (Consejo: dentro del juego puedes presionar 'V' para alternar entre camaras)
echo.
".venv\Scripts\python.exe" main.py
if errorlevel 1 (
    echo.
    echo Ocurrio un error al ejecutar. Presiona cualquier tecla para cerrar.
    pause >nul
)
