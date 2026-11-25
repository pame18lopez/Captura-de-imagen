@echo off
cd /d "%~dp0"
echo ========================================================
echo   DIAGNOSTICO DE CAMARAS CONECTADAS POR USB
echo ========================================================
".venv\Scripts\python.exe" -c "from camera_manager import list_available_cameras; cams = list_available_cameras(); print('\nTotal de camaras detectadas:', len(cams)); [print(f' -> Camara #{c[\"index\"]}: {c[\"label\"]} ({c[\"width\"]}x{c[\"height\"]} via {c[\"backend\"]})') for c in cams]"
echo.
echo Presiona cualquier tecla para cerrar esta ventana...
pause >nul
