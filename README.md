# Sistema de Visión por Computadora e Inteligencia Artificial: Gato y Ratón

Este proyecto implementa un sistema interactivo en tiempo real en Python que combina **visión por computadora (OpenCV)** e **inteligencia artificial (Minimax con Poda Alfa-Beta)** para jugar al **Gato y Ratón (Tic-Tac-Toe / 3 en Raya)** contra un humano mediante una cámara web.

---

## 🌟 Características Principales

1. **Captura de Imagen en Tiempo Real**:
   - Conexión directa a la cámara web (`cv2.VideoCapture`).
   - Si no se detecta cámara física o se desea probar en laboratorio, incluye un **Modo Simulación** (`--simulate`) completamente interactivo.

2. **Detección y Calibración del Tablero**:
   - **Corrección de Perspectiva (Homografía)**: Elimina la distorsión del ángulo de la cámara proyectando el tablero a una vista superior plana de 450x450 píxeles.
   - **Calibración Manual por Clics (Tecla `C`)**: Permite marcar las 4 esquinas del tablero físico en la mesa.
   - **Detección Automática por Contornos (Tecla `A`)**: Localiza el contorno cuadrangular de mayor contraste.
   - Guarda y recuerda la calibración en `board_config.json`.

3. **Detección y Clasificación de Fichas**:
   - Clasificación en espacio de color **HSV** entre fichas de **Ratón** (Jugador humano, azul por defecto) y **Gato** (IA, rojo por defecto).
   - **Calibrador Interactivo por Clic (Teclas `1` y `2`)**: Puedes usar cualquier ficha u objeto que tengas (taparroscas, fichas impresas en 3D, monedas, etc.). Presiona `1` o `2` y haz clic sobre la ficha para calibrar su color instantáneamente.
   - **Filtro Temporal de Estabilidad (Debounce)**: Exige que una ficha se detecte durante varios fotogramas consecutivos antes de registrarla, evitando falsos positivos causados por el movimiento de la mano al colocar la pieza.

4. **Inteligencia Artificial y Toma de Decisiones**:
   - Algoritmo **Minimax con Poda Alfa-Beta**: evalúa el tablero y calcula matemáticamente la mejor jugada (bloqueos, ataques y victorias).
   - **Asistencia Visual Aumentada (AR)**: La IA proyecta un resalte animado y luminoso sobre la casilla exacta donde debe colocarse la pieza del Gato.
   - Una vez que la cámara detecta que la ficha fue colocada físicamente en dicha casilla, la IA valida la jugada y le cede el turno al usuario.
   - Detección automática de tres en raya, trazado de la línea de victoria y marcador de puntajes.

---

## 📁 Estructura del Proyecto

- `main.py`: Bucle principal de la aplicación, captura de video, máquina de estados y control de eventos.
- `board_detector.py`: Detección de contornos, cálculo de homografía, transformación de perspectiva y mapeo de casillas 3x3.
- `piece_detector.py`: Segmentación HSV, filtros morfológicos, calibración de color y filtro debounce.
- `game_ai.py`: Motor de reglas de juego, algoritmo Minimax con poda alfa-beta y evaluación de partidas.
- `hud_renderer.py`: Renderizado gráfico en pantalla (HUD, indicadores AR, vista rectificada y atajos).
- `test_system.py`: Suite de pruebas unitarias automatizadas.
- `requirements.txt`: Dependencias del proyecto (`opencv-python`, `numpy`, `pillow`).

---

## 🚀 Guía de Instalación y Uso

### 1. Activar el Entorno Virtual

En PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

*(O ejecutando directamente con el ejecutable del entorno: `.\.venv\Scripts\python main.py`)*

### 2. Ejecutar el Programa

#### A) Modo Normal (con Cámara Web)
```powershell
.\.venv\Scripts\python main.py
```
Si cuentas con múltiples cámaras (ej. cámara integrada vs cámara USB externa):
```powershell
.\.venv\Scripts\python main.py --camera 1
```

#### B) Modo Simulación (sin necesidad de cámara física)
Permite probar la visión, la IA y las reglas haciendo clic sobre el tablero virtual:
```powershell
.\.venv\Scripts\python main.py --simulate
```

#### C) Ajustar Dificultad de la IA
Opciones: `easy`, `medium`, `hard` (por defecto `hard`):
```powershell
.\.venv\Scripts\python main.py --difficulty medium
```

---

## 🎮 Controles y Atajos de Teclado

| Tecla | Acción |
| :---: | :--- |
| **`C`** | **Calibrar Tablero Manualmente**: Haz clic en las 4 esquinas del tablero en el video (Superior Izq -> Superior Der -> Inferior Der -> Inferior Izq). |
| **`A`** | **Detección Automática**: Intenta encontrar automáticamente el cuadrilátero del tablero por visión. |
| **`1`** | **Muestrear Ficha de Ratón**: Activa el puntero; luego haz clic sobre una ficha de Ratón en la imagen para capturar su color HSV. |
| **`2`** | **Muestrear Ficha de Gato**: Activa el puntero; luego haz clic sobre una ficha de Gato en la imagen para capturar su color HSV. |
| **`S`** | **Alternar Turno Inicial**: Cambia quién tira primero (Ratón / Jugador o Gato / IA). |
| **`ESPACIO`** | **Confirmar Jugada Virtual**: Confirma la jugada de la IA de forma manual en pantalla sin requerir colocar la ficha física. |
| **`R`** | **Reiniciar Partida**: Limpia el tablero y comienza una nueva partida conservando el marcador. |
| **`Q` / `ESC`** | **Salir** del programa. |

---

## 🧪 Ejecución de Pruebas Automatizadas

Para verificar el funcionamiento de la IA, la detección de esquinas y el reconocimiento de fichas:
```powershell
.\.venv\Scripts\python test_system.py
```
Resultado esperado: `Ran 9 tests in 0.2s - OK`.
