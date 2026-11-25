"""
Programa Principal de Visión por Computadora e Inteligencia Artificial
para el juego de Gato y Ratón en tiempo real.

Características:
- Captura de video en tiempo real desde cámara web conectada por USB.
- Búsqueda automática de cámaras USB y conmutación en caliente (tecla V).
- Detección y calibración del tablero de juego (homografía y perspectiva plana).
- Detección y clasificación de fichas físicas (Ratón vs Gato) en espacio HSV.
- Delay de 2 segundos para confirmar la jugada del usuario tras detectar la ficha (evita errores con manos).
- Inteligencia Artificial con algoritmo Minimax para decidir la jugada óptima.
- Asistencia visual aumentada (AR): la IA señala la casilla donde debe colocarse la pieza.
- Detección de turnos, confirmación automática por estabilidad y conteo de victorias.
- Modo simulación (--simulate) para pruebas sin cámara física.
"""

import argparse
import sys
import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from board_detector import BoardDetector
from camera_manager import list_available_cameras, open_usb_camera
from game_ai import (
    EMPTY, GATO, RATON,
    check_winner, create_empty_board, get_available_moves, get_best_move
)
from hud_renderer import (
    draw_ai_target_cell, draw_confirmation_countdown, draw_hud,
    draw_text_with_shadow, draw_virtual_board_view
)
from piece_detector import PieceDetector


class GatoRatonApp:
    def __init__(self, camera_idx: Optional[int] = None, simulate: bool = False, difficulty: str = "hard"):
        self.camera_idx = camera_idx
        self.simulate = simulate
        self.difficulty = difficulty

        # Módulos del sistema
        self.board_detector = BoardDetector(warped_size=450)
        self.piece_detector = PieceDetector()

        # Estado del juego
        self.logical_board = create_empty_board()
        self.current_turn = RATON       # El Ratón (jugador) inicia por defecto
        self.ai_target_move: Optional[Tuple[int, int]] = None
        self.winner: Optional[str] = None
        self.winning_line: Optional[List[Tuple[int, int]]] = None
        self.scores = {RATON: 0, GATO: 0, "DRAW": 0}

        # Sistema de retardo (delay de 2 segundos) para confirmación de jugadas
        self.confirmation_delay = 2.0  # 2 segundos exactos
        self.pending_candidate_move: Optional[Tuple[int, int]] = None
        self.candidate_first_detected_time: Optional[float] = None
        self.pending_confirmation_info: Optional[Dict] = None

        # Interacción y calibración
        self.calibrating_corners = False
        self.manual_corners: List[List[int]] = []
        self.sample_mode: Optional[str] = None  # RATON, GATO o None
        self.last_mouse_pos = (0, 0)
        self.simulated_frame: Optional[np.ndarray] = None

        # Variables de tiempo y animación
        self.start_time = time.time()
        self.last_ai_calc_time = 0.0

        # Ventanas
        self.main_window_name = "Gato y Raton - Vision por Computadora e IA"
        self.warped_window_name = "Tablero Vista Superior (Warped)"

    def init_camera(self) -> Optional[cv2.VideoCapture]:
        """Inicializa la cámara web USB o detecta la mejor opción."""
        if self.simulate:
            print("[Info] Iniciando en MODO SIMULACION (sin camara fisica).")
            return None

        cap, chosen_idx = open_usb_camera(target_index=self.camera_idx)
        self.camera_idx = chosen_idx

        if cap is None:
            print("[Aviso] No fue posible conectar con una cámara física.")
            print("[Aviso] Iniciando MODO SIMULACION automáticamente.")
            self.simulate = True
            return None

        return cap

    def reset_game(self, starting_player: str = RATON):
        """Reinicia la partida actual manteniendo puntuaciones."""
        self.logical_board = create_empty_board()
        self.current_turn = starting_player
        self.ai_target_move = None
        self.winner = None
        self.winning_line = None
        self.pending_candidate_move = None
        self.candidate_first_detected_time = None
        self.pending_confirmation_info = None
        self.piece_detector.reset_history()
        print(f"[Juego] Nueva partida iniciada. Turno inicial: {'RATON (Jugador)' if starting_player == RATON else 'GATO (IA)'}")

    def on_mouse(self, event, x, y, flags, param):
        """Manejador de clics del mouse en la ventana principal."""
        self.last_mouse_pos = (x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            # 1. Calibración manual de 4 esquinas del tablero
            if self.calibrating_corners:
                self.manual_corners.append([x, y])
                print(f"[Calibracion] Esquina {len(self.manual_corners)} seleccionada: ({x}, {y})")
                if len(self.manual_corners) == 4:
                    self.board_detector.set_corners(np.array(self.manual_corners, dtype="float32"))
                    self.board_detector.save_config()
                    self.calibrating_corners = False
                    self.manual_corners.clear()
                    print("[Calibracion] ¡Tablero calibrado y guardado exitosamente!")
                return

            # 2. Muestreo de color HSV para fichas
            if self.sample_mode is not None:
                frame_hsv = cv2.cvtColor(param, cv2.COLOR_BGR2HSV)
                self.piece_detector.calibrate_color_at_point(frame_hsv, x, y, self.sample_mode)
                print(f"[Muestreo] Color calibrado para {self.sample_mode}")
                self.sample_mode = None
                return

            # 3. En modo simulación: hacer clic para colocar ficha
            if self.simulate and self.winner is None:
                self.handle_simulation_click(x, y)

    def handle_simulation_click(self, x: int, y: int):
        """Maneja clics sobre el tablero simulado para colocar fichas manualmente."""
        if self.board_detector.corners is None:
            return

        if self.board_detector.M is not None:
            pt = np.array([[[x, y]]], dtype="float32")
            warped_pt = cv2.perspectiveTransform(pt, self.board_detector.M)[0][0]
            wx, wy = warped_pt[0], warped_pt[1]

            if 0 <= wx < 450 and 0 <= wy < 450:
                col = int(wx // 150)
                row = int(wy // 150)

                if self.logical_board[row][col] == EMPTY:
                    if self.current_turn == RATON:
                        self.logical_board[row][col] = RATON
                        print(f"[Simulacion] Jugador colocó RATON en casilla [{row}, {col}]")
                        self.check_game_status()
                        if self.winner is None:
                            self.current_turn = GATO
                    elif self.current_turn == GATO:
                        self.logical_board[row][col] = GATO
                        print(f"[Simulacion] Ficha GATO colocada en casilla [{row}, {col}]")
                        self.check_game_status()
                        if self.winner is None:
                            self.current_turn = RATON
                            self.ai_target_move = None

    def generate_synthetic_frame(self, width: int = 960, height: int = 540) -> np.ndarray:
        """Crea un frame simulado con un tablero físico dibujado sobre una mesa."""
        frame = np.ones((height, width, 3), dtype=np.uint8) * 45

        bx, by, bsize = (width - 400) // 2, (height - 400) // 2, 400
        corners = np.array([
            [bx, by],
            [bx + bsize, by],
            [bx + bsize, by + bsize],
            [bx, by + bsize]
        ], dtype="float32")

        if self.board_detector.corners is None:
            self.board_detector.set_corners(corners)

        cv2.rectangle(frame, (bx, by), (bx + bsize, by + bsize), (240, 240, 240), -1)
        cv2.rectangle(frame, (bx, by), (bx + bsize, by + bsize), (30, 30, 30), 4)

        csize = bsize // 3
        for i in range(1, 3):
            cv2.line(frame, (bx + i * csize, by), (bx + i * csize, by + bsize), (50, 50, 50), 3)
            cv2.line(frame, (bx, by + i * csize), (bx + bsize, by + i * csize), (50, 50, 50), 3)

        for r in range(3):
            for c in range(3):
                piece = self.logical_board[r][c]
                cx = bx + c * csize + csize // 2
                cy = by + r * csize + csize // 2
                rad = int(csize * 0.35)

                if piece == RATON:
                    cv2.circle(frame, (cx, cy), rad, (220, 130, 30), -1)
                    cv2.circle(frame, (cx, cy), rad, (255, 255, 255), 2)
                elif piece == GATO:
                    cv2.circle(frame, (cx, cy), rad, (30, 30, 220), -1)
                    cv2.circle(frame, (cx, cy), rad, (255, 255, 255), 2)

        return frame

    def check_game_status(self):
        """Evalúa si la partida ha terminado."""
        win, line = check_winner(self.logical_board)
        if win:
            self.winner = win
            self.winning_line = line
            if win in self.scores:
                self.scores[win] += 1
            print(f"[Juego] Fin de la partida. Ganador: {self.winner}")

    def update_game_logic_from_vision(self, stable_board: List[List[str]]):
        """
        Sincroniza el estado del juego con la detección física de la cámara.
        Aplica un delay de 2 segundos para confirmar la jugada del usuario tras detectar la ficha.
        """
        if self.winner is not None:
            self.pending_confirmation_info = None
            return

        now = time.time()

        # 1. TURNO DEL RATON (Jugador humano)
        if self.current_turn == RATON:
            # Buscar casillas vacías donde se observe una nueva ficha de RATON
            new_moves = []
            for r in range(3):
                for c in range(3):
                    if self.logical_board[r][c] == EMPTY and stable_board[r][c] == RATON:
                        new_moves.append((r, c))

            if len(new_moves) == 1:
                candidate = new_moves[0]
                # Si es una casilla nueva recién detectada, iniciar temporizador de 2 segundos
                if self.pending_candidate_move != candidate:
                    self.pending_candidate_move = candidate
                    self.candidate_first_detected_time = now
                    print(f"[Vision] Ficha de RATON detectada en casilla {candidate}. Esperando 2 segundos para confirmar...")

                # Calcular progreso del delay de 2 segundos
                elapsed = now - self.candidate_first_detected_time
                remaining = max(0.0, self.confirmation_delay - elapsed)
                progress = min(1.0, elapsed / self.confirmation_delay)

                self.pending_confirmation_info = {
                    "cell": candidate,
                    "piece": RATON,
                    "remaining": remaining,
                    "progress": progress
                }

                # Si ya pasaron los 2 segundos completos con la ficha estable, confirmar la jugada
                if elapsed >= self.confirmation_delay:
                    nr, nc = candidate
                    self.logical_board[nr][nc] = RATON
                    print(f"[Vision] ¡Jugada confirmada tras 2s! RATON colocado en casilla [{nr}, {nc}]")
                    self.pending_candidate_move = None
                    self.candidate_first_detected_time = None
                    self.pending_confirmation_info = None
                    self.check_game_status()

                    if self.winner is None:
                        self.current_turn = GATO
                        self.ai_target_move = None

            else:
                # Si la ficha se retiró antes de los 2 segundos o se detectaron múltiples a la vez
                if self.pending_candidate_move is not None:
                    print("[Vision] Detección interrumpida antes de los 2 segundos. Temporizador reiniciado.")
                self.pending_candidate_move = None
                self.candidate_first_detected_time = None
                self.pending_confirmation_info = None

        # 2. TURNO DEL GATO (Inteligencia Artificial)
        elif self.current_turn == GATO:
            if self.ai_target_move is None:
                best = get_best_move(self.logical_board, ai_piece=GATO, human_piece=RATON, difficulty=self.difficulty)
                if best is not None:
                    self.ai_target_move = best
                    print(f"[IA] Decisión tomada: Colocar ficha GATO en casilla {self.ai_target_move}")
                else:
                    self.check_game_status()

            if self.ai_target_move is not None:
                tr, tc = self.ai_target_move
                if stable_board[tr][tc] == GATO:
                    # Delay suave para la colocación de la ficha de la IA (1.5 segundos)
                    if self.pending_candidate_move != (tr, tc):
                        self.pending_candidate_move = (tr, tc)
                        self.candidate_first_detected_time = now

                    elapsed = now - self.candidate_first_detected_time
                    remaining = max(0.0, 1.5 - elapsed)
                    progress = min(1.0, elapsed / 1.5)

                    self.pending_confirmation_info = {
                        "cell": (tr, tc),
                        "piece": GATO,
                        "remaining": remaining,
                        "progress": progress
                    }

                    if elapsed >= 1.5:
                        self.logical_board[tr][tc] = GATO
                        print(f"[Vision] Ficha de la IA confirmada correctamente en [{tr}, {tc}]")
                        self.pending_candidate_move = None
                        self.candidate_first_detected_time = None
                        self.pending_confirmation_info = None
                        self.check_game_status()

                        if self.winner is None:
                            self.current_turn = RATON
                            self.ai_target_move = None
                else:
                    self.pending_candidate_move = None
                    self.candidate_first_detected_time = None
                    self.pending_confirmation_info = None

    def run(self):
        """Bucle principal de la aplicación."""
        cap = self.init_camera()
        cv2.namedWindow(self.main_window_name, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.warped_window_name, cv2.WINDOW_NORMAL)

        print("\n" + "=" * 65)
        print("  INICIANDO SISTEMA DE VISION E INTELIGENCIA ARTIFICIAL")
        print("  JUEGO DE GATO Y RATON (CAMARA USB / EN TIEMPO REAL)")
        print("=" * 65)
        print("  Atajos de teclado:")
        print("  [D] o [M] : Cambiar Modalidad / Dificultad (Facil -> Medio -> Dificil)")
        print("  [V] : Cambiar a la siguiente cámara (USB / Integrada)")
        print("  [C] : Calibrar esquinas del tablero (4 clics con el mouse)")
        print("  [A] : Detección automática del tablero por contornos")
        print("  [1] : Muestrear color de ficha RATON con clic de mouse")
        print("  [2] : Muestrear color de ficha GATO con clic de mouse")
        print("  [S] : Alternar quién inicia (Jugador / IA)")
        print("  [SPACE] : Confirmar jugada de IA directamente (modo virtual)")
        print("  [R] : Reiniciar tablero y comenzar nueva partida")
        print("  [Q] o [ESC] : Salir")
        print("=" * 65 + "\n")

        pulse_phase = 0.0

        while True:
            # 1. Obtención de frame
            if self.simulate or cap is None:
                raw_frame = self.generate_synthetic_frame()
            else:
                ret, raw_frame = cap.read()
                if not ret:
                    print("[Aviso] Frame no disponible o cámara desconectada. Activando vista simulada...")
                    raw_frame = self.generate_synthetic_frame()

            display_frame = raw_frame.copy()
            h, w = raw_frame.shape[:2]

            if self.board_detector.corners is None:
                self.board_detector.set_default_centered_roi(w, h, box_size=min(w, h) - 150)

            # 2. Procesamiento de perspectiva y detección de fichas
            warped = self.board_detector.get_warped_board(raw_frame)
            if warped is not None:
                if not self.simulate:
                    _, stable_board = self.piece_detector.update_and_detect_board(warped)
                    self.update_game_logic_from_vision(stable_board)
                else:
                    stable_board = self.logical_board
                    if self.current_turn == GATO and self.ai_target_move is None and self.winner is None:
                        self.ai_target_move = get_best_move(self.logical_board, ai_piece=GATO, human_piece=RATON, difficulty=self.difficulty)

            # 3. Dibujar elementos de Realidad Aumentada (AR) sobre la cámara
            self.board_detector.draw_board_overlay(display_frame, color=(0, 255, 100), thickness=2)

            pulse_phase += 0.12
            pulse = (np.sin(pulse_phase) + 1.0) / 2.0

            # Si la IA tiene una casilla objetivo, resaltarla con animación de pulsación
            if self.current_turn == GATO and self.ai_target_move is not None:
                tr, tc = self.ai_target_move
                cell_quad = self.board_detector.get_cell_quad_in_frame(tr, tc)
                draw_ai_target_cell(display_frame, cell_quad, pulse=pulse)

            # Si hay una casilla en proceso de confirmación de 2 segundos, dibujar cuenta regresiva
            if self.pending_confirmation_info is not None:
                cr, cc = self.pending_confirmation_info["cell"]
                cell_quad = self.board_detector.get_cell_quad_in_frame(cr, cc)
                draw_confirmation_countdown(
                    display_frame,
                    cell_quad,
                    progress=self.pending_confirmation_info["progress"],
                    remaining=self.pending_confirmation_info["remaining"]
                )

            if self.calibrating_corners:
                for idx, pt in enumerate(self.manual_corners):
                    cv2.circle(display_frame, tuple(pt), 7, (0, 0, 255), -1)
                    cv2.putText(display_frame, str(idx + 1), (pt[0] + 10, pt[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # 4. Renderizar HUD de control con indicador de progreso y delay
            draw_hud(
                display_frame,
                game_state="ACTIVO",
                current_turn=self.current_turn,
                ai_move=self.ai_target_move,
                winner=self.winner,
                scores=self.scores,
                stable_board=stable_board,
                calibrating_corners=self.calibrating_corners,
                manual_corners_count=len(self.manual_corners),
                difficulty=self.difficulty,
                pending_confirmation=self.pending_confirmation_info
            )

            # 5. Vista de tablero rectificado (Warped)
            if warped is not None:
                warped_view = draw_virtual_board_view(
                    warped,
                    board_state=self.logical_board,
                    ai_move=self.ai_target_move,
                    winning_line=self.winning_line,
                    pending_confirmation=self.pending_confirmation_info
                )
                cv2.imshow(self.warped_window_name, warped_view)

            cv2.setMouseCallback(self.main_window_name, self.on_mouse, raw_frame)
            cv2.imshow(self.main_window_name, display_frame)

            # 6. Manejo de teclas
            key = cv2.waitKey(20) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                break
            elif key in [ord('v'), ord('V')]:
                current_idx = self.camera_idx if self.camera_idx is not None else 0
                next_idx = (current_idx + 1) % 4
                print(f"\n[Cámara] Conmutando a cámara en índice [{next_idx}]...")
                if cap is not None:
                    cap.release()
                new_cap, actual_idx = open_usb_camera(target_index=next_idx)
                if new_cap is not None:
                    cap = new_cap
                    self.camera_idx = actual_idx
                    self.simulate = False
                    print(f"[Cámara] Conectado exitosamente a la cámara [{actual_idx}]")
                else:
                    print(f"[Aviso] Cámara {next_idx} no disponible. Intentando recuperar la anterior...")
                    cap, self.camera_idx = open_usb_camera(target_index=current_idx)
            elif key in [ord('c'), ord('C')]:
                self.calibrating_corners = True
                self.manual_corners.clear()
                print("\n[Calibracion] Haz clic en las 4 esquinas del tablero físico en orden:")
                print("              1. Superior Izquierda | 2. Superior Derecha | 3. Inferior Derecha | 4. Inferior Izquierda")
            elif key in [ord('a'), ord('A')]:
                print("[Calibracion] Buscando contornos de tablero automáticamente...")
                found = self.board_detector.auto_detect_board(raw_frame)
                if found:
                    self.board_detector.save_config()
                    print("[Calibracion] ¡Tablero detectado y calibrado con éxito!")
                else:
                    print("[Calibracion] No se detectó un cuadrilátero claro. Utiliza [C] para calibrar manualmente.")
            elif key == ord('1'):
                self.sample_mode = RATON
                print("[Muestreo] Haz clic sobre una ficha de RATON en la imagen para capturar su color.")
            elif key == ord('2'):
                self.sample_mode = GATO
                print("[Muestreo] Haz clic sobre una ficha de GATO en la imagen para capturar su color.")
            elif key in [ord('s'), ord('S')]:
                next_start = GATO if self.current_turn == RATON else RATON
                self.reset_game(starting_player=next_start)
            elif key == ord(' '):
                if self.current_turn == GATO and self.ai_target_move is not None:
                    tr, tc = self.ai_target_move
                    self.logical_board[tr][tc] = GATO
                    print(f"[Virtual] Jugada de IA confirmada en [{tr}, {tc}]")
                    self.check_game_status()
                    if self.winner is None:
                        self.current_turn = RATON
                        self.ai_target_move = None
            elif key in [ord('d'), ord('D'), ord('m'), ord('M')]:
                # Ciclar entre las dificultades / modalidades
                levels = ["easy", "medium", "hard"]
                curr_idx = levels.index(self.difficulty) if self.difficulty in levels else 0
                next_level = levels[(curr_idx + 1) % len(levels)]
                self.difficulty = next_level
                nombres = {"easy": "FACIL", "medium": "MEDIO", "hard": "DIFICIL (IMBATIBLE)"}
                print(f"\n[Modalidad] Dificultad de la IA cambiada a: {nombres[self.difficulty]}")
            elif key in [ord('r'), ord('R')]:
                self.reset_game()

        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print("\n[Fin] Programa finalizado correctamente.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Juego de Gato y Raton con Vision e IA")
    parser.add_argument("--camera", type=int, default=None,
                        help="Índice de la cámara USB (si no se especifica, se detecta automáticamente)")
    parser.add_argument("--simulate", action="store_true", help="Ejecutar en modo simulación interactivo")
    parser.add_argument("--difficulty", type=str, default="hard", choices=["easy", "medium", "hard"],
                        help="Dificultad de la IA (easy, medium, hard)")

    args = parser.parse_args()
    app = GatoRatonApp(camera_idx=args.camera, simulate=args.simulate, difficulty=args.difficulty)
    app.run()
