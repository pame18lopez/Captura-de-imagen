"""
Módulo de Renderizado Visual y Gráficos (HUD) para Gato y Ratón.
Dibuja paneles interactivos, información de estado, casillas objetivo de la IA,
líneas de victoria, temporizador de confirmación de jugada y vista de tablero rectificado.
"""

from typing import List, Optional, Tuple, Dict
import cv2
import numpy as np
from game_ai import EMPTY, GATO, RATON


# Paleta de colores BGR
COLOR_BG_PANEL = (28, 28, 30)
COLOR_TEXT = (245, 245, 245)
COLOR_TEXT_DIM = (160, 160, 160)
COLOR_ACCENT = (255, 178, 50)       # Ámbar / Dorado
COLOR_RATON = (235, 145, 30)        # Azul brillante (BGR: B=235, G=145, R=30)
COLOR_GATO = (50, 60, 240)          # Rojo brillante (BGR: B=50, G=60, R=240)
COLOR_SUCCESS = (60, 220, 80)       # Verde
COLOR_WARNING = (30, 200, 255)      # Amarillo / Naranja
COLOR_GRID = (180, 180, 180)


def draw_text_with_shadow(img: np.ndarray, text: str, pos: Tuple[int, int],
                          font_scale: float = 0.6, color: Tuple[int, int, int] = COLOR_TEXT,
                          thickness: int = 1, shadow_offset: int = 1):
    """Dibuja texto con sombra para garantizar legibilidad en cualquier fondo."""
    x, y = pos
    cv2.putText(img, text, (x + shadow_offset, y + shadow_offset),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (10, 10, 10), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def draw_ai_target_cell(frame: np.ndarray, cell_quad: np.ndarray, pulse: float = 1.0):
    """
    Dibuja un resalte visual animado sobre la casilla donde la IA recomienda colocar la pieza.
    """
    overlay = frame.copy()
    cv2.fillPoly(overlay, [cell_quad], (0, 215, 255))
    alpha = 0.35 + 0.15 * pulse
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    cv2.polylines(frame, [cell_quad], isClosed=True, color=(0, 240, 255), thickness=3, lineType=cv2.LINE_AA)

    cx = int(np.mean(cell_quad[:, 0, 0]))
    cy = int(np.mean(cell_quad[:, 0, 1]))
    cv2.circle(frame, (cx, cy), int(10 + 4 * pulse), (0, 230, 255), -1)
    draw_text_with_shadow(frame, "IA AQUI", (cx - 30, cy + 5), font_scale=0.55, color=(10, 10, 10), thickness=2)


def draw_confirmation_countdown(frame: np.ndarray, cell_quad: np.ndarray, progress: float, remaining: float):
    """
    Dibuja un indicador visual de cuenta regresiva y progreso sobre la casilla detectada
    durante el delay de 2 segundos antes de registrar la jugada.
    """
    # Relleno verde semitransparente que aumenta con el progreso
    overlay = frame.copy()
    cv2.fillPoly(overlay, [cell_quad], (0, 230, 120))
    alpha = 0.20 + 0.30 * progress
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Borde de confirmación
    cv2.polylines(frame, [cell_quad], isClosed=True, color=(0, 255, 150), thickness=3, lineType=cv2.LINE_AA)

    # Texto con cuenta regresiva en el centro de la casilla
    cx = int(np.mean(cell_quad[:, 0, 0]))
    cy = int(np.mean(cell_quad[:, 0, 1]))

    # Círculo de progreso
    cv2.circle(frame, (cx, cy), 22, (20, 20, 20), -1)
    cv2.circle(frame, (cx, cy), 22, (0, 255, 150), 2, lineType=cv2.LINE_AA)

    txt = f"{remaining:.1f}s"
    draw_text_with_shadow(frame, txt, (cx - 16, cy + 5), font_scale=0.50, color=(255, 255, 255), thickness=2)


def draw_hud(
    frame: np.ndarray,
    game_state: str,
    current_turn: str,
    ai_move: Optional[Tuple[int, int]],
    winner: Optional[str],
    scores: dict,
    stable_board: List[List[str]],
    calibrating_corners: bool,
    manual_corners_count: int,
    difficulty: str,
    pending_confirmation: Optional[Dict] = None
):
    """Dibuja la barra de estado superior e información de control."""
    h, w = frame.shape[:2]

    # Panel superior de estado
    top_bar_height = 82
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, top_bar_height), COLOR_BG_PANEL, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.line(frame, (0, top_bar_height), (w, top_bar_height), (70, 70, 75), 2)

    # Título y marcador
    draw_text_with_shadow(frame, "GATO Y RATON - VISION IA", (15, 26), font_scale=0.72, color=COLOR_ACCENT, thickness=2)
    draw_text_with_shadow(frame, f"Dificultad: {difficulty.upper()} | Puntos -> Ratón: {scores['R']} | Gato: {scores['G']} | Empates: {scores['DRAW']}",
                          (15, 48), font_scale=0.46, color=COLOR_TEXT_DIM, thickness=1)

    # Mensaje de estado central
    if calibrating_corners:
        status_msg = f"MODO CALIBRACION: Haz clic en las 4 esquinas del tablero ({manual_corners_count}/4)"
        status_color = COLOR_WARNING
    elif winner:
        if winner == RATON:
            status_msg = "¡FELICIDADES! ¡GANA EL RATON (JUGADOR)!"
            status_color = COLOR_SUCCESS
        elif winner == GATO:
            status_msg = "¡PARTIDA TERMINADA! ¡GANA EL GATO (IA)!"
            status_color = COLOR_GATO
        else:
            status_msg = "¡EMPATE! NO HAY MAS MOVIMIENTOS"
            status_color = COLOR_WARNING
    elif pending_confirmation is not None:
        # Se detectó una ficha y estamos en el delay de 2 segundos
        r, c = pending_confirmation["cell"]
        rem = pending_confirmation["remaining"]
        prog = pending_confirmation["progress"]
        player_name = "RATON" if pending_confirmation["piece"] == RATON else "GATO"
        status_msg = f"¡{player_name} DETECTADO EN [{r}, {c}]! Confirmando en {rem:.1f}s... (Retira tu mano)"
        status_color = (0, 230, 255)  # Amarillo/Dorado brillante

        # Barra de progreso visual en el panel superior
        bar_x1 = 15
        bar_x2 = w - 15
        bar_y = top_bar_height - 6
        cv2.line(frame, (bar_x1, bar_y), (bar_x2, bar_y), (50, 50, 50), 3)
        fill_x = int(bar_x1 + (bar_x2 - bar_x1) * prog)
        cv2.line(frame, (bar_x1, bar_y), (fill_x, bar_y), (0, 255, 150), 3)

    elif current_turn == RATON:
        status_msg = "TURNO DEL JUGADOR: Coloca tu ficha de RATON en el tablero"
        status_color = COLOR_RATON
    elif current_turn == GATO:
        if ai_move:
            r, c = ai_move
            status_msg = f"TURNO DE LA IA: Coloca ficha GATO en casilla [{r}, {c}]"
        else:
            status_msg = "TURNO DE LA IA: Calculando mejor jugada..."
        status_color = COLOR_WARNING
    else:
        status_msg = f"Estado: {game_state}"
        status_color = COLOR_TEXT

    draw_text_with_shadow(frame, status_msg, (15, 68), font_scale=0.52, color=status_color, thickness=2)

    # Panel lateral / inferior con atajos de teclado
    guide_y = h - 24
    guide_text = "[D] Dificultad | [V] Camara | [C] Calibrar | [1] Raton | [2] Gato | [S] Turno | [R] Reiniciar | [Q] Salir"
    cv2.rectangle(frame, (0, h - 35), (w, h), COLOR_BG_PANEL, -1)
    draw_text_with_shadow(frame, guide_text, (15, guide_y), font_scale=0.40, color=COLOR_TEXT_DIM, thickness=1)


def draw_virtual_board_view(
    warped_board: np.ndarray,
    board_state: List[List[str]],
    ai_move: Optional[Tuple[int, int]] = None,
    winning_line: Optional[List[Tuple[int, int]]] = None,
    pending_confirmation: Optional[Dict] = None
) -> np.ndarray:
    """
    Crea una vista gráfica enriquecida del tablero 450x450 mostrando casillas,
    fichas detectadas, resalte de jugada sugerida, confirmación y línea ganadora.
    """
    view = warped_board.copy()
    size = view.shape[0]
    cell_size = size // 3

    # Dibujar líneas del tablero
    for i in range(1, 3):
        pos = i * cell_size
        cv2.line(view, (pos, 0), (pos, size), (40, 40, 40), 4)
        cv2.line(view, (0, pos), (size, pos), (40, 40, 40), 4)
        cv2.line(view, (pos, 0), (pos, size), (220, 220, 220), 2)
        cv2.line(view, (0, pos), (size, pos), (220, 220, 220), 2)

    # Dibujar casillas y piezas
    for r in range(3):
        for c in range(3):
            cx = c * cell_size + cell_size // 2
            cy = r * cell_size + cell_size // 2
            radius = int(cell_size * 0.32)

            piece = board_state[r][c]

            # Si es la casilla elegida por la IA
            if ai_move == (r, c) and piece == EMPTY:
                cv2.rectangle(view, (c * cell_size + 8, r * cell_size + 8),
                              ((c + 1) * cell_size - 8, (r + 1) * cell_size - 8),
                              (0, 230, 255), 3)
                draw_text_with_shadow(view, "OBJETIVO IA", (cx - 45, cy + 5), font_scale=0.45, color=(0, 230, 255), thickness=1)

            # Si es la casilla en proceso de confirmación de 2 segundos
            if pending_confirmation and pending_confirmation["cell"] == (r, c) and piece == EMPTY:
                cv2.rectangle(view, (c * cell_size + 6, r * cell_size + 6),
                              ((c + 1) * cell_size - 6, (r + 1) * cell_size - 6),
                              (0, 255, 150), 3)
                rem = pending_confirmation["remaining"]
                draw_text_with_shadow(view, f"OK {rem:.1f}s", (cx - 30, cy + 5), font_scale=0.55, color=(0, 255, 150), thickness=2)

            if piece == RATON:
                # Círculo azul para Ratón
                cv2.circle(view, (cx, cy), radius, COLOR_RATON, -1, lineType=cv2.LINE_AA)
                cv2.circle(view, (cx, cy), radius, (255, 255, 255), 2, lineType=cv2.LINE_AA)
                cv2.circle(view, (cx, cy), radius - 6, (200, 100, 0), 2, lineType=cv2.LINE_AA)
                draw_text_with_shadow(view, "RATON", (cx - 26, cy + 6), font_scale=0.52, color=(255, 255, 255), thickness=2)

            elif piece == GATO:
                # Círculo rojo para Gato
                cv2.circle(view, (cx, cy), radius, COLOR_GATO, -1, lineType=cv2.LINE_AA)
                cv2.circle(view, (cx, cy), radius, (255, 255, 255), 2, lineType=cv2.LINE_AA)
                cv2.circle(view, (cx, cy), radius - 6, (0, 0, 180), 2, lineType=cv2.LINE_AA)
                draw_text_with_shadow(view, "GATO", (cx - 22, cy + 6), font_scale=0.52, color=(255, 255, 255), thickness=2)

    # Dibujar línea ganadora
    if winning_line:
        (r1, c1), (r2, c2), (r3, c3) = winning_line
        pt1 = (c1 * cell_size + cell_size // 2, r1 * cell_size + cell_size // 2)
        pt3 = (c3 * cell_size + cell_size // 2, r3 * cell_size + cell_size // 2)
        cv2.line(view, pt1, pt3, (0, 255, 120), 8, lineType=cv2.LINE_AA)
        cv2.line(view, pt1, pt3, (255, 255, 255), 3, lineType=cv2.LINE_AA)

    return view
