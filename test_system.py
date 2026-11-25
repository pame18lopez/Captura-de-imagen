"""
Suite de Pruebas Unitarias y de Integración para el Sistema de Visión e IA
de Gato y Ratón.
"""

import unittest
import numpy as np
import cv2

from game_ai import (
    EMPTY, GATO, RATON,
    check_winner, get_available_moves, get_best_move, minimax
)
from board_detector import BoardDetector, order_points
from piece_detector import PieceDetector


class TestGameAI(unittest.TestCase):
    def test_winner_horizontal(self):
        board = [
            [RATON, RATON, RATON],
            [EMPTY, GATO,  EMPTY],
            [GATO,  EMPTY, EMPTY]
        ]
        winner, line = check_winner(board)
        self.assertEqual(winner, RATON)
        self.assertEqual(line, [(0, 0), (0, 1), (0, 2)])

    def test_winner_diagonal(self):
        board = [
            [GATO,  RATON, EMPTY],
            [RATON, GATO,  EMPTY],
            [EMPTY, RATON, GATO]
        ]
        winner, line = check_winner(board)
        self.assertEqual(winner, GATO)
        self.assertEqual(line, [(0, 0), (1, 1), (2, 2)])

    def test_draw(self):
        board = [
            [RATON, GATO,  RATON],
            [RATON, GATO,  GATO],
            [GATO,  RATON, RATON]
        ]
        winner, line = check_winner(board)
        self.assertEqual(winner, "DRAW")
        self.assertIsNone(line)

    def test_ai_blocks_immediate_loss(self):
        """Si el jugador humano (RATON) está a punto de ganar, la IA debe bloquear."""
        board = [
            [RATON, RATON, EMPTY],
            [EMPTY, GATO,  EMPTY],
            [EMPTY, EMPTY, EMPTY]
        ]
        # La casilla (0, 2) da la victoria al humano, la IA debe bloquear en (0, 2)
        best_move = get_best_move(board, ai_piece=GATO, human_piece=RATON, difficulty="hard")
        self.assertEqual(best_move, (0, 2))

    def test_ai_takes_immediate_win(self):
        """Si la IA tiene oportunidad de ganar inmediatamente, debe tomarla."""
        board = [
            [GATO,  GATO,  EMPTY],
            [RATON, RATON, EMPTY],
            [EMPTY, EMPTY, EMPTY]
        ]
        best_move = get_best_move(board, ai_piece=GATO, human_piece=RATON, difficulty="hard")
        self.assertEqual(best_move, (0, 2))


class TestBoardDetector(unittest.TestCase):
    def test_order_points(self):
        # Puntos desordenados
        pts = np.array([[200, 200], [0, 200], [200, 0], [0, 0]], dtype="float32")
        ordered = order_points(pts)
        np.testing.assert_array_equal(ordered[0], [0, 0])      # Top-Left
        np.testing.assert_array_equal(ordered[1], [200, 0])    # Top-Right
        np.testing.assert_array_equal(ordered[2], [200, 200])  # Bottom-Right
        np.testing.assert_array_equal(ordered[3], [0, 200])    # Bottom-Left

    def test_warp_perspective(self):
        detector = BoardDetector(warped_size=450, config_file="test_board_cfg.json")
        frame = np.zeros((600, 800, 3), dtype=np.uint8)
        # Dibujar un cuadrado
        cv2.rectangle(frame, (100, 100), (500, 500), (255, 255, 255), -1)

        detector.set_corners(np.array([
            [100, 100], [500, 100], [500, 500], [100, 500]
        ], dtype="float32"))

        warped = detector.get_warped_board(frame)
        self.assertIsNotNone(warped)
        self.assertEqual(warped.shape, (450, 450, 3))
        # El centro debe ser blanco
        self.assertTrue(np.all(warped[225, 225] == 255))


class TestPieceDetector(unittest.TestCase):
    def test_detect_pieces(self):
        detector = PieceDetector(config_file="test_piece_cfg.json", history_len=4, min_stable_frames=3)

        # Crear celdas de prueba: una vacía, una con ficha azul (Ratón), una con ficha roja (Gato)
        empty_cell = np.ones((100, 100, 3), dtype=np.uint8) * 200  # Gris claro (vacía)

        # Ficha azul (Ratón): en BGR azul brillante es (255, 50, 50)
        blue_cell = empty_cell.copy()
        cv2.circle(blue_cell, (50, 50), 30, (230, 140, 20), -1)

        # Ficha roja (Gato): en BGR rojo puro es (20, 20, 230)
        red_cell = empty_cell.copy()
        cv2.circle(red_cell, (50, 50), 30, (30, 30, 230), -1)

        p_empty, _ = detector.detect_cell_piece(empty_cell)
        p_blue, _ = detector.detect_cell_piece(blue_cell)
        p_red, _ = detector.detect_cell_piece(red_cell)

        self.assertEqual(p_empty, EMPTY)
        self.assertEqual(p_blue, RATON)
        self.assertEqual(p_red, GATO)

    def test_temporal_stability(self):
        detector = PieceDetector(config_file="test_piece_cfg.json", history_len=5, min_stable_frames=4)
        
        # Simular un tablero 450x450 donde la casilla [1, 1] tiene una ficha de Ratón
        warped = np.ones((450, 450, 3), dtype=np.uint8) * 200
        # Casilla [1, 1] centro es (225, 225)
        cv2.circle(warped, (225, 225), 45, (230, 140, 20), -1)

        # Frame 1 a 3: la detección no es estable aún
        for _ in range(3):
            raw, stable = detector.update_and_detect_board(warped)
            self.assertEqual(raw[1][1], RATON)
            self.assertEqual(stable[1][1], EMPTY)

        # Frame 4: alcanza min_stable_frames (4), ahora debe ser estable
        raw, stable = detector.update_and_detect_board(warped)
        self.assertEqual(stable[1][1], RATON)


class TestGameDelayLogic(unittest.TestCase):
    def test_two_second_confirmation_delay(self):
        """Verifica que la jugada del usuario espere exactamente el delay de 2 segundos."""
        from main import GatoRatonApp
        import time

        app = GatoRatonApp(simulate=True)
        self.assertEqual(app.current_turn, RATON)

        # Tablero estable detectando una nueva ficha en [0, 1]
        stable_board = [[EMPTY for _ in range(3)] for _ in range(3)]
        stable_board[0][1] = RATON

        # Primer llamado: detecta la ficha pero NO debe registrar la jugada aún
        app.update_game_logic_from_vision(stable_board)
        self.assertEqual(app.logical_board[0][1], EMPTY)
        self.assertIsNotNone(app.pending_confirmation_info)
        self.assertEqual(app.pending_confirmation_info["cell"], (0, 1))
        self.assertAlmostEqual(app.pending_confirmation_info["remaining"], 2.0, delta=0.2)

        # Simular avance de 1 segundo: aún no debe registrarse
        app.candidate_first_detected_time -= 1.0
        app.update_game_logic_from_vision(stable_board)
        self.assertEqual(app.logical_board[0][1], EMPTY)

        # Simular avance a más de 2 segundos: ahora SÍ debe registrarse
        app.candidate_first_detected_time -= 1.1
        app.update_game_logic_from_vision(stable_board)
        self.assertEqual(app.logical_board[0][1], RATON)
        self.assertEqual(app.current_turn, GATO)
        self.assertIsNone(app.pending_confirmation_info)


if __name__ == "__main__":
    unittest.main()
