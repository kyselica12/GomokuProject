from unittest import TestCase

from game import GameState, Game
import numpy as np


class GameStateTests(TestCase):

    def test_empty_state(self):
        s = GameState(size=5)
        board = s.get_board()

        self.assertEqual(len(board), 5, msg=f'Wrong first dimenstion.')
        self.assertEqual(len(board[0]), 5, msg=f'Wrong second dimenstion.')

    def test_move_on_place(self):
        moves = ((0, 0),)
        s = GameState(moves, size=4)
        board = s.get_board()

        target = np.array([
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])

        self.assertEqual(board[0][0], 1, 'Wrong move marked')
        self.assertTrue(np.all(board == target), "Correct Board state.")

    def test_less_than_3_moves(self):
        moves = ((0, 0), (1, 1), (2, 2))
        s = GameState(moves, size=4)
        board = s.get_board()

        target = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0]
        ])

        self.assertTrue(np.all(board == target), "Incorrect Board state.")

    def test_game(self):
        moves = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 0), (0, 2), (0, 3))
        s = GameState(moves, size=4)
        board = s.get_board()

        target = np.array([
            [1, 1, 1, -1],
            [-1, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0]
        ])


class GameTests(TestCase):
    standard_game = Game(size=15, win_size=5)

    def test_get_new_state(self):
        size = 10
        game = Game(size=size)
        s = game.get_new_state()
        board = s.get_board()

        self.assertTrue(np.all(board == 0), "Not empty board")
        self.assertEqual(len(board), size, "Wrong first dimension")
        self.assertEqual(len(board[0]), size, "Wrong second dimension")

    def test_available_moves(self):
        size = 10
        game = Game(size=size)
        s = game.get_new_state()

        self.assertEqual(len(game.available_moves(s)), size ** 2, "Wrong number of moves.")

        s = GameState(moves=((0, 0),), size=size)
        self.assertEqual(len(game.available_moves(s)), size ** 2 - 1, "Wrong number of moves.")

    def test_move(self):
        s = self.standard_game.get_new_state()
        on_turn = 1

        for i in range(self.standard_game.size):
            for j in range(self.standard_game.size):
                move = (i, j)
                s = self.standard_game.move(s, move)
                self.assertEqual(s.get_board()[i, j], on_turn, "Wrong move marked")
                on_turn *= -1

    def test_winner_after_move_with_win_sequence_3(self):
        size = 10
        win_len = 3
        game = Game(size=size, win_size=win_len)

        moves = ((0, 0), (1, 1), (0, 1), (2, 2))
        s = GameState(moves=moves, size=size, on_turn=1)
        winning_move = (0, 2)

        s2 = GameState(moves=moves[:-1], size=size, on_turn=-1)

        self.assertFalse(game.winner_after_move(s2, moves[-1]), "Not winning move")
        self.assertTrue(game.winner_after_move(s, winning_move), "Win with sequence lenght of 3")

    def test_winner_after_move_with_win_sequence_5(self):
        size = 10
        win_len = 5
        game = Game(size=size, win_size=win_len)

        moves = ((0, 0), (1, 1), (0, 1), (2, 2), (0, 2), (3, 3), (0, 3), (4, 4))
        s = GameState(moves=moves, size=size, on_turn=1)
        winning_move = (0, 4)

        self.assertTrue(game.winner_after_move(s, winning_move), "Win with sequence lenght of 5")

    def test_winner_after_move_horizontal_line(self):
        def winner_test(moves, move, msg):
            s = GameState(size=self.standard_game.size, moves=moves, on_turn=1)
            self.assertTrue(self.standard_game.winner_after_move(s, move), f"Winning move is {msg} in sequence")

        moves = moves = ((0, 0), (1, 1), (0, 1), (2, 2), (0, 2), (3, 3), (0, 3), (4, 4), (0, 4))

        winner_test(moves[:-1], moves[-1], "last")
        winner_test(moves[1:], moves[0], "first")

        # need to reverse order in second half of moves because we need to
        # continue with move of play one
        winner_test(moves[:4] + moves[5:][::-1], moves[4], "in the middle")

    def test_winner_after_move_diagonal_line(self):
        def winner_test(moves, move, msg):
            s = GameState(size=self.standard_game.size, moves=moves, on_turn=1)
            self.assertTrue(self.standard_game.winner_after_move(s, move), f"Winning move is {msg} in sequence")

        moves = moves = ((0, 0), (0, 1), (1, 1), (0, 2), (2, 2), (0, 3), (3, 3), (0, 4), (4, 4))

        winner_test(moves[:-1], moves[-1], "last")
        winner_test(moves[1:], moves[0], "first")

        # need to reverse order in second half of moves because we need to
        # continue with move of play one
        winner_test(moves[:4] + moves[5:][::-1], moves[4], "in the middle")

    def test_winner_after_move_vertical_line(self):
        def winner_test(moves, move, msg):
            s = GameState(size=self.standard_game.size, moves=moves, on_turn=1)
            self.assertTrue(self.standard_game.winner_after_move(s, move), f"Winning move is {msg} in sequence")

        moves = moves = ((0, 0), (1, 1), (1, 0), (2, 2), (2, 0), (3, 3), (3, 0), (4, 4), (4, 0))

        winner_test(moves[:-1], moves[-1], "last")
        winner_test(moves[1:], moves[0], "first")

        # need to reverse order in second half of moves because we need to
        # continue with move of play one
        winner_test(moves[:4] + moves[5:][::-1], moves[4], "in the middle")

