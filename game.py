import numpy as np
from dataclasses import dataclass
from prettytable import PrettyTable
from typing import List, Tuple
import json

@dataclass
class GameState:
    
    moves: Tuple   = tuple()
    on_turn: int   = 1
    terminal: bool = False
    reward: int    = 0
    size: int      = 15
    
    def get_board(self):
        
        board = np.zeros((self.size, self.size))

        if len(self.moves) > 0:
            if len(self.moves) < 3:
                board[self.moves[0]] = 1
            else:
                board[tuple(zip(*self.moves[::2]))] = 1
        if len(self.moves) > 1:
            if len(self.moves) < 4:
                board[self.moves[1]] = -1
            else:
                board[tuple(zip(*self.moves[1::2]))] = -1

        return board

    def __str__(self):
        x = PrettyTable()
        board = self.get_board()
        
        x.field_names = [' '] + [str(i+1) for i in range(len(board))]

        def change(c):
            if c == 0:
                return ''
            if c == 1:
                return 'X'
            return 'O'

        for i, row in enumerate(board):
            x.add_row([i+1]+list(map(change, row) ))

        return x.__str__()


class Game:

    def __init__(self, size=15):
        self.win_len = 5
        self.size = size
        self.ALL_MOVES = set((i,j) for i in range(15) for j in range(15))

    def get_new_state(self):
        return GameState()
    
    def available_moves(self, state:GameState):
        if state.terminal:
            return set()

        return self.ALL_MOVES - set(state.moves)
        
    def move(self, state:GameState, move):

        reward = -1 if self.winner_after_move(state, move) else 0
        on_turn = -state.on_turn
        moves = state.moves + (move,)
        terminal = reward != 0 or len(self.available_moves(state)) == 0

        return GameState(moves, on_turn, terminal, reward)

    def winner_after_move(self, state:GameState, move):
        
        board = state.get_board()
        board[move] = state.on_turn

        x, y = move
        for i in range(-4, 1):
            # win sequence in row
            if x+i >= 0 and x+4 < self.size:
                if np.sum(board[x+i:x+i+5,y]) == self.win_len * state.on_turn:
                    return True
            # win sequence in column
            if y+i >= 0 and y+4 < self.size:
                if np.sum(board[x,y+i:y+i+5]) == self.win_len * state.on_turn:
                    return True

            # win sequence in diagonal from bottom to top
            if x+i >= 0 and x+4 < self.size and y+i >= 0 and y+4 < self.size:
                points = np.array([[x+i+j,y+i+j] for j in range(5)])
                if np.sum(board[points[:,0],points[:,1]]) == self.win_len * state.on_turn:
                    return True
            
            # win sequence in diagonal from top to bottom
            if x+i >= 0 and x+4 < self.size and y+i >= 0 and y+4 < self.size:
                points = np.array([[x+i+j,y+i-j] for j in range(5)])
                if np.sum(board[points[:,0],points[:,1]]) == self.win_len * state.on_turn:
                    return True
        
        return False

    def _find_winner(self, board, player):

        def lines(board):

            for row in board:
                for i in range(0, len(row)-self.win_len+1):
                    if sum(row[i:i+self.win_len]) == player*self.win_len:
                        return True
            return False

        def columns(board):
            board = np.rot90(board)
            return lines(board)

        def diagonals(board):
            size = len(board)
            data = [np.diag(board, k=i) for i in range(-size+self.win_len-1, size-self.win_len+1, 1)]
            board = np.flip(board, axis=1)
            data += [np.diag(board, k=i) for i in range(-size+3, size-3, 1)]
            return lines(data)

        return lines(board) or columns(board) or diagonals(board)

if __name__ == "__main__":

    game = Game()
    s = game.get_new_state()
    print(s)
    moves = list((i,j) for i in range(5) for j in range(5))


    for m in moves:
        print(m)
        s = game.move(s, m)
        print(s)
        if s.terminal:
            print('Terminal')
            break
