import glob
import json
from queue import Queue

import numpy as np

from game import Game, GameState


class TrainingDataLoader:
    RESOURCE_FOLDER = "C:\\Users\\user\\Documents\\GomokuProject\\resources"

    def __init__(self, board_size, batch_size=32, only_winning=True):
        self.board_size = board_size
        self.batch_size = batch_size
        self.game = Game(size=self.board_size, win_size=5)
        self.i = 0
        self.batches = []
        self.data = []
        self.only_winning = only_winning

    def _load_file(self, file):
        with open(file, 'r') as f:
            data = json.load(f)

        return data

    def load(self):

        games = []

        for file_name in glob.iglob(f"{self.RESOURCE_FOLDER}/games_{self.board_size}x{self.board_size}/*.json"):
            print(file_name)
            result = file_name[-6]
            if result == "X":
                value = 1
            elif result == "D":
                value = 0
            else:
                value = -1

            games.extend([(x, value) for x in self._load_file(file_name)])

        self.data = games
        self._shuffle_batch_indices()

    def _shuffle_batch_indices(self):
        indices = np.arange(len(self.data))
        np.random.shuffle(indices)
        self.batches = []
        for i in range(0, len(indices), self.batch_size):
            self.batches.append(indices[i: i + self.batch_size])

    def __len__(self):
        return len(self.data)

    def get_batch(self, batch_size=1):
        games = [self.data[i] for i in self.batches[self.i]]
        boards, probs, values = decompose_games(games, self.board_size, self.game, self.only_winning)

        indices = np.arange(len(boards))
        np.random.shuffle(indices)
        boards = boards[indices]
        probs = probs[indices]
        values = values[indices]

        self.i += 1
        if self.i == len(self.batches):
            self._shuffle_batch_indices()
            self.i = 0

        boards = boards.reshape(-1,1, self.board_size, self.board_size)

        return boards, probs, values

class GameDatabase:

    def __init__(self, size, board_size, batch_size):
        self.size = size
        self.board_size = board_size
        self.game = Game(size=self.board_size, win_size=5)
        self.batch_size = self.batch_size
        self.queue = Queue(maxsize=self.size)

    def add_games(self, states: list[GameState]):
        for state in states:
            if state.terminal and state.reward != 0:
                value = -state.on_turn
                self.queue.put((state.moves, value))

    def get_batch(self, batch_size):
        #TODO create batch -> batch_size is number of games or number of positions??
        #TODO use list as queue due to random choice
        return []



def decompose_games(data, board_size, game, only_winning=True):
    boards, probs, values = [], [], []
    for moves, value in data:
        state = game.get_new_state()
        for move in moves:
            if value > 0 or not only_winning: # only_winning -> value > 0
                board = state.get_board() * state.on_turn
                prob = np.zeros((board_size, board_size))
                prob[move] = 1

                boards.append(board)
                probs.append(prob)
                values.append(value)

            state = game.move(state, move)
            value = -value

    return np.array(boards), np.array(probs), np.array(values)


if __name__ == "__main__":
    file = "C:\\Users\\user\\Documents\\GomokuProject\\resources\\games_20x20\\games_2019_X.json"
