import time

import tqdm

from game import Game
from nn.betazero import BetaZeroConfig
from nn.dataset import GameDatabase
from nn.nn_wraper import NetWrapper
from play.human_player import HumanPlayer
from play.mcts_player import MCTSPlayer
from play.play import play_game, play_games


from copy import deepcopy

class SelfPlayTraining:
    MODEL_PATH = "./"

    def __init__(self, model_path, model_conf: BetaZeroConfig, mcts_iters, storage_size):
        self.size = model_conf.board_size
        self.mcts_iters = mcts_iters
        self.storage_size = storage_size
        self.database = GameDatabase(storage_size, self.size)

        self.best_model = NetWrapper(model_conf)
        self.best_model.load_model(model_path)
        self.training_model = deepcopy(self.best_model)

        self.game = Game(self.size, win_size=5)



    def fill_database(self):

        p1 = MCTSPlayer(self.game, self.best_model, exploration=True, iters=self.mcts_iters)
        p2 = MCTSPlayer(self.game, self.best_model, exploration=True, iters=self.mcts_iters)

        states, _ = play_games(p1, p2, self.game, n_games=self.storage_size, random_start=True)
        self.database.add_games(states)

        if not self.database.full():

            while not self.database.full():
                state = play_game(p1, p2, self.game, show=False, random_start=True)
                self.database.add_games([state])

    def _next_population(self):
        self.best_model = self.training_model
        self.best_model.save_model(folder=self.MODEL_PATH, name=f"best_model_{time.time_ns()}")
        self.training_model = deepcopy(self.best_model)

    def self_play(self, n_games=20, n_iters=50, batch_size=5):

        p1 = MCTSPlayer(self.game, self.best_model, exploration=True, iters=self.mcts_iters)
        p2 = MCTSPlayer(self.game, self.training_model, exploration=True, iters=self.mcts_iters)

        states, score = play_games(p1, p2, self.game, n_games, random_start=True)
        print(f"Score: {score}")
        self.database.add_games(states)

        if score['p2'] / n_games > 0.55:
            self._next_population()
        else:
            self.training_model.self_play_train(self.database, batch_size, n_iters, loss_visual_step=int(n_iters//10))


if __name__ == "__main__":
    conf = BetaZeroConfig(board_size=15,
                          num_states=1,
                          num_res_layers=1,
                          hidden_dim=128,
                          input_cov_size=64,
                          output_cov_size=64,
                          device='cpu')
    wrapper = NetWrapper(conf)
    path = "C:\\Users\\user\\Documents\\GomokuProject\\resources\\models\\model_15x15_28_e"
    wrapper.load_model(path)

    game = Game(size=15, win_size=5)
    player1 = MCTSPlayer(game, wrapper, exploration=True, iters=100)#HumanPlayer(game)
    player2 = MCTSPlayer(game, wrapper, exploration=False, iters=100)

    state = play_game(player1, player2, game, show=True)


