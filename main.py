from game import Game
from nn.betazero import BetaZeroConfig
from nn.nn_wraper import NetWrapper
from play.human_player import HumanPlayer
from play.mcts_player import MCTSPlayer
from play.play import play_game

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
    player1 = MCTSPlayer(game, wrapper, exploration=False, iters=100)#HumanPlayer(game)
    player2 = MCTSPlayer(game, wrapper, exploration=False, iters=100)

    state = play_game(player1, player2, game, show=True)