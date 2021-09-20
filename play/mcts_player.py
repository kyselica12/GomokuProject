import numpy as np

from game import GameState
from mcts import MCTS, Node
from play.player import Player
from nn.nn_wraper import NetWrapper

class MCTSPlayer(Player):

    def __init__(self, game,net, exploration=False, discount=0, iters=50):
        super().__init__(game)
        self.net_wraper = net
        self.exploration = exploration
        self.mcts = MCTS(self.net_wraper, self.game, discount=discount)
        self.iters = iters

    def move(self, state: GameState):

        node: Node = self.mcts.run(state, num_simulations=self.iters)
        if self.exploration:
            move = self._move_from_distribution(node, state)
        else:
            move = self._most_visited_move(node)

        return move

    def _most_visited_move(self, node: Node):
        move = max(node.children, key=lambda m: (node.children[m].visit_count, m))
        return move

    def _move_from_distribution(self, node: Node, state):
        available_moves = self.game.available_moves(state)
        moves, probs = [], []
        for m in available_moves:
            probs.append(node.children[m].visit_count / node.visit_count)
            moves.append(m)

        winner_i = np.random.choice(len(available_moves), 1, probs)
        move = moves[winner_i]

        return move






