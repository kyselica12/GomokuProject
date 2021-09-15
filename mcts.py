import math
from typing import Dict

from game import GameState, Game
from model import Model


class Node:
    def __init__(self, prior, state=None):
        self.prior: float = prior
        self.reward: float = 0
        self.children: Dict = None
        self.visit_count = 0
        self.value_sum = 0
        self.state: GameState = state

    def value(self):
        return self.value_sum / self.visit_count

    def expand(self, model: Model, game: Game, state: GameState):
        reward, probs = model.predict(state)
        self.reward = reward
        available_moves = game.available_moves(state)

        self.children = {}
        for (r, c) in available_moves:
            self.children[(r,c)] = Node(prior=probs[r,c], state=game.move(self.state, (r, c)))

    def select_child(self):
        # c -> tuple (move, node) -> c[1] is the children node
        return max(self.children.items(), key=lambda c: ucb_score(self, c[1]))

    def expanded(self):
        return self.children is not None

    def __str__(self):
        return f"{self.visit_count}, {self.expanded()}"

class MCTS:

    def __init__(self, model, game, discount=0):
        self.model: Model = model
        self.game: Game = game
        self.discount: float = discount

    def run(self, state, num_simulations=5):
        root = Node(0, state=state)
        # EXPAND root
        root.expand(self.model, self.game, state)

        for _ in range(num_simulations):
            node = root
            search_path = [node]
            # SELECT
            while node.expanded():
                move, node = node.select_child()
                search_path.append(node)

            next_state = node.state

            if next_state.terminal:
                value = next_state.reward
            else:
                # EXPAND
                node.expand(self.model, self.game, next_state)
                value = node.reward

            self.backup(search_path, value)

        return root

    def backup(self, search_path, value):

        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            value = -value * (1-self.discount)


def ucb_score(parent, child):
    prior_score = child.prior * math.sqrt(parent.visit_count) / (child.visit_count + 1)
    if child.visit_count > 0:
        # The value of the child is from the perspective of the opposing player
        value_score = -child.value()
    else:
        value_score = 0

    return value_score + prior_score

if __name__ == "__main__":
    a = Node(0)
    b = Node(1)

    a.children[1] = 2
    print(a.children)
    print(b.children)