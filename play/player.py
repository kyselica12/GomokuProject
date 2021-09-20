from abc import ABC, abstractmethod

from game import GameState


class Player(ABC):

    def __init__(self, game):
        self.game = game

    @abstractmethod
    def move(self, state: GameState):
        pass