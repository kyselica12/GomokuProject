from abc import ABC, abstractmethod

from game import GameState


class Player(ABC):

    @abstractmethod
    def move(self, state: GameState):
        pass