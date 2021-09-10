from abc import ABC, abstractmethod

from game import GameState


class Model(ABC):

    @abstractmethod
    def predict(self, state: GameState):
        pass

