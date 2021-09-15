from abc import ABC, abstractmethod

from game import GameState
import numpy as np


class Model(ABC):

    @abstractmethod
    def predict(self, state: GameState):
        pass

class DummyModel(Model):

    def __init__(self, size=15):
        self.size = size

    def predict(self, state: GameState):
        reward = 0
        probs = np.ones((self.size, self.size)) * 1 / (self.size ** 2)

        return reward, probs
