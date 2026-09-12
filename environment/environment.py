from typing import Any

from abc import ABC, abstractmethod
from enum import Enum

class Environment(ABC):

    def __init__(self):
        pass

    # Returns tuple of next state, reward, and whether the episode is terminated
    @abstractmethod
    def step(self, action) -> tuple[Any, float, bool]:
        pass

    @abstractmethod
    def reset(self):
        pass