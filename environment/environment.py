from typing import Any

from abc import ABC, abstractmethod
from enum import Enum

class Environment(ABC):

    def __init__(self):
        pass

    # Returns dict of reward, is terminated, and next state
    @abstractmethod
    def step(self, action) -> tuple[Any, float, bool]:
        pass

    @abstractmethod
    def reset(self):
        pass