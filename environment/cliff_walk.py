from .environment import Environment
from enum import Enum
import random

class CliffWalk(Environment):

    def __init__(self, x_dim:int, y_dim:int, is_slippery:bool = False) -> None:
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.position = (0, self.y_dim - 1)
        self.is_slippery = is_slippery

    def step(self, action:int):
        if action == 0:
            new_position = (self.position[0], self.position[1] - 1)
        elif action == 1:
            new_position = (self.position[0] + 1, self.position[1])
        elif action == 2:
            new_position = (self.position[0], self.position[1] + 1)
        elif action == 3:
            new_position = (self.position[0] - 1, self.position[1])
        else:
            raise ValueError(f"Invalid action: {action}")
        
        if new_position == (self.x_dim - 1, self.y_dim - 1):
            return None, 0, True
        elif new_position[0] < 0 or new_position[0] >= self.x_dim or new_position[1] < 0 or new_position[1] >= self.y_dim or (new_position[0] > 0 and new_position[1] == self.y_dim - 1):
            return None, -100, True
        else:
            self.position = new_position
            return self._compact_state(new_position), -1, False

    def reset(self):
        self.position = (0, self.y_dim - 1)
        return self._compact_state(self.position)
    
    def get_state(self):
        return self._uncompact_state(self.position)
    
    def set_state(self, state:int):
        assert state >= 0 and state < self.x_dim * self.y_dim, "State out of bounds"
        self.position = self._uncompact_state(state)

    def get_random_state(self):
        return random.randint(0, self.x_dim * (self.y_dim - 1))
    
    def _compact_state(self, state):
        return state[1] * self.x_dim + state[0]
    
    def _uncompact_state(self, state):
        return (state % self.x_dim, state // self.x_dim)
