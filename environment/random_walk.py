from .environment import Environment
from enum import Enum


class RandomWalk(Environment):

    def __init__(self, num_states:int = 5):
        self.num_states = num_states
        self.bounds = tuple((0, self.num_states-1))
        self.position = self.num_states // 2

    def step(self, action):
        new_pos = self.position - 1 if action == RandomWalkAction.LEFT else self.position + 1
        terminal = False
        reward = 0
        if new_pos < self.bounds[0] or new_pos > self.bounds[1]:
            terminal = True
            if new_pos > self.bounds[1]:
                reward = 1

        self.position = new_pos

        return self.position if not terminal else None, reward, terminal

    def reset(self):
        self.position = self.num_states // 2
        return self.position

    def get_state(self):
        return self.position

    def set_state(self, state:int):
        assert state >= self.bounds[0] and state <= self.bounds[1], "State out of bounds"
        self.position = state

class RandomWalkAction(Enum):
    LEFT = 0
    RIGHT = 1