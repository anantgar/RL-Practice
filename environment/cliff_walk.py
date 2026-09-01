from .environment import Environment
from enum import Enum
import random
import numpy as np

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
    
    def get_random_action(self):
        return random.randint(0, 3)
    
    def _compact_state(self, state):
        return state[1] * self.x_dim + state[0]
    
    def _uncompact_state(self, state):
        return (state % self.x_dim, state // self.x_dim)

    def visualize_action_value_function(self, value_function: np.ndarray):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon

        values = np.asarray(value_function).reshape(self.x_dim * self.y_dim, 4)

        fig, ax = plt.subplots(figsize=(self.x_dim * 1.1, self.y_dim * 1.1))
        ax.set_xlim(0, self.x_dim)
        ax.set_ylim(0, self.y_dim)
        ax.set_aspect("equal")
        ax.axis("off")

        start = (0, self.y_dim - 1)
        goal = (self.x_dim - 1, self.y_dim - 1)

        for y in range(self.y_dim):
            for x in range(self.x_dim):
                left, right = x, x + 1
                bottom, top = self.y_dim - y - 1, self.y_dim - y
                mid_x = (left + right) / 2
                mid_y = (bottom + top) / 2

                if (x, y) == start:
                    facecolor = "yellow"
                elif (x, y) == goal:
                    facecolor = "green"
                else:
                    facecolor = "white"

                ax.add_patch(
                    plt.Rectangle(
                        (left, bottom),
                        1,
                        1,
                        facecolor=facecolor,
                        edgecolor="black",
                        linewidth=1,
                    )
                )

                state = self._compact_state((x, y))
                action_values = [int(values[state, action]) for action in range(4)]

                top_left = (left, top)
                top_right = (right, top)
                bottom_right = (right, bottom)
                bottom_left = (left, bottom)
                center = (mid_x, mid_y)

                triangles = [
                    (0, [top_left, center, top_right]),
                    (1, [top_right, center, bottom_right]),
                    (2, [bottom_right, center, bottom_left]),
                    (3, [bottom_left, center, top_left]),
                ]

                for action, vertices in triangles:
                    ax.add_patch(
                        Polygon(
                            vertices,
                            closed=True,
                            facecolor="none",
                            edgecolor="black",
                            linewidth=0.5,
                        )
                    )
                    centroid_x = sum(v[0] for v in vertices) / 3
                    centroid_y = sum(v[1] for v in vertices) / 3
                    ax.text(
                        centroid_x,
                        centroid_y,
                        str(action_values[action]),
                        ha="center",
                        va="center",
                        fontsize=8,
                    )

        plt.tight_layout()
        plt.show()
