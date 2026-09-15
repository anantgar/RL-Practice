import numpy as np
from config import LEARNING_RATE, GAMMA, NUM_ITERATIONS, NUM_EPISODES, MAX_EPISODE_LENGTH, EPSILON
from monte_carlo import generate_episode
from typing import Callable
from environment import CliffWalk

def epsilon_greedy_policy(policy:np.ndarray, state:int, epsilon:float):
    return np.argmax(policy[state]) if np.random.random() > epsilon else np.random.choice(a=policy.shape[1])

def td_0_prediction(value_fn:np.ndarray, env, action_selection:Callable=epsilon_greedy_policy):
    for _ in range(NUM_EPISODES):
        state = env.get_random_state()
        env.set_state(state)
        terminal = False
        num_steps = 0
        while not terminal and num_steps < MAX_EPISODE_LENGTH:
            action = action_selection(value_fn, state, EPSILON)
            next_state, reward, terminal = env.step(action)
            value_fn[state] += LEARNING_RATE * (reward + GAMMA * value_fn[next_state] - value_fn[state])
            state = next_state
            num_steps += 1

def td_control(value_fn:np.ndarray, env, off_policy:bool=False, action_selection:Callable=epsilon_greedy_policy):
    for _ in range(NUM_EPISODES):
        state = env.get_random_state()
        env.set_state(state)
        action = action_selection(value_fn, state, EPSILON)
        terminal = False
        num_steps = 0
        while not terminal and num_steps < MAX_EPISODE_LENGTH:
            next_state, reward, terminal = env.step(action)
            next_action = action_selection(value_fn, next_state, EPSILON) if not terminal else None
            if terminal:
                value_fn[state, action] += LEARNING_RATE * (reward - value_fn[state, action])
            elif off_policy:
                value_fn[state, action] += LEARNING_RATE * (reward + GAMMA * np.max(value_fn[next_state]) - value_fn[state, action])
            else:
                value_fn[state, action] += LEARNING_RATE * (reward + GAMMA * value_fn[next_state, next_action] - value_fn[state, action])
            state, action = next_state, next_action
            num_steps += 1

def main():
    env = CliffWalk(12, 4, False)
    policy = np.ones((12*4, 4)) / 4
    value_fn = np.zeros_like(policy)
    td_control(value_fn, env, False)
    env.visualize_action_value_function(value_fn)

if __name__ == "__main__":
    main()