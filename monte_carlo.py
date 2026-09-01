from typing import Any
import numpy as np
from environment import CliffWalk

GAMMA=0.9
MAX_EPISODE_LENGTH = 100

def generate_episode(policy, env, S_0:int, A_0:int|None=None) -> list[tuple[Any, Any, float]]:
    states, actions, rewards = [], [], []
    state = S_0
    action = A_0
    env.set_state(state)
    terminal = False
    while not terminal and len(states) < MAX_EPISODE_LENGTH:
        action = policy[state] if action is None else action
        next_state, reward, terminal = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = next_state
    return states, actions, rewards

def mc_prediction(policy, value_fn, env, num_iterations:int, first_visit:bool=True):
    returns = {}
    for _ in range(num_iterations):
        S_0 = env.get_random_state()
        states, actions, rewards = generate_episode(policy, env, S_0)
        G = 0
        for i in range(len(states) - 1, -1, -1):
            state, action, reward = states[i], actions[i], rewards[i]
            G = reward + GAMMA * G
            
            if state not in states[:i] and first_visit:
                if state not in returns:
                    returns[state] = []
                returns[state].append(G)
                value_fn[state] = np.mean(returns[state])
            

def exploring_starts(policy, value_fn, env, num_iterations:int, first_visit:bool=True):

    def average_return(mean, n, g:float) -> float:
        return (mean * n + g) / (n + 1) , n + 1

    returns = {}
    for _ in range(num_iterations):
        S_0, A_0 = env.get_random_state(), env.get_random_action()
        states, actions, rewards = generate_episode(policy, env, S_0, A_0)
        G = 0
        for i in range(len(states) - 1, -1, -1):
            state, action, reward = states[i], actions[i], rewards[i]
            G = reward + GAMMA * G

            if (state, action) not in zip(states[:i], actions[:i]) and first_visit:
                if (state, action) not in returns:
                    returns[(state, action)] = (0.0, 0)
                returns[(state, action)] = average_return(*returns[(state, action)], G)
                value_fn[state, action] = returns[(state, action)][0]
                policy[state] = np.argmax(value_fn[state])


def main():
    env = CliffWalk(x_dim=12, y_dim=4, is_slippery=False)
    policy = np.random.choice(a=[0,1,2,3], size=env.x_dim * env.y_dim, replace=True)
    value_fn = 2 * np.zeros((env.x_dim * env.y_dim, 4))
    exploring_starts(policy, value_fn, env, num_iterations=1000, first_visit=True)
    env.visualize_action_value_function(value_fn)
    

if __name__ == "__main__":
    main()