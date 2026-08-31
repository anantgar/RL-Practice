from typing import Any
import numpy as np
from environment import CliffWalk

GAMMA=0.9
MAX_EPISODE_LENGTH = 100

def generate_episode(policy, env) -> list[tuple[Any, Any, float]]:
    states, actions, rewards = [], [], []
    state = env.get_random_state()
    env.set_state(state)
    terminal = False
    while not terminal and len(states) < MAX_EPISODE_LENGTH:
        action = policy[state]
        next_state, reward, terminal = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = next_state
    return states, actions, rewards

def mc_prediction(policy, value_fn, env, num_iterations:int, first_visit:bool=True):

    returns = {}
    for _ in range(num_iterations):
        states, actions, rewards = generate_episode(policy, env)
        G = 0
        for i in range(len(states) - 1, -1, -1):
            state, action, reward = states[i], actions[i], rewards[i]
            G = reward + GAMMA * G
            
            if state not in states[:i] and first_visit:
                if state not in returns:
                    returns[state] = []
                returns[state].append(G)
                value_fn[state] = np.mean(returns[state])
            

def main():
    env = CliffWalk(x_dim=12, y_dim=4, is_slippery=False)
    policy = np.random.choice(a=[0,1,2,3], size=env.x_dim * env.y_dim, replace=True)
    value_fn = 2 * np.random.rand(env.x_dim * env.y_dim) - 1
    mc_prediction(policy, value_fn, env, num_iterations=10, first_visit=True)
    print(value_fn.reshape(env.y_dim, env.x_dim))
    

if __name__ == "__main__":
    main()