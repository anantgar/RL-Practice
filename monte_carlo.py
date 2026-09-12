from typing import Any
import numpy as np
from environment import CliffWalk, Blackjack
from config import GAMMA, MAX_EPISODE_LENGTH, EPSILON

def generate_episode(policy:np.ndarray, env, S_0:int, A_0:int|None=None, greedy:bool=False) -> list[tuple[Any, Any, float]]:
    states, actions, rewards = [], [], []
    state = S_0
    action = A_0
    env.set_state(state)
    terminal = False
    while not terminal and len(states) < MAX_EPISODE_LENGTH:
        if action is None:
            if greedy:
                action = np.argmax(policy[state])
            else:
                action = np.random.choice(a=np.arange(env.num_actions), p=policy[state])
        next_state, reward, terminal = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = next_state
        action = None
    return states, actions, rewards

def mc_prediction_off_policy(target_policy:np.ndarray, value_fn:np.ndarray, env, num_iterations:int):
    cumulative_weights = {}

    for _ in range(num_iterations):
        behavior_policy = np.random.rand(target_policy.shape) + 0.1
        S_0 = env.get_random_state()
        states, actions, rewards = generate_episode(behavior_policy, env, S_0)
        G = 0
        W = 1
        for i in range(len(states) - 1, -1, -1):
            if W == 0:
                break
            state, action, reward = states[i], actions[i], rewards[i]
            G = reward + GAMMA * G
            cumulative_weights[state, action] = cumulative_weights.get((state, action), 0.0) + W
            value_fn[state, action] += (W / cumulative_weights[state, action]) * (G - value_fn[state, action])
            W *= target_policy[state, action] / behavior_policy[state, action]
            

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

            if not first_visit or (state, action) not in zip(states[:i], actions[:i]):
                if (state, action) not in returns:
                    returns[(state, action)] = (0.0, 0)
                returns[(state, action)] = average_return(*returns[(state, action)], G)
                value_fn[state, action] = returns[(state, action)][0]
                policy[state] = np.argmax(value_fn[state])



def main():
    # env = CliffWalk(x_dim=12, y_dim=4, is_slippery=False)
    # policy = np.random.choice(a=[0,1,2,3], size=env.x_dim * env.y_dim, replace=True)
    # value_fn = 2 * np.zeros((env.x_dim * env.y_dim, 4))
    # exploring_starts(policy, value_fn, env, num_iterations=10000, first_visit=False)
    # env.visualize_action_value_function(value_fn)
    # print(policy.reshape(env.y_dim, env.x_dim))

    env = Blackjack()
    policy = np.random.choice(a=[0,1], size=env.num_states, replace=True)
    value_fn = 2 * np.zeros((env.num_states, env.num_actions))
    exploring_starts(policy, value_fn, env, num_iterations=10000, first_visit=False)
    print(policy.reshape(env.num_states))
    print(value_fn.reshape(env.num_states, env.num_actions))

if __name__ == "__main__":
    main()