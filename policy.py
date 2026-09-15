import numpy as np
from environment import RandomWalk, RandomWalkAction
from config import GAMMA, THETA

def init_fn(num_states:int, seed:int = 42):
    np.random.seed(seed)
    value_fn = 2 * np.random.rand(num_states) - 1
    policy = np.random.choice(a=[0,1], size=num_states, replace=True)
    return policy, value_fn


def policy_evaluation(policy, value_fn, env):
    num_states = policy.shape[0]

    while True:
        delta = 0.0
        for i in range(num_states):
            old_value = value_fn[i]
            env.set_state(i)
            next_state, reward, terminal = env.step(RandomWalkAction(policy[i]))
            v = reward
            if not terminal:
                v = v + GAMMA * value_fn[next_state]
            value_fn[i] = v
            delta = max(delta, abs(old_value - v))
        if delta < THETA:
            break


def policy_improvement(policy, value_fn, env):

    def get_best_action(env, state):
        num_actions = 2
        a = np.empty(num_actions)
        for i in range(num_actions):
            env.set_state(state)
            next_state, reward, terminal = env.step(RandomWalkAction(i))
            a[i] = reward 
            if not terminal:
                a[i] = a[i] + GAMMA * value_fn[next_state]
        return np.argmax(a)


    stable = True
    num_states = policy.shape[0]

    for i in range(num_states):
        old_action = policy[i]
        policy[i] = get_best_action(env, i)
        if old_action != policy[i]:
            stable = False

    return stable

def policy_iteration(num_states:int, max_iterations:int = 100, env:RandomWalk = None):
    policy, value_fn = init_fn(num_states, seed=42)

    for _ in range(max_iterations):

        policy_evaluation(policy, value_fn, env)
        stable = policy_improvement(policy, value_fn)
        if stable:
            break

    print("Policy:")
    print(policy)
    print("Value Function:")
    print(value_fn)

def value_iteration(num_states:int, max_iterations:int = 100, env:RandomWalk = None):
    policy, value_fn = init_fn(num_states, seed=42)
    num_actions = 2
    while True:
        delta = 0.0
        for i in range(num_states):
            old_value = value_fn[i]

            a_values = np.empty(num_actions)
            for j in range(num_actions):
                env.set_state(i)
                next_state, reward, terminal = env.step(RandomWalkAction(j))
                a_values[j] = reward
                if not terminal:
                    a_values[j] = a_values[j] + GAMMA * value_fn[next_state]
            value_fn[i] = np.max(a_values)
            delta = max(delta, abs(old_value - value_fn[i]))
        if delta < THETA:
            break
    
    policy_improvement(policy, value_fn)
    print("Policy:")
    print(policy)
    print("Value Function:")
    print(value_fn)

def main():
    print("* * * Policy Iteration:")
    policy_iteration(5, env=RandomWalk(5))
    print("* * * Value Iteration:")
    value_iteration(5, env=RandomWalk(5))

if __name__ == "__main__":
    main()