# Dynamic Programming - Policy Iteration

Repeated sequences of policy evaluation and improvement. We can alternate with one sweep of each.

Value iteration is the process combining both of these with limited number of sweeps across the entire state space.

In async DP methods, the evaluation and improvement process are interleaved at an even finer grain. Some value state improvements may only occur on a single state before switching to the other process.

This process is generally called *generalized polciy iteration*

## Policy evaluation

The process of creating a value function V(s) given a policy π

V(s) is the expected future return from state s following π forever

## Policy improvement

The process of improving π(s) using our V(s)

Setting π(s) = the action that leads to the highest expected future return

# Monte Carlo Methods



## Advantages over DP

1. MC works when we don't have a reliable state transition model. DP requires us to know the probability of each state transition (we need to know the probability of reaching state s' given action a in state s). For example, blackjack environment works better with MC since it's uncertain what s' will be after the player hits.
2. State value estimates in MC are independent of each other, so we could create precise estimations for a select set of states and not all states. In DP, value estimates for state s are based on the estimates of its neighboring "transition" states.

## MC action value estimation

Estimation of value associated with (state, action) pairs.

Issue with this is that the state-action space is much larger than the state space. With a deterministic policy, we may only learn one action in a given state. It's important to continually explore and capture episodes including all actions from any state. Solution is to use a stochastic policy with a nonzero probability of selecting all actions in each state.

## Off Policy Learning

In order to learn the optimal action in a state s, we need to visit it, but we're only visiting states by following our "optimal policy." To develop our policy we need to explore the state-action pairs we haven't already discovered.

Maintain one optimal policy that learns the best actions in all states. Keep a separate policy that encourages exploration.

### Prediction Problem

Given only episodes following fixed policy b, learn the value function for a target policy π.

We assume that every action taken under π is also taken under b occasionally.

#### Importance Sampling

The probability of the trajectory occurring in the target policy over the probability of it occurring in the behavior policy. The state transition probability cancels out, so it's the ratio of the target policy to the behavior policy across the trajectory.

If we multiply the return observed by b, G_t, with the ratio, we have the expectation of the return under π.

When we average this across all visits of state s, we can either uniformly average or weighted average with the IS ratios.