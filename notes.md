# Policy Iteration

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

