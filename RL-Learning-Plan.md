# Reinforcement Learning From Scratch: Job-Prep Curriculum

This plan is organized around the algorithms and concepts most likely to matter for reinforcement-learning engineering and research roles. It is intentionally learning-by-doing focused: read only the sections of a paper needed to understand and implement the work, then spend most of the time coding, testing, and debugging.

Nothing from the original plan has been removed. Less urgent material has been moved into secondary implementation, survey-only, or optional-branch sections.

The [Spinning Up key-papers list](https://spinningup.openai.com/en/latest/spinningup/keypapers.html) remains useful as a broader reference, but it should not be treated as a sequence to complete.

## How to use this plan

For each core item:

1. Read the abstract, introduction, algorithm section, key equations, and experiment setup.
2. Skip most related work and proofs on the first pass.
3. Write the update equations and pseudocode from memory.
4. Implement the smallest faithful version.
5. Add unit tests for the important equations.
6. Run two or three small experiments.
7. Write a short note: what changed, why it should help, and what actually happened.

Keep “faithful implementation” and “improved implementation” separate. For example, implement original DDPG first, then implement TD3 as a separate successor.

At 4–5 hours per day, use roughly:

- 20–40 minutes reading;
- 20–40 minutes deriving and planning;
- 2–3 hours coding;
- 1–2 hours testing, debugging, and experiments.

Do not read the entire Sutton and Barto textbook. Use selected chapters and return to individual sections when an implementation exposes a gap.

## Targeted textbook reading

Use [Reinforcement Learning: An Introduction — Sutton and Barto](http://incompleteideas.net/book/the-book-2nd.html) selectively:

- Chapters 1–3: terminology, bandits, and finite MDPs;
- Chapters 4–6: dynamic programming, Monte Carlo methods, and TD learning;
- Chapter 7: n-step methods;
- Chapter 8: planning and learning;
- Chapter 13: policy-gradient methods.

Initially skip detailed proofs, eligibility traces, function approximation theory, and average-reward material. Return to them when the corresponding implementation requires them.

## Core curriculum: execute these steps in order

This is the slimmed job-prep path. Do not read a group of papers and postpone the work. Finish each step before moving on.

For every step, the sequence is: **read the listed sections → implement the listed task → run the validation checks → record what you learned**.

### Step 0 — Build the small experimental foundation

Read:

- Sutton and Barto, [Chapters 1–4](http://incompleteideas.net/book/the-book-2nd.html): MDPs, bandits, dynamic programming, and Bellman equations.

Build:

- `Env.reset()` and `Env.step(action)` interfaces;
- deterministic seeding;
- episode return and environment-step logging;
- a tabular policy and value representation;
- a simple CSV/JSON experiment logger;
- exact policy evaluation and value iteration.

Implement this task:

- Five-state Random Walk: start in the center, move left or right, terminate at either endpoint, reward `+1` only on reaching the right endpoint and `0` otherwise.

Done when:

- value iteration matches the analytical solution;
- a random policy, greedy policy, and optimal policy can be evaluated separately;
- one run can be reproduced from its seed.

### Step 1 — Monte Carlo, TD(0), SARSA, and Q-learning

Read:

1. [Learning to Predict by the Methods of Temporal Differences — Sutton](https://ics.uci.edu/~dechter/courses/ics-295/winter-2018/papers/Sutton-td.pdf): TD prediction and bootstrapping sections.
2. [On-Line Q-Learning Using Connectionist Systems — Rummery and Niranjan](https://www.cs.utexas.edu/~shivaram/readings/b2hd-RummeryNiranjan1994.html): SARSA update.
3. [Q-Learning — Watkins and Dayan](https://doi.org/10.1007/BF00992698): Q-learning update and convergence assumptions.
4. Sutton and Barto, [Chapters 5–6](http://incompleteideas.net/book/the-book-2nd.html): Monte Carlo and TD control.

Implement:

1. Monte Carlo prediction.
2. TD(0) prediction.
3. SARSA.
4. Q-learning.
5. Epsilon-greedy action selection.

Use this exact task:

- Cliff Walking: a 4×12 grid; start at the lower-left, goal at the lower-right, the cells between them are cliffs, each normal step has reward `-1`, falling off the cliff gives `-100` and resets the episode.

Validation:

- Plot or log learned values for Random Walk.
- Compare SARSA and Q-learning on Cliff Walking.
- Explain why SARSA learns a safer path while Q-learning tends toward the shortest risky path.

### Step 2 — REINFORCE and the policy-gradient theorem

Read:

1. [Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning — Williams](https://doi.org/10.1007/BF00992696): REINFORCE update.
2. [Policy Gradient Methods for Reinforcement Learning with Function Approximation — Sutton et al.](https://papers.nips.cc/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html): policy-gradient theorem and baseline conditions.
3. Sutton and Barto, [Chapter 13](http://incompleteideas.net/book/the-book-2nd.html): policy-gradient basics.

Implement:

1. Tabular softmax REINFORCE on a small Gridworld.
2. MLP REINFORCE on your CartPole-like environment.
3. Reward-to-go returns.
4. A learned state-value baseline.
5. A Gaussian policy for your continuous Pendulum-like environment.

Use this CartPole-like task:

- Four-dimensional state: cart position, cart velocity, pole angle, and pole angular velocity.
- Two discrete actions: left and right force.
- Reward `+1` per step alive.
- Terminate when the cart or pole exceeds a fixed limit or after 500 steps.

Validation:

- Compare full-return REINFORCE against reward-to-go.
- Compare variance with and without a learned baseline.
- Numerically compare the analytic policy gradient with a finite-difference estimate on a tiny network.

### Step 3 — DQN

Read:

1. [Playing Atari with Deep Reinforcement Learning — Mnih et al.](https://arxiv.org/abs/1312.5602): replay buffer and target network.
2. [Human-Level Control Through Deep Reinforcement Learning — Mnih et al.](https://www.nature.com/articles/nature14236): full DQN training details.

Implement:

1. Replay buffer.
2. Two-layer MLP `state → 64 → 64 → action values`.
3. Online and target Q-networks.
4. Bellman target with terminal masking.
5. Epsilon-greedy exploration.
6. Periodic target-network copying.

Use the CartPole-like task from Step 2. Run a second experiment on a small stochastic Gridworld where value iteration provides a reference solution.

Validation:

- Verify target values do not receive gradients.
- Verify terminal transitions do not bootstrap.
- Compare learned action values against value iteration on the small Gridworld.
- Beat a random policy consistently on the CartPole-like task.

### Step 4 — Double DQN and prioritized replay

Read:

1. [Deep Reinforcement Learning with Double Q-Learning — van Hasselt et al.](https://arxiv.org/abs/1509.06461): decoupled action selection and evaluation.
2. [Prioritized Experience Replay — Schaul et al.](https://arxiv.org/abs/1511.05952): priority definition and importance-sampling correction.

Implement in two separate changes:

1. Double DQN using the existing DQN code.
2. Prioritized replay using a simple proportional-priority implementation.

Use this Acrobot-like task:

- Two connected pendulum links.
- Three discrete torques: negative, zero, positive.
- Reward `-1` per step until the tip reaches the target height.
- Fixed episode horizon.

Validation:

- Run vanilla DQN and Double DQN with identical seeds.
- Log Q-value magnitude and return separately.
- Compare uniform versus prioritized replay.
- Verify importance-sampling weights are applied only to the loss.

### Step 5 — A2C and generalized advantage estimation

Read:

1. [Asynchronous Methods for Deep Reinforcement Learning — Mnih et al.](https://arxiv.org/abs/1602.01783): actor-critic and asynchronous rollout ideas.
2. [High-Dimensional Continuous Control Using Generalized Advantage Estimation — Schulman et al.](https://arxiv.org/abs/1506.02438): GAE derivation and bias-variance tradeoff.

Implement:

1. Synchronous A2C with separate actor and value networks.
2. Batched rollout collection from several copies of your custom CartPole-like environment.
3. GAE with configurable `gamma` and `lambda`.

Use:

- CartPole-like environment first;
- Pendulum second.

Validation:

- Compare Monte Carlo returns, one-step TD targets, and GAE advantages.
- Verify rollout termination and time-limit masks.
- Compare one environment copy against several synchronous copies.

Do not implement multiprocessing A3C yet. It is retained in the secondary queue as a systems exercise.

### Step 6 — Natural policy gradient, TRPO, and PPO

Read:

1. [A Natural Policy Gradient — Kakade](https://proceedings.neurips.cc/paper/2001/hash/4b86abe48d358ecf194c56c69108433c-Abstract.html): natural-gradient motivation.
2. [Trust Region Policy Optimization — Schulman et al.](https://arxiv.org/abs/1502.05477): surrogate objective, KL constraint, and line search.
3. [Proximal Policy Optimization Algorithms — Schulman et al.](https://arxiv.org/abs/1707.06347): clipped surrogate objective.

Implement in this order:

1. Natural policy gradient on the CartPole-like task.
2. TRPO with conjugate-gradient Fisher-vector products and backtracking line search.
3. PPO-Clip using the same rollout and advantage code.

Use the same network family throughout:

- actor: `state → 64 → 64 → logits`;
- critic: `state → 64 → 64 → scalar value`.

Validation:

- Log approximate KL divergence for every update.
- Compare actual versus predicted surrogate improvement.
- Vary PPO clipping from `0.1` to `0.3`.
- Demonstrate that uncontrolled policy-gradient steps can collapse performance.

### Step 7 — DDPG, TD3, and SAC

Read:

1. [Deterministic Policy Gradient Algorithms — Silver et al.](https://proceedings.mlr.press/v32/silver14.html): deterministic policy-gradient theorem.
2. [Continuous Control With Deep Reinforcement Learning — Lillicrap et al.](https://arxiv.org/abs/1509.02971): DDPG.
3. [Addressing Function Approximation Error in Actor-Critic Methods — Fujimoto et al.](https://arxiv.org/abs/1802.09477): TD3’s three stabilizing changes.
4. [Soft Q-Learning — Haarnoja et al.](https://arxiv.org/abs/1702.08165): entropy-regularized value learning.
5. [Soft Actor-Critic — Haarnoja et al.](https://proceedings.mlr.press/v80/haarnoja18b.html): original SAC derivation.
6. [Soft Actor-Critic Algorithms and Applications — Haarnoja et al.](https://arxiv.org/abs/1812.05905): practical SAC formulation.

Implement in this order:

1. DDPG on Pendulum.
2. TD3 by making only the paper’s three changes: clipped double Q-learning, delayed policy updates, and target-policy smoothing.
3. Practical SAC with a tanh-Gaussian actor, twin critics, target critics, and automatic temperature tuning.

Use this Pendulum task:

- State: angle and angular velocity.
- Action: one clipped continuous torque.
- Reward: `-(angle² + 0.1 * angular_velocity² + 0.001 * torque²)`.
- Fixed episode horizon.

Use LQR as a second task with known dynamics and a known approximate optimum.

Validation:

- Test action scaling and tanh log-probability correction independently.
- Verify no target-network gradients.
- Compare critic overestimation across DDPG, TD3, and SAC.
- Compare deterministic and stochastic exploration.

### Step 8 — Choose one modern specialization

Choose one branch. Do not implement both during the first pass.

#### Branch A: offline RL

Read:

1. [Off-Policy Deep Reinforcement Learning without Exploration — BCQ](https://arxiv.org/abs/1812.02900): offline distribution shift problem.
2. [Conservative Q-Learning for Offline Reinforcement Learning — CQL](https://arxiv.org/abs/2006.04779): conservative value regularization.
3. [Offline Reinforcement Learning with Implicit Q-Learning — IQL](https://arxiv.org/abs/2110.06169): expectile regression and advantage-weighted behavior cloning.

Implement:

1. Behavior cloning.
2. IQL first, or CQL if the target role emphasizes value-based offline RL.

Task definition:

- Use your point-mass navigation environment.
- Collect 10,000–100,000 transitions from random, mediocre, and expert policies.
- Train only from the saved dataset.
- Hold out environment seeds for evaluation.

Validation:

- Compare behavior cloning with IQL/CQL.
- Evaluate datasets with good and poor action coverage.
- Log the learned policy’s action distribution relative to the dataset.

#### Branch B: goal-conditioned robotics-style RL

Read:

1. [Universal Value Function Approximators — Schaul et al.](https://arxiv.org/abs/1506.04416): conditioning on goals.
2. [Hindsight Experience Replay — Andrychowicz et al.](https://arxiv.org/abs/1707.01495): relabeling failed episodes.

Implement:

1. Goal-conditioned critic taking `[state, action, goal]`.
2. HER replay relabeling.
3. TD3+HER or SAC+HER.

Task definition:

- Two-dimensional point mass with position and velocity.
- Random target position in a square.
- Dense version: negative distance reward.
- Sparse version: `0` on success and `-1` otherwise.
- Success when within a fixed radius of the target.

Validation:

- Compare ordinary replay against HER.
- Test held-out target positions.
- Confirm relabeled transitions have internally consistent goals and rewards.

### Step 9 — Job-prep capstone

Build one clean experiment, not several incomplete projects.

Compare the algorithms you actually implemented on tasks they share:

- PPO on CartPole-like and Pendulum;
- TD3 and SAC on Pendulum and point-mass navigation;
- IQL/CQL on offline point-mass data, or HER on sparse point-mass navigation.

Deliver:

- three to five seeds;
- environment-step-based learning curves;
- one ablation for each main algorithm;
- failure analysis;
- a short technical report;
- a README explaining each update equation and design choice.

Completion criterion:

- You can implement a simplified DQN, PPO, or SAC from a blank file;
- explain on-policy/off-policy tradeoffs;
- explain the main instability sources;
- identify bugs from learning curves and logged quantities;
- discuss why your evaluation is or is not statistically convincing.

## Secondary implementation queue

Implement these after the core only when they are relevant to a role, a project, or a specific knowledge gap.

### Classical extensions

Read and optionally implement:

- [Dyna, an Integrated Architecture for Learning, Planning, and Reacting — Sutton](https://doi.org/10.1145/122344.122377)
- Expected SARSA;
- n-step TD and n-step SARSA;
- eligibility traces and TD(λ);
- prioritized sweeping;
- [Dyna-style Planning with Linear Function Approximation and Prioritized Sweeping — Sutton et al.](https://proceedings.mlr.press/r6/sutton08a.html)

### Actor-critic extensions

- Exact asynchronous A3C with multiprocessing;
- PPO-Penalty;
- separate natural-policy-gradient experiments if you want a deeper trust-region implementation.

### DQN extensions

Read and implement individually before attempting Rainbow:

- [Dueling Network Architectures for Deep Reinforcement Learning — Wang et al.](https://arxiv.org/abs/1511.06581)
- [Noisy Networks for Exploration — Fortunato et al.](https://arxiv.org/abs/1706.10295)
- [A Distributional Perspective on Reinforcement Learning — Bellemare et al.](https://arxiv.org/abs/1707.06887)
- [Distributional Reinforcement Learning with Quantile Regression — Dabney et al.](https://arxiv.org/abs/1710.10044)
- [Rainbow — Hessel et al.](https://arxiv.org/abs/1710.02298)

Rainbow is an integration project combining:

- Double Q-learning;
- dueling architecture;
- prioritized replay;
- multi-step returns;
- distributional value prediction;
- noisy networks.

### Exploration, representation, and skills

Read and implement selectively:

1. [Unifying Count-Based Exploration and Intrinsic Motivation — Bellemare et al.](https://arxiv.org/abs/1606.01868)
2. [Curiosity-Driven Exploration by Self-Supervised Prediction — Pathak et al.](https://arxiv.org/abs/1705.05363)
3. [Large-Scale Study of Curiosity-Driven Learning — Burda et al.](https://arxiv.org/abs/1810.04355)
4. [Exploration by Random Network Distillation — Burda et al.](https://arxiv.org/abs/1810.12894)
5. [Diversity Is All You Need — Eysenbach et al.](https://arxiv.org/abs/1802.06070)

Recommended implementation order:

```text
tabular count bonus
  → RND or ICM
  → UVFA
  → HER
  → DIAYN
```

### Model-based RL

Read and implement selectively:

1. [Deep Reinforcement Learning in a Handful of Trials Using Probabilistic Dynamics Models — PETS](https://arxiv.org/abs/1805.12114)
2. [When to Trust Your Model: Model-Based Policy Optimization — MBPO](https://arxiv.org/abs/1906.08291)
3. [Recurrent World Models Facilitate Policy Evolution — World Models](https://arxiv.org/abs/1803.10122)
4. [Learning and Acting with a Spatial Memory — PlaNet](https://arxiv.org/abs/1811.04551)
5. [Dream to Control — Dreamer](https://arxiv.org/abs/1912.01603)
6. [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model — MuZero](https://arxiv.org/abs/1911.08265)

Recommended order:

```text
Dyna-Q
  → learned dynamics model
  → random-shooting MPC
  → PETS
  → MBPO
  → Dreamer or MuZero
```

PETS is the most suitable first serious model-based project for a laptop. Dreamer and MuZero are capstones.

### Partial observability and memory

Read and implement selectively:

1. [Deep Recurrent Q-Learning for Partially Observable MDPs — Hausknecht and Stone](https://arxiv.org/abs/1507.06527)
2. [Recurrent Experience Replay in Distributed Reinforcement Learning — R2D2](https://openreview.net/forum?id=r1lyTjAqYX)

Implement DRQN or recurrent PPO on:

- aliased Gridworld;
- delayed-reward navigation;
- flickering CartPole;
- tasks where the correct action depends on observation history.

When implementing the recurrent agent, also implement:

- sequence replay;
- burn-in steps;
- hidden-state resets at episode boundaries;
- padding and loss masks for variable-length sequences.

R2D2 is primarily a distributed replay-system project.

### Imitation learning and offline extensions

Read and implement selectively:

1. [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning — DAgger](https://www.cs.cmu.edu/~sross1/publications/Ross-AIStats10-noSpelling.pdf)
2. [Generative Adversarial Imitation Learning — Ho and Ermon](https://arxiv.org/abs/1606.03476)
3. [Off-Policy Deep Reinforcement Learning without Exploration — BCQ](https://arxiv.org/abs/1812.02900)
4. [Conservative Q-Learning for Offline Reinforcement Learning — CQL](https://arxiv.org/abs/2006.04779)
5. [Offline Reinforcement Learning with Implicit Q-Learning — IQL](https://arxiv.org/abs/2110.06169)
6. [Decision Transformer](https://arxiv.org/abs/2106.01345)

Recommended order:

```text
behavior cloning
  → DAgger
  → GAIL
  → BCQ
  → CQL or IQL
  → Decision Transformer
```

## Survey-only material

Read the abstract, introduction, algorithm overview, and conclusion. Do not implement unless a job or project directly requires it.

### Policy-gradient variants

- ACKTR;
- ACER;
- Q-Prop;
- Stein control variates;
- PCL;
- Trust-PCL;
- PGQL;
- Reactor;
- IPG;
- PPO-Penalty if you already understand PPO-Clip.

### Exploration variants

- VIME;
- EX2;
- PixelCNN pseudocounts;
- VIC;
- VALOR.

### Transfer and multitask learning

- Progressive Networks;
- PathNet;
- MATL;
- IU Agent;
- most older transfer methods.

### Memory architectures

- MFEC;
- NEC;
- Neural Map;
- MERLIN;
- Relational Recurrent Networks.

### Distributed RL and platforms

- [IMPALA](https://arxiv.org/abs/1802.01561);
- [Distributed Prioritized Experience Replay — Ape-X](https://arxiv.org/abs/1803.00933);
- [R2D2](https://openreview.net/forum?id=r1lyTjAqYX);
- RLlib;
- most platform papers.

### Real-world and robotics papers

Read these for experimental design and application context, not as first implementation targets:

- most robotics benchmark papers;
- applied manipulation papers;
- platform and systems papers.

## Optional branches

Choose one or two only if they match your target roles.

### Hierarchical RL

1. [Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction — Sutton et al.](https://www.sciencedirect.com/science/article/pii/S0004370298000401)
2. [The Option-Critic Architecture](https://arxiv.org/abs/1609.05140)
3. [Data-Efficient Hierarchical Reinforcement Learning — HIRO](https://arxiv.org/abs/1805.08296)

Implement Option-Critic before HIRO.

### Meta-RL

1. [RL²: Fast Reinforcement Learning via Slow Reinforcement Learning](https://arxiv.org/abs/1611.02779)
2. [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400)
3. [A Simple Neural Attentive Meta-Learner — SNAIL](https://arxiv.org/abs/1703.10622)

MAML is easier to isolate; RL² is more directly meta-RL.

### Multi-agent RL

1. [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments — MADDPG](https://arxiv.org/abs/1706.02275)
2. [QMIX](https://arxiv.org/abs/1803.11485)
3. [The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games — MAPPO](https://arxiv.org/abs/2103.01955)

Use matrix games, predator-prey, and small cooperative navigation tasks.

### Self-play and planning

1. [Mastering the Game of Go Without Human Knowledge — AlphaZero](https://arxiv.org/abs/1712.01815)
2. [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model — MuZero](https://arxiv.org/abs/1911.08265)

Implement AlphaZero on Tic-Tac-Toe or Connect Four before considering MuZero.

### Safety and constraints

Read:

1. [Concrete Problems in AI Safety — Amodei et al.](https://arxiv.org/abs/1606.06565)
2. [Constrained Policy Optimization — Achiam et al.](https://arxiv.org/abs/1705.10528)
3. [Safe Exploration in Continuous Action Spaces — Dalal et al.](https://arxiv.org/abs/1805.09461)
4. [Deep Reinforcement Learning from Human Preferences — Christiano et al.](https://arxiv.org/abs/1706.03741)

Implement only if the target role emphasizes safety, constraints, or preference learning.

## Environment suite

Write these environments yourself using Python and PyTorch only.

### Discrete environments

- Multi-armed bandits;
- Random Walk;
- two-state MDPs;
- Chain MDP;
- Cliff Walking;
- Windy Gridworld;
- key-and-door maze;
- stochastic Gridworld;
- aliased/POMDP Gridworld;
- Tic-Tac-Toe;
- small Connect Four.

### Continuous environments

- linear dynamical system;
- linear-quadratic regulator;
- Pendulum;
- Mountain Car Continuous;
- two-dimensional point-mass navigation;
- sparse-reward point-mass navigation;
- point-mass navigation with obstacles.

For the core curriculum, prioritize:

1. Random Walk and Chain MDP;
2. Cliff Walking and Gridworld;
3. CartPole-like environment;
4. Pendulum;
5. LQR;
6. point-mass navigation;
7. one sparse-reward task.

## Job-prep competencies to demonstrate

By the end of the core curriculum, you should be able to explain and implement:

- Bellman expectation and optimality equations;
- Monte Carlo versus TD learning;
- SARSA versus Q-learning;
- on-policy versus off-policy learning;
- replay buffers and target networks;
- Double Q-learning and overestimation;
- policy gradients and the log-derivative trick;
- actor-critic methods;
- GAE;
- PPO clipping and trust regions;
- deterministic versus stochastic policies;
- DDPG, TD3, and SAC tradeoffs;
- entropy regularization;
- exploration versus exploitation;
- offline distribution shift;
- evaluation variance and random seeds;
- common sources of RL instability.

You should also be able to debug:

- incorrect terminal masks;
- bootstrapping through terminal states;
- stale target networks;
- incorrect action scaling;
- accidental gradients through targets;
- incorrect log-probability calculations;
- replay-buffer sampling bugs;
- reward normalization mistakes;
- time-limit handling;
- high-variance evaluation results.

## Timeline at 4–5 hours per day

### Lean core

The core is now organized as ten executable steps. At your stated pace, target **8–12 weeks** for the core, with the understanding that difficult debugging may add time.

Suggested schedule:

| Step | Work | Target |
|---|---|---:|
| 0 | Experimental foundation and Random Walk | 2–3 days |
| 1 | Monte Carlo, TD(0), SARSA, and Q-learning | 4–6 days |
| 2 | REINFORCE and policy gradients | 4–6 days |
| 3 | DQN | 5–7 days |
| 4 | Double DQN and prioritized replay | 3–5 days |
| 5 | A2C and GAE | 4–6 days |
| 6 | Natural policy gradient, TRPO, and PPO | 6–9 days |
| 7 | DDPG, TD3, and SAC | 7–10 days |
| 8 | Offline RL or HER | 5–8 days |
| 9 | Capstone and technical write-up | 7–12 days |

For each step, stop when the validation checks pass. Do not wait for a paper-level score before moving on; record failures and return later if the failure teaches you something useful.

### Core plus a polished capstone

Budget **14–18 weeks**, or approximately **400–600 hours**.

### Selected secondary material

Add approximately:

- 3–7 days for a small algorithmic variant;
- 1–2 weeks for a substantial algorithm;
- 2–4 weeks for PETS, MBPO, recurrent agents, or a serious offline-RL implementation;
- 3–6 weeks for Dreamer, MuZero, or AlphaZero.

### Literally doing everything

Implementing every secondary and optional branch would still be approximately **800–1,400 additional hours**. At your schedule, that is roughly **7–12 months beyond the core**, and is not the best use of job-prep time.

## Capstone options

### Capstone 1: Sample-efficient continuous control

Compare:

- PPO;
- TD3;
- SAC;
- PETS, if implemented;
- MBPO, if implemented.

Use Pendulum, LQR, and point-mass navigation.

### Capstone 2: Sparse-reward navigation

Compare:

- PPO;
- SAC;
- HER;
- RND;
- DIAYN, optionally.

Use procedurally generated mazes with held-out layouts.

### Capstone 3: Learning under imperfect information

Compare:

- DQN;
- DRQN;
- PPO;
- recurrent PPO;
- behavior cloning or IQL.

Use aliased Gridworlds, delayed rewards, and datasets with varying quality.

## Mac-friendly experiment budgets

For initial debugging:

- 10,000–50,000 environment steps;
- one seed;
- small networks, such as two 64-unit hidden layers.

For meaningful comparisons:

- 100,000–1,000,000 environment steps;
- three to five seeds;
- report mean, standard deviation, and individual runs.

Track environment interaction steps, not merely gradient updates. Save configurations, seeds, metrics, and checkpoints with the standard library.

The immediate objective is not to complete every paper. It is to become capable of implementing, explaining, debugging, and evaluating the algorithms most relevant to RL jobs. The rest of this document is a deliberate backlog rather than a required syllabus.
