# EX_05 - MONTE CARLO CONTROL ALGORITHM

## AIM

To implement the Monte Carlo Control algorithm on the FrozenLake environment using the OpenAI Gym framework in order to find an optimal policy that maximizes the expected return from any given state.

## PROBLEM STATEMENT

In reinforcement learning, an agent interacts with an environment to learn a policy that maximizes cumulative rewards over time. The FrozenLake environment is a classic problem where the agent must navigate a frozen grid to reach a goal while avoiding holes. The environment is stochastic (slippery), making it difficult to predict exact outcomes of actions.

The goal of this experiment is to use the Monte Carlo Control algorithm to estimate the action-value function (Q-values), derive the optimal policy based on those values, and evaluate the policy by calculating the probability of reaching the goal and the average return.

## MONTE CARLO CONTROL ALGORITHM

![image](https://github.com/user-attachments/assets/ef734cd4-8a96-4cb4-a462-194b41623770

![image](https://github.com/user-attachments/assets/32d2552d-3607-4d27-8e7f-905e039a8283)

## MONTE CARLO CONTROL FUNCTION
```
DEVELOPED BY : NIRAUNJANA GAYATHRI G R
REGISTER NO. : 212222230096
```
```
def mc_control(env, gamma=1.0,
               init_alpha=0.5, min_alpha=0.01, alpha_decay_ratio=0.5,
               init_epsilon=1.0, min_epsilon=0.1, epsilon_decay_ratio=0.9,
               n_episodes=3000, max_steps=200, first_visit=True):
    
    nS, nA = env.observation_space.n, env.action_space.n
    Q = np.zeros((nS, nA))
    returns_count = np.zeros((nS, nA))
    
    def select_action(Qs): return np.argmax(Qs)

    alpha_schedule = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    epsilon_schedule = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)
    
    for i in range(n_episodes):
        epsilon, alpha = epsilon_schedule[i], alpha_schedule[i]
        trajectory = generate_trajectory(select_action, Q, epsilon, env, max_steps)
        G, visited = 0.0, set()
        for t in reversed(range(len(trajectory))):
            s, a, r = trajectory[t]
            G = gamma * G + r
            if first_visit and (s, a) in visited:
                continue
            returns_count[s, a] += 1
            Q[s, a] += alpha * (G - Q[s, a])
            visited.add((s, a))

    V = np.max(Q, axis=1)
    pi = lambda s: np.argmax(Q[s])
    return Q, V, pi
```

## OUTPUT:
```
DEVELOPED BY : NIRAUNJANA GAYATHRI G R
REGISTER NO. : 212222230096
```
### Mention the Action value function, optimal value function, optimal policy, and success rate for the optimal policy.

![image](https://github.com/user-attachments/assets/e8a2d0b8-946c-4f27-8616-97b3f9d51106)

![image](https://github.com/user-attachments/assets/5a357b80-fb3b-4a3d-85a5-fe5d68878e04)


## RESULT:

Thus to implement the Monte Carlo Control algorithm on the FrozenLake environment using the OpenAI Gym framework in order to find an optimal policy that maximizes the expected return from any given state is successfully implemented.
