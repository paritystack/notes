# Reinforcement Learning

Reinforcement Learning (RL) is about learning to make decisions by interacting with an environment to maximize cumulative reward.

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Markov Decision Processes](#markov-decision-processes)
3. [Dynamic Programming](#dynamic-programming)
4. [Monte Carlo Methods](#monte-carlo-methods)
5. [Temporal Difference Learning](#temporal-difference-learning)
6. [Q-Learning](#q-learning)
7. [SARSA](#sarsa)
8. [Policy Gradient Methods](#policy-gradient-methods)
9. [Actor-Critic Methods](#actor-critic-methods)
10. [Multi-Armed Bandits](#multi-armed-bandits)

## Core Concepts

### The RL Framework

**Key Components:**
- **Agent**: The learner/decision maker
- **Environment**: What the agent interacts with
- **State (s)**: Current situation
- **Action (a)**: What the agent can do
- **Reward (r)**: Feedback signal
- **Policy (π)**: Strategy for selecting actions
- **Value Function (V)**: Expected future reward from a state
- **Q-Function (Q)**: Expected future reward for state-action pairs

**Mathematical Framework:**
```
At each time step t:
- Agent observes state s_t
- Agent takes action a_t
- Environment transitions to s_{t+1}
- Agent receives reward r_{t+1}
```

```python
import numpy as np
import matplotlib.pyplot as plt

# Simple RL environment example
class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.state = (0, 0)
        self.goal = (size-1, size-1)
        
    def reset(self):
        self.state = (0, 0)
        return self.state
    
    def step(self, action):
        # Actions: 0=up, 1=right, 2=down, 3=left
        x, y = self.state
        
        if action == 0:  # up
            x = max(0, x - 1)
        elif action == 1:  # right
            y = min(self.size - 1, y + 1)
        elif action == 2:  # down
            x = min(self.size - 1, x + 1)
        elif action == 3:  # left
            y = max(0, y - 1)
        
        self.state = (x, y)
        
        # Reward
        if self.state == self.goal:
            reward = 1.0
            done = True
        else:
            reward = -0.01  # Small penalty for each step
            done = False
        
        return self.state, reward, done
    
    def render(self):
        grid = np.zeros((self.size, self.size))
        grid[self.state] = 1
        grid[self.goal] = 0.5
        plt.imshow(grid, cmap='hot')
        plt.title(f'State: {self.state}')
        plt.show()

# Example usage
env = GridWorld(size=5)
state = env.reset()
print(f"Initial state: {state}")

# Take random actions
for _ in range(5):
    action = np.random.randint(0, 4)
    state, reward, done = env.step(action)
    print(f"State: {state}, Reward: {reward}, Done: {done}")
    if done:
        break
```

### Return and Discounting

**Return (G_t)**: Total cumulative reward from time t
```
G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ... = Σ_{k=0}^∞ γ^k R_{t+k+1}
```

Where γ (gamma) is the discount factor (0 ≤ γ ≤ 1):
- γ = 0: Only immediate rewards matter
- γ = 1: All future rewards equally important
- γ closer to 1: More far-sighted agent

```python
def calculate_return(rewards, gamma=0.99):
    """Calculate discounted return from a list of rewards"""
    G = 0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns

# Example
rewards = [1, 0, 0, 1, 0]
returns = calculate_return(rewards, gamma=0.9)
print(f"Rewards: {rewards}")
print(f"Returns: {returns}")
```

## Markov Decision Processes

An MDP is defined by a tuple (S, A, P, R, γ):
- **S**: Set of states
- **A**: Set of actions
- **P**: Transition probability P(s'|s,a)
- **R**: Reward function R(s,a,s')
- **γ**: Discount factor

**Markov Property**: Future depends only on current state, not history
```
P(s_{t+1}|s_t, a_t, s_{t-1}, a_{t-1}, ...) = P(s_{t+1}|s_t, a_t)
```

```python
class MDP:
    def __init__(self, states, actions, transitions, rewards, gamma=0.99):
        self.states = states
        self.actions = actions
        self.transitions = transitions  # P(s'|s,a)
        self.rewards = rewards          # R(s,a,s')
        self.gamma = gamma
    
    def get_transition_prob(self, state, action, next_state):
        return self.transitions.get((state, action, next_state), 0.0)
    
    def get_reward(self, state, action, next_state):
        return self.rewards.get((state, action, next_state), 0.0)

# Example: Simple MDP
states = ['s0', 's1', 's2']
actions = ['a0', 'a1']

transitions = {
    ('s0', 'a0', 's1'): 0.8,
    ('s0', 'a0', 's0'): 0.2,
    ('s0', 'a1', 's2'): 0.9,
    ('s0', 'a1', 's0'): 0.1,
    ('s1', 'a0', 's2'): 1.0,
    ('s2', 'a0', 's2'): 1.0,
}

rewards = {
    ('s0', 'a0', 's1'): -1,
    ('s0', 'a1', 's2'): 10,
    ('s1', 'a0', 's2'): 5,
}

mdp = MDP(states, actions, transitions, rewards)
```

### Value Functions

**State-Value Function V^π(s):**
```
V^π(s) = E_π[G_t | S_t = s]
       = E_π[Σ_{k=0}^∞ γ^k R_{t+k+1} | S_t = s]
```

**Action-Value Function Q^π(s,a):**
```
Q^π(s,a) = E_π[G_t | S_t = s, A_t = a]
```

**Bellman Equations:**
```
V^π(s) = Σ_a π(a|s) Σ_{s',r} p(s',r|s,a)[r + γV^π(s')]
Q^π(s,a) = Σ_{s',r} p(s',r|s,a)[r + γΣ_{a'} π(a'|s')Q^π(s',a')]
```

**Optimal Value Functions:**
```
V*(s) = max_π V^π(s) = max_a Q*(s,a)
Q*(s,a) = E[R_{t+1} + γV*(S_{t+1}) | S_t=s, A_t=a]
```

## Dynamic Programming

DP methods assume full knowledge of the MDP.

### Policy Evaluation

Compute value function for a given policy.

```python
def policy_evaluation(policy, mdp, theta=1e-6):
    """
    Evaluate a policy using iterative policy evaluation
    
    Args:
        policy: dict mapping states to action probabilities
        mdp: MDP object
        theta: convergence threshold
    """
    V = {s: 0 for s in mdp.states}
    
    while True:
        delta = 0
        for s in mdp.states:
            v = V[s]
            new_v = 0
            
            # Sum over actions
            for a in mdp.actions:
                action_prob = policy.get((s, a), 0)
                
                # Sum over next states
                for s_prime in mdp.states:
                    trans_prob = mdp.get_transition_prob(s, a, s_prime)
                    reward = mdp.get_reward(s, a, s_prime)
                    new_v += action_prob * trans_prob * (reward + mdp.gamma * V[s_prime])
            
            V[s] = new_v
            delta = max(delta, abs(v - V[s]))
        
        if delta < theta:
            break
    
    return V

# Example: Uniform random policy
random_policy = {
    ('s0', 'a0'): 0.5,
    ('s0', 'a1'): 0.5,
    ('s1', 'a0'): 1.0,
    ('s2', 'a0'): 1.0,
}

V = policy_evaluation(random_policy, mdp)
print("State values:", V)
```

### Policy Iteration

```python
def policy_iteration(mdp, theta=1e-6):
    """
    Find optimal policy using policy iteration
    """
    # Initialize random policy
    policy = {}
    for s in mdp.states:
        action = np.random.choice(mdp.actions)
        for a in mdp.actions:
            policy[(s, a)] = 1.0 if a == action else 0.0
    
    while True:
        # Policy Evaluation
        V = policy_evaluation(policy, mdp, theta)
        
        # Policy Improvement
        policy_stable = True
        
        for s in mdp.states:
            old_action = None
            for a in mdp.actions:
                if policy.get((s, a), 0) == 1.0:
                    old_action = a
                    break
            
            # Find best action
            action_values = {}
            for a in mdp.actions:
                q = 0
                for s_prime in mdp.states:
                    trans_prob = mdp.get_transition_prob(s, a, s_prime)
                    reward = mdp.get_reward(s, a, s_prime)
                    q += trans_prob * (reward + mdp.gamma * V[s_prime])
                action_values[a] = q
            
            best_action = max(action_values, key=action_values.get)
            
            # Update policy
            for a in mdp.actions:
                policy[(s, a)] = 1.0 if a == best_action else 0.0
            
            if best_action != old_action:
                policy_stable = False
        
        if policy_stable:
            break
    
    return policy, V

optimal_policy, optimal_V = policy_iteration(mdp)
print("Optimal policy:", optimal_policy)
print("Optimal values:", optimal_V)
```

### Value Iteration

```python
def value_iteration(mdp, theta=1e-6):
    """
    Find optimal policy using value iteration
    """
    V = {s: 0 for s in mdp.states}
    
    while True:
        delta = 0
        for s in mdp.states:
            v = V[s]
            
            # Find max over actions
            action_values = []
            for a in mdp.actions:
                q = 0
                for s_prime in mdp.states:
                    trans_prob = mdp.get_transition_prob(s, a, s_prime)
                    reward = mdp.get_reward(s, a, s_prime)
                    q += trans_prob * (reward + mdp.gamma * V[s_prime])
                action_values.append(q)
            
            V[s] = max(action_values) if action_values else 0
            delta = max(delta, abs(v - V[s]))
        
        if delta < theta:
            break
    
    # Extract policy
    policy = {}
    for s in mdp.states:
        action_values = {}
        for a in mdp.actions:
            q = 0
            for s_prime in mdp.states:
                trans_prob = mdp.get_transition_prob(s, a, s_prime)
                reward = mdp.get_reward(s, a, s_prime)
                q += trans_prob * (reward + mdp.gamma * V[s_prime])
            action_values[a] = q
        
        best_action = max(action_values, key=action_values.get)
        for a in mdp.actions:
            policy[(s, a)] = 1.0 if a == best_action else 0.0
    
    return policy, V

optimal_policy, optimal_V = value_iteration(mdp)
```

## Monte Carlo Methods

MC methods learn from complete episodes without needing environment model.

### First-Visit MC Prediction

```python
def first_visit_mc_prediction(env, policy, num_episodes=1000, gamma=0.99):
    """
    Estimate state-value function using first-visit MC
    """
    returns = {s: [] for s in env.states}
    V = {s: 0 for s in env.states}
    
    for episode in range(num_episodes):
        # Generate episode
        episode_data = []
        state = env.reset()
        done = False
        
        while not done:
            action = policy[state]
            next_state, reward, done = env.step(action)
            episode_data.append((state, action, reward))
            state = next_state
        
        # Calculate returns
        G = 0
        visited_states = set()
        
        for t in reversed(range(len(episode_data))):
            state, action, reward = episode_data[t]
            G = reward + gamma * G
            
            # First-visit: only update if state not seen earlier
            if state not in visited_states:
                returns[state].append(G)
                V[state] = np.mean(returns[state])
                visited_states.add(state)
    
    return V

# Example usage with GridWorld
env = GridWorld(size=4)
# Define a simple policy
policy = {state: np.random.randint(0, 4) for state in 
          [(i, j) for i in range(4) for j in range(4)]}

V = first_visit_mc_prediction(env, policy, num_episodes=10000)
```

### Monte Carlo Control (Epsilon-Greedy)

```python
def mc_control_epsilon_greedy(env, num_episodes=10000, gamma=0.99, epsilon=0.1):
    """
    Monte Carlo control with epsilon-greedy policy
    """
    Q = {}
    returns = {}
    
    # Initialize Q-values
    for state in env.get_all_states():
        for action in range(env.num_actions):
            Q[(state, action)] = 0
            returns[(state, action)] = []
    
    for episode in range(num_episodes):
        # Generate episode with epsilon-greedy policy
        episode_data = []
        state = env.reset()
        done = False
        
        while not done:
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                action = np.random.randint(0, env.num_actions)
            else:
                q_values = [Q.get((state, a), 0) for a in range(env.num_actions)]
                action = np.argmax(q_values)
            
            next_state, reward, done = env.step(action)
            episode_data.append((state, action, reward))
            state = next_state
        
        # Update Q-values
        G = 0
        visited = set()
        
        for t in reversed(range(len(episode_data))):
            state, action, reward = episode_data[t]
            G = reward + gamma * G
            
            if (state, action) not in visited:
                returns[(state, action)].append(G)
                Q[(state, action)] = np.mean(returns[(state, action)])
                visited.add((state, action))
    
    # Extract policy
    policy = {}
    for state in env.get_all_states():
        q_values = [Q.get((state, a), 0) for a in range(env.num_actions)]
        policy[state] = np.argmax(q_values)
    
    return policy, Q
```

## Temporal Difference Learning

TD methods learn from incomplete episodes by bootstrapping.

### TD(0) Prediction

**TD Update Rule:**
```
V(S_t) ← V(S_t) + α[R_{t+1} + γV(S_{t+1}) - V(S_t)]
```

Where:
- α is the learning rate
- R_{t+1} + γV(S_{t+1}) is the TD target
- δ_t = R_{t+1} + γV(S_{t+1}) - V(S_t) is the TD error

```python
def td_0_prediction(env, policy, num_episodes=1000, alpha=0.1, gamma=0.99):
    """
    TD(0) prediction for estimating state values
    """
    V = {s: 0 for s in env.get_all_states()}
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            action = policy[state]
            next_state, reward, done = env.step(action)
            
            # TD update
            if not done:
                td_target = reward + gamma * V[next_state]
            else:
                td_target = reward
            
            td_error = td_target - V[state]
            V[state] += alpha * td_error
            
            state = next_state
    
    return V
```

### TD(λ) - Eligibility Traces

```python
def td_lambda_prediction(env, policy, num_episodes=1000, 
                        alpha=0.1, gamma=0.99, lambda_=0.9):
    """
    TD(λ) prediction with eligibility traces
    """
    V = {s: 0 for s in env.get_all_states()}
    
    for episode in range(num_episodes):
        E = {s: 0 for s in env.get_all_states()}  # Eligibility traces
        state = env.reset()
        done = False
        
        while not done:
            action = policy[state]
            next_state, reward, done = env.step(action)
            
            # Calculate TD error
            if not done:
                td_error = reward + gamma * V[next_state] - V[state]
            else:
                td_error = reward - V[state]
            
            # Update eligibility trace for current state
            E[state] += 1
            
            # Update all states
            for s in env.get_all_states():
                V[s] += alpha * td_error * E[s]
                E[s] *= gamma * lambda_
            
            state = next_state
    
    return V
```

## Q-Learning

Q-Learning is an off-policy TD control algorithm.

**Q-Learning Update:**
```
Q(S_t, A_t) ← Q(S_t, A_t) + α[R_{t+1} + γ max_a Q(S_{t+1}, a) - Q(S_t, A_t)]
```

```python
class QLearningAgent:
    def __init__(self, state_space, action_space, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.state_space = state_space
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Initialize Q-table
        self.q_table = {}
        for state in state_space:
            for action in action_space:
                self.q_table[(state, action)] = 0.0
    
    def get_action(self, state, training=True):
        """Epsilon-greedy action selection"""
        if training and np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)
        else:
            q_values = [self.q_table.get((state, a), 0) for a in self.action_space]
            return self.action_space[np.argmax(q_values)]
    
    def update(self, state, action, reward, next_state, done):
        """Q-learning update"""
        # Current Q-value
        current_q = self.q_table[(state, action)]
        
        # Maximum Q-value for next state
        if not done:
            max_next_q = max([self.q_table.get((next_state, a), 0) 
                            for a in self.action_space])
        else:
            max_next_q = 0
        
        # Q-learning update
        td_target = reward + self.gamma * max_next_q
        td_error = td_target - current_q
        self.q_table[(state, action)] += self.alpha * td_error
        
        return td_error
    
    def train(self, env, num_episodes=1000):
        """Train the agent"""
        episode_rewards = []
        
        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = self.get_action(state, training=True)
                next_state, reward, done = env.step(action)
                
                self.update(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
            
            episode_rewards.append(total_reward)
            
            # Decay epsilon
            self.epsilon = max(0.01, self.epsilon * 0.995)
            
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.3f}")
        
        return episode_rewards

# Example usage
env = GridWorld(size=5)
state_space = [(i, j) for i in range(5) for j in range(5)]
action_space = [0, 1, 2, 3]  # up, right, down, left

agent = QLearningAgent(state_space, action_space)
rewards = agent.train(env, num_episodes=5000)

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.plot(np.convolve(rewards, np.ones(100)/100, mode='valid'))
plt.xlabel('Episode')
plt.ylabel('Average Reward (100 episodes)')
plt.title('Q-Learning Training Progress')
plt.grid(True)
plt.show()
```

### Double Q-Learning

Reduces maximization bias in Q-learning.

```python
class DoubleQLearningAgent:
    def __init__(self, state_space, action_space, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.state_space = state_space
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Two Q-tables
        self.q_table_1 = {(s, a): 0.0 for s in state_space for a in action_space}
        self.q_table_2 = {(s, a): 0.0 for s in state_space for a in action_space}
    
    def get_action(self, state, training=True):
        """Epsilon-greedy using average of both Q-tables"""
        if training and np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)
        else:
            q_values = [(self.q_table_1[(state, a)] + self.q_table_2[(state, a)]) / 2 
                       for a in self.action_space]
            return self.action_space[np.argmax(q_values)]
    
    def update(self, state, action, reward, next_state, done):
        """Double Q-learning update"""
        # Randomly choose which Q-table to update
        if np.random.random() < 0.5:
            q_table_update = self.q_table_1
            q_table_target = self.q_table_2
        else:
            q_table_update = self.q_table_2
            q_table_target = self.q_table_1
        
        current_q = q_table_update[(state, action)]
        
        if not done:
            # Use one Q-table to select action, other to evaluate
            best_action = max(self.action_space, 
                            key=lambda a: q_table_update[(next_state, a)])
            max_next_q = q_table_target[(next_state, best_action)]
        else:
            max_next_q = 0
        
        td_target = reward + self.gamma * max_next_q
        td_error = td_target - current_q
        q_table_update[(state, action)] += self.alpha * td_error
        
        return td_error
```

## SARSA

SARSA is an on-policy TD control algorithm.

**SARSA Update:**
```
Q(S_t, A_t) ← Q(S_t, A_t) + α[R_{t+1} + γQ(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]
```

```python
class SARSAAgent:
    def __init__(self, state_space, action_space, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.state_space = state_space
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Initialize Q-table
        self.q_table = {(s, a): 0.0 for s in state_space for a in action_space}
    
    def get_action(self, state, training=True):
        """Epsilon-greedy action selection"""
        if training and np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)
        else:
            q_values = [self.q_table[(state, a)] for a in self.action_space]
            return self.action_space[np.argmax(q_values)]
    
    def update(self, state, action, reward, next_state, next_action, done):
        """SARSA update"""
        current_q = self.q_table[(state, action)]
        
        if not done:
            next_q = self.q_table[(next_state, next_action)]
        else:
            next_q = 0
        
        td_target = reward + self.gamma * next_q
        td_error = td_target - current_q
        self.q_table[(state, action)] += self.alpha * td_error
        
        return td_error
    
    def train(self, env, num_episodes=1000):
        """Train the agent"""
        episode_rewards = []
        
        for episode in range(num_episodes):
            state = env.reset()
            action = self.get_action(state, training=True)
            total_reward = 0
            done = False
            
            while not done:
                next_state, reward, done = env.step(action)
                next_action = self.get_action(next_state, training=True)
                
                self.update(state, action, reward, next_state, next_action, done)
                
                state = next_state
                action = next_action
                total_reward += reward
            
            episode_rewards.append(total_reward)
            self.epsilon = max(0.01, self.epsilon * 0.995)
        
        return episode_rewards
```

## Policy Gradient Methods

Policy gradient methods directly optimize the policy.

### REINFORCE Algorithm

**Policy Gradient Theorem:**
```
∇_θ J(θ) = E_π[∇_θ log π(a|s,θ) Q^π(s,a)]
```

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
        return x

class REINFORCEAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99):
        self.gamma = gamma
        self.policy_net = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        self.saved_log_probs = []
        self.rewards = []
    
    def select_action(self, state):
        """Select action using current policy"""
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy_net(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        
        # Save log probability for training
        self.saved_log_probs.append(action_dist.log_prob(action))
        
        return action.item()
    
    def update(self):
        """Update policy using REINFORCE"""
        R = 0
        returns = []
        
        # Calculate returns
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        # Normalize returns
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate loss
        policy_loss = []
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        # Update policy
        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        
        # Clear saved values
        self.saved_log_probs = []
        self.rewards = []
        
        return policy_loss.item()
    
    def train(self, env, num_episodes=1000):
        """Train the agent"""
        episode_rewards = []
        
        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = self.select_action(state)
                next_state, reward, done = env.step(action)
                
                self.rewards.append(reward)
                state = next_state
                total_reward += reward
            
            # Update policy after episode
            loss = self.update()
            episode_rewards.append(total_reward)
            
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}")
        
        return episode_rewards
```

## Actor-Critic Methods

Combine value-based and policy-based methods.

### Advantage Actor-Critic (A2C)

```python
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        shared_features = self.shared(x)
        action_probs = self.actor(shared_features)
        state_value = self.critic(shared_features)
        return action_probs, state_value

class A2CAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99):
        self.gamma = gamma
        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    
    def select_action(self, state):
        """Select action and get state value"""
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs, state_value = self.model(state)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        
        return action.item(), action_dist.log_prob(action), state_value
    
    def train_step(self, log_prob, value, reward, next_value, done):
        """Single training step"""
        # Calculate advantage
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * next_value
        
        advantage = td_target - value
        
        # Actor loss (policy gradient)
        actor_loss = -log_prob * advantage.detach()
        
        # Critic loss (value function)
        critic_loss = F.mse_loss(value, torch.tensor([td_target]))
        
        # Total loss
        loss = actor_loss + critic_loss
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
```

## Multi-Armed Bandits

Simplified RL problem with one state.

### Epsilon-Greedy Bandit

```python
class EpsilonGreedyBandit:
    def __init__(self, n_arms, epsilon=0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.q_values = np.zeros(n_arms)  # Estimated action values
        self.action_counts = np.zeros(n_arms)  # Number of times each action selected
    
    def select_action(self):
        """Epsilon-greedy action selection"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        else:
            return np.argmax(self.q_values)
    
    def update(self, action, reward):
        """Update Q-value estimate"""
        self.action_counts[action] += 1
        alpha = 1 / self.action_counts[action]
        self.q_values[action] += alpha * (reward - self.q_values[action])

# Test bandit
true_rewards = [0.1, 0.5, 0.3, 0.7, 0.2]
bandit = EpsilonGreedyBandit(n_arms=5, epsilon=0.1)

total_reward = 0
for t in range(1000):
    action = bandit.select_action()
    reward = true_rewards[action] + np.random.normal(0, 0.1)
    bandit.update(action, reward)
    total_reward += reward

print(f"True rewards: {true_rewards}")
print(f"Estimated rewards: {bandit.q_values}")
print(f"Total reward: {total_reward:.2f}")
```

### Upper Confidence Bound (UCB)

```python
class UCBBandit:
    def __init__(self, n_arms, c=2):
        self.n_arms = n_arms
        self.c = c
        self.q_values = np.zeros(n_arms)
        self.action_counts = np.zeros(n_arms)
        self.t = 0
    
    def select_action(self):
        """UCB action selection"""
        self.t += 1
        
        # Select each arm at least once
        if 0 in self.action_counts:
            return np.argmin(self.action_counts)
        
        # UCB formula
        ucb_values = self.q_values + self.c * np.sqrt(np.log(self.t) / self.action_counts)
        return np.argmax(ucb_values)
    
    def update(self, action, reward):
        """Update Q-value estimate"""
        self.action_counts[action] += 1
        alpha = 1 / self.action_counts[action]
        self.q_values[action] += alpha * (reward - self.q_values[action])
```

## Practical Tips

1. **Start Simple**: Begin with simple environments and algorithms
2. **Hyperparameter Tuning**: Learning rate, discount factor, and exploration rate are crucial
3. **Experience Replay**: Store and replay past experiences (covered in deep RL)
4. **Reward Shaping**: Design rewards carefully to guide learning
5. **Exploration vs Exploitation**: Balance is key for good performance
6. **Curriculum Learning**: Start with easy tasks and gradually increase difficulty

## Resources

- "Reinforcement Learning: An Introduction" by Sutton and Barto
- OpenAI Gym: https://gym.openai.com/
- Stable Baselines3: https://stable-baselines3.readthedocs.io/
- David Silver's RL Course: https://www.davidsilver.uk/teaching/

