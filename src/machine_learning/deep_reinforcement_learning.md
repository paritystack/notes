# Deep Reinforcement Learning

Deep RL combines deep learning with reinforcement learning to handle high-dimensional state and action spaces.

## Table of Contents

1. [Deep Q-Networks (DQN)](#deep-q-networks-dqn)
2. [Policy Gradient Methods](#policy-gradient-methods)
3. [Actor-Critic Methods](#actor-critic-methods)
4. [A3C (Asynchronous Advantage Actor-Critic)](#a3c)
5. [PPO (Proximal Policy Optimization)](#ppo)
6. [DDPG (Deep Deterministic Policy Gradient)](#ddpg)
7. [SAC (Soft Actor-Critic)](#sac)
8. [TD3 (Twin Delayed DDPG)](#td3)
9. [Model-Based RL](#model-based-rl)
10. [Multi-Agent RL](#multi-agent-rl)

## Deep Q-Networks (DQN)

DQN uses deep neural networks to approximate Q-values for high-dimensional states.

### Basic DQN

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque, namedtuple
import random

# Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[128, 128]):
        super(QNetwork, self).__init__()
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        return self.network(state)

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
        self.experience = namedtuple('Experience', 
                                    ['state', 'action', 'reward', 'next_state', 'done'])
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append(self.experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor([e.state for e in experiences])
        actions = torch.LongTensor([e.action for e in experiences])
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states = torch.FloatTensor([e.next_state for e in experiences])
        dones = torch.FloatTensor([e.done for e in experiences])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

# DQN Agent
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 buffer_size=100000, batch_size=64, target_update_freq=10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Q-Networks
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        self.steps = 0
    
    def select_action(self, state, training=True):
        """Epsilon-greedy action selection"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state)
            return q_values.argmax(1).item()
    
    def train_step(self):
        """Single training step"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def train(self, env, num_episodes=1000, max_steps=1000):
        """Train the agent"""
        episode_rewards = []
        
        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0
            
            for step in range(max_steps):
                # Select and perform action
                action = self.select_action(state, training=True)
                next_state, reward, done, _ = env.step(action)
                
                # Store transition
                self.replay_buffer.push(state, action, reward, next_state, done)
                
                # Train
                loss = self.train_step()
                
                state = next_state
                total_reward += reward
                
                if done:
                    break
            
            episode_rewards.append(total_reward)
            
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"Episode {episode+1}, Avg Reward: {avg_reward:.2f}, "
                      f"Epsilon: {self.epsilon:.3f}")
        
        return episode_rewards
```

### Double DQN

Reduces overestimation bias in Q-learning.

```python
class DoubleDQNAgent(DQNAgent):
    def train_step(self):
        """Double DQN training step"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Double DQN: use online network to select actions, target network to evaluate
        with torch.no_grad():
            # Select actions using online network
            next_actions = self.q_network(next_states).argmax(1)
            # Evaluate using target network
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
```

### Dueling DQN

Separates value and advantage functions.

```python
class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DuelingQNetwork, self).__init__()
        
        # Feature extraction
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state):
        features = self.feature(state)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values
```

### Prioritized Experience Replay

```python
class PrioritizedReplayBuffer:
    def __init__(self, capacity=100000, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]
        
        # Calculate sampling probabilities
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # Importance sampling weights
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        
        # Get experiences
        experiences = [self.buffer[idx] for idx in indices]
        states = torch.FloatTensor([e[0] for e in experiences])
        actions = torch.LongTensor([e[1] for e in experiences])
        rewards = torch.FloatTensor([e[2] for e in experiences])
        next_states = torch.FloatTensor([e[3] for e in experiences])
        dones = torch.FloatTensor([e[4] for e in experiences])
        weights = torch.FloatTensor(weights)
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self):
        return len(self.buffer)
```

## Policy Gradient Methods

### REINFORCE with Baseline

```python
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state):
        return self.network(state)

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super(ValueNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        return self.network(state)

class REINFORCEWithBaseline:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99):
        self.gamma = gamma
        
        self.policy_net = PolicyNetwork(state_dim, action_dim)
        self.value_net = ValueNetwork(state_dim)
        
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        
        self.saved_log_probs = []
        self.saved_values = []
        self.rewards = []
    
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        
        # Get action probabilities and value
        probs = self.policy_net(state)
        value = self.value_net(state)
        
        # Sample action
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        
        # Save log prob and value
        self.saved_log_probs.append(action_dist.log_prob(action))
        self.saved_values.append(value)
        
        return action.item()
    
    def train_step(self):
        R = 0
        returns = []
        
        # Calculate returns
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        policy_losses = []
        value_losses = []
        
        # Calculate losses
        for log_prob, value, R in zip(self.saved_log_probs, self.saved_values, returns):
            advantage = R - value.item()
            
            # Policy loss (REINFORCE with baseline)
            policy_losses.append(-log_prob * advantage)
            
            # Value loss
            value_losses.append(F.mse_loss(value, torch.tensor([[R]])))
        
        # Update policy network
        self.policy_optimizer.zero_grad()
        policy_loss = torch.stack(policy_losses).sum()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Update value network
        self.value_optimizer.zero_grad()
        value_loss = torch.stack(value_losses).sum()
        value_loss.backward()
        self.value_optimizer.step()
        
        # Clear saved values
        self.saved_log_probs = []
        self.saved_values = []
        self.rewards = []
        
        return policy_loss.item(), value_loss.item()
```

## Actor-Critic Methods

### Advantage Actor-Critic (A2C)

```python
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCritic, self).__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        shared_features = self.shared(state)
        action_probs = self.actor(shared_features)
        state_value = self.critic(shared_features)
        return action_probs, state_value

class A2CAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, 
                 value_coef=0.5, entropy_coef=0.01):
        self.gamma = gamma
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs, state_value = self.model(state)
        
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        
        return action.item(), action_dist.log_prob(action), action_dist.entropy(), state_value
    
    def train_step(self, states, actions, rewards, next_states, dones):
        """Train on a batch of experiences"""
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # Get action probabilities and values
        action_probs, values = self.model(states)
        _, next_values = self.model(next_states)
        
        # Calculate advantages
        td_targets = rewards + (1 - dones) * self.gamma * next_values.squeeze()
        advantages = td_targets - values.squeeze()
        
        # Actor loss
        action_dist = torch.distributions.Categorical(action_probs)
        log_probs = action_dist.log_prob(actions)
        actor_loss = -(log_probs * advantages.detach()).mean()
        
        # Critic loss
        critic_loss = F.mse_loss(values.squeeze(), td_targets.detach())
        
        # Entropy bonus (encourages exploration)
        entropy = action_dist.entropy().mean()
        
        # Total loss
        loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
        
        return loss.item(), actor_loss.item(), critic_loss.item(), entropy.item()
```

## A3C

Asynchronous Advantage Actor-Critic uses multiple parallel workers.

```python
import torch.multiprocessing as mp

class A3CAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        
        # Global network (shared across workers)
        self.global_model = ActorCritic(state_dim, action_dim)
        self.global_model.share_memory()
        
        self.optimizer = optim.Adam(self.global_model.parameters(), lr=lr)
    
    def worker(self, worker_id, env_fn, num_episodes=1000):
        """Worker process for A3C"""
        local_model = ActorCritic(self.state_dim, self.action_dim)
        env = env_fn()
        
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            
            states, actions, rewards = [], [], []
            
            while not done:
                # Sync local model with global model
                local_model.load_state_dict(self.global_model.state_dict())
                
                # Select action
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action_probs, _ = local_model(state_tensor)
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()
                
                # Take action
                next_state, reward, done, _ = env.step(action.item())
                
                # Store transition
                states.append(state)
                actions.append(action.item())
                rewards.append(reward)
                
                state = next_state
                
                # Update global network periodically
                if len(states) >= 20 or done:
                    self._update_global(local_model, states, actions, rewards, next_state, done)
                    states, actions, rewards = [], [], []
    
    def _update_global(self, local_model, states, actions, rewards, next_state, done):
        """Update global network using local gradients"""
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.LongTensor(actions)
        
        # Calculate returns
        R = 0
        if not done:
            _, next_value = local_model(torch.FloatTensor(next_state).unsqueeze(0))
            R = next_value.item()
        
        returns = []
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns)
        
        # Calculate loss
        action_probs, values = local_model(states_tensor)
        action_dist = torch.distributions.Categorical(action_probs)
        log_probs = action_dist.log_prob(actions_tensor)
        
        advantages = returns - values.squeeze()
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()
        entropy = action_dist.entropy().mean()
        
        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
        
        # Update global network
        self.optimizer.zero_grad()
        loss.backward()
        
        # Transfer gradients to global network
        for local_param, global_param in zip(local_model.parameters(), 
                                             self.global_model.parameters()):
            if global_param.grad is None:
                global_param.grad = local_param.grad
            else:
                global_param.grad += local_param.grad
        
        self.optimizer.step()
    
    def train(self, env_fn, num_workers=4, num_episodes=1000):
        """Train using multiple parallel workers"""
        processes = []
        
        for worker_id in range(num_workers):
            p = mp.Process(target=self.worker, args=(worker_id, env_fn, num_episodes))
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()
```

## PPO

Proximal Policy Optimization is a policy gradient method with clipped objective.

```python
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=0.0003, gamma=0.99, 
                 epsilon=0.2, value_coef=0.5, entropy_coef=0.01, 
                 epochs=10, batch_size=64):
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Storage for rollouts
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def select_action(self, state):
        """Select action using current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action_probs, value = self.model(state_tensor)
        
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()
    
    def store_transition(self, state, action, reward, log_prob, value, done):
        """Store transition in buffer"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def compute_gae(self, next_value, gamma=0.99, lam=0.95):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        
        values = self.values + [next_value]
        
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + gamma * values[t + 1] * (1 - self.dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, self.values)]
        
        return advantages, returns
    
    def update(self, next_state):
        """PPO update"""
        # Get next value for GAE
        with torch.no_grad():
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            _, next_value = self.model(next_state_tensor)
            next_value = next_value.item()
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(next_value)
        
        # Convert to tensors
        states = torch.FloatTensor(self.states)
        actions = torch.LongTensor(self.actions)
        old_log_probs = torch.FloatTensor(self.log_probs)
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update for multiple epochs
        for _ in range(self.epochs):
            # Get current policy
            action_probs, values = self.model(states)
            action_dist = torch.distributions.Categorical(action_probs)
            log_probs = action_dist.log_prob(actions)
            entropy = action_dist.entropy()
            
            # Ratio for clipping
            ratio = torch.exp(log_probs - old_log_probs)
            
            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            critic_loss = F.mse_loss(values.squeeze(), returns)
            
            # Entropy bonus
            entropy_loss = -entropy.mean()
            
            # Total loss
            loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
        
        # Clear buffers
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
        return loss.item()
```

## DDPG

Deep Deterministic Policy Gradient for continuous action spaces.

```python
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        super(Actor, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        self.max_action = max_action
    
    def forward(self, state):
        return self.max_action * self.network(state)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        return self.network(torch.cat([state, action], dim=1))

class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action, lr=0.001, 
                 gamma=0.99, tau=0.005, noise_std=0.1):
        self.gamma = gamma
        self.tau = tau
        self.noise_std = noise_std
        self.max_action = max_action
        
        # Actor networks
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        # Critic networks
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.replay_buffer = ReplayBuffer()
    
    def select_action(self, state, training=True):
        """Select action with optional exploration noise"""
        state = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]
        
        if training:
            noise = np.random.normal(0, self.noise_std, size=action.shape)
            action = np.clip(action + noise, -self.max_action, self.max_action)
        
        return action
    
    def train_step(self, batch_size=64):
        """Single DDPG training step"""
        if len(self.replay_buffer) < batch_size:
            return None, None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Update critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * target_q
        
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target networks
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)
        
        return actor_loss.item(), critic_loss.item()
    
    def _soft_update(self, source, target):
        """Soft update of target network"""
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1 - self.tau) * target_param.data
            )
```

## SAC

Soft Actor-Critic with entropy maximization.

```python
class SACAgent:
    def __init__(self, state_dim, action_dim, max_action, lr=0.0003, 
                 gamma=0.99, tau=0.005, alpha=0.2):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha  # Temperature parameter
        self.max_action = max_action
        
        # Actor (stochastic policy)
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        # Two Q-functions (critics)
        self.critic_1 = Critic(state_dim, action_dim)
        self.critic_2 = Critic(state_dim, action_dim)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=lr)
        
        # Target critics
        self.critic_1_target = Critic(state_dim, action_dim)
        self.critic_2_target = Critic(state_dim, action_dim)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        
        self.replay_buffer = ReplayBuffer()
    
    def train_step(self, batch_size=256):
        """SAC training step"""
        if len(self.replay_buffer) < batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Update critics
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            target_q1 = self.critic_1_target(next_states, next_actions)
            target_q2 = self.critic_2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * target_q
        
        current_q1 = self.critic_1(states, actions)
        current_q2 = self.critic_2(states, actions)
        
        critic_1_loss = F.mse_loss(current_q1, target_q)
        critic_2_loss = F.mse_loss(current_q2, target_q)
        
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()
        
        # Update actor
        new_actions, log_probs = self.actor.sample(states)
        q1 = self.critic_1(states, new_actions)
        q2 = self.critic_2(states, new_actions)
        q = torch.min(q1, q2)
        
        actor_loss = (self.alpha * log_probs - q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target networks
        self._soft_update(self.critic_1, self.critic_1_target)
        self._soft_update(self.critic_2, self.critic_2_target)
        
        return actor_loss.item(), critic_1_loss.item(), critic_2_loss.item()
```

## TD3

Twin Delayed Deep Deterministic Policy Gradient.

```python
class TD3Agent(DDPGAgent):
    def __init__(self, state_dim, action_dim, max_action, lr=0.001, 
                 gamma=0.99, tau=0.005, policy_noise=0.2, 
                 noise_clip=0.5, policy_delay=2):
        super().__init__(state_dim, action_dim, max_action, lr, gamma, tau)
        
        # Twin critics
        self.critic_2 = Critic(state_dim, action_dim)
        self.critic_2_target = Critic(state_dim, action_dim)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=lr)
        
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.total_iterations = 0
    
    def train_step(self, batch_size=64):
        """TD3 training step"""
        self.total_iterations += 1
        
        if len(self.replay_buffer) < batch_size:
            return None, None
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Update critics
        with torch.no_grad():
            # Target policy smoothing
            noise = torch.randn_like(actions) * self.policy_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            
            next_actions = self.actor_target(next_states)
            next_actions = torch.clamp(next_actions + noise, -self.max_action, self.max_action)
            
            # Twin Q targets
            target_q1 = self.critic_target(next_states, next_actions)
            target_q2 = self.critic_2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * target_q
        
        current_q1 = self.critic(states, actions)
        current_q2 = self.critic_2(states, actions)
        
        critic_1_loss = F.mse_loss(current_q1, target_q)
        critic_2_loss = F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_optimizer.step()
        
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()
        
        # Delayed policy updates
        actor_loss = None
        if self.total_iterations % self.policy_delay == 0:
            # Update actor
            actor_loss = -self.critic(states, self.actor(states)).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Soft update target networks
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)
            self._soft_update(self.critic_2, self.critic_2_target)
        
        return actor_loss.item() if actor_loss else None, critic_1_loss.item()
```

## Practical Tips

1. **Hyperparameter Tuning**: Learning rates, discount factors crucial
2. **Reward Scaling**: Normalize rewards for stable training
3. **Network Architecture**: Start simple, increase complexity as needed
4. **Exploration**: Balance exploration vs exploitation
5. **Curriculum Learning**: Start with easier tasks
6. **Distributed Training**: Use parallel environments for faster learning

## Resources

- OpenAI Spinning Up: https://spinningup.openai.com/
- Stable Baselines3: https://stable-baselines3.readthedocs.io/
- "Deep Reinforcement Learning Hands-On" by Maxim Lapan
- DeepMind papers: https://www.deepmind.com/research

