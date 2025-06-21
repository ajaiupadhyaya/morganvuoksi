"""
Reinforcement Learning Models for Trading
Implements TD3 and SAC algorithms for autonomous trading strategies.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class Actor(nn.Module):
    """Actor network for TD3 and SAC."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.tanh = nn.Tanh()
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = self.tanh(self.fc3(x))
        return action

class Critic(nn.Module):
    """Critic network for TD3 and SAC."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value

class TD3Agent:
    """Twin Delayed Deep Deterministic Policy Gradient (TD3) Agent."""
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or {}
        
        # Hyperparameters
        self.lr_actor = self.config.get('lr_actor', 3e-4)
        self.lr_critic = self.config.get('lr_critic', 3e-4)
        self.gamma = self.config.get('gamma', 0.99)
        self.tau = self.config.get('tau', 0.005)
        self.noise_clip = self.config.get('noise_clip', 0.5)
        self.policy_noise = self.config.get('policy_noise', 0.2)
        self.policy_freq = self.config.get('policy_freq', 2)
        
        # Networks
        self.actor = Actor(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic1 = Critic(state_dim, action_dim)
        self.critic2 = Critic(state_dim, action_dim)
        self.critic1_target = Critic(state_dim, action_dim)
        self.critic2_target = Critic(state_dim, action_dim)
        
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.lr_critic)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.lr_critic)
        
        self.total_iterations = 0
        
    def select_action(self, state: np.ndarray, noise: float = 0.1) -> np.ndarray:
        """Select action with exploration noise."""
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().numpy()[0]
        
        if noise > 0:
            action += np.random.normal(0, noise, size=action.shape)
            action = np.clip(action, -1, 1)
            
        return action
    
    def train(self, replay_buffer, batch_size: int = 100) -> Dict:
        """Train the TD3 agent."""
        if len(replay_buffer) < batch_size:
            return {}
            
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        
        # Convert to tensors
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor(done).unsqueeze(1)
        
        # Critic loss
        with torch.no_grad():
            noise = torch.randn_like(action) * self.policy_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            
            next_action = self.actor_target(next_state) + noise
            next_action = torch.clamp(next_action, -1, 1)
            
            target_q1 = self.critic1_target(next_state, next_action)
            target_q2 = self.critic2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1 - done) * self.gamma * target_q
        
        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)
        
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        # Update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Actor loss (delayed policy updates)
        actor_loss = 0
        if self.total_iterations % self.policy_freq == 0:
            actor_loss = -self.critic1(state, self.actor(state)).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Update target networks
            self._update_target_networks()
        
        self.total_iterations += 1
        
        return {
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'actor_loss': actor_loss.item() if actor_loss != 0 else 0
        }
    
    def _update_target_networks(self):
        """Soft update target networks."""
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save(self, path: str):
        """Save the model."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic1_target_state_dict': self.critic1_target.state_dict(),
            'critic2_target_state_dict': self.critic2_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
            'total_iterations': self.total_iterations
        }, path)
    
    def load(self, path: str):
        """Load the model."""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target_state_dict'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
        self.total_iterations = checkpoint['total_iterations']

class SACAgent:
    """Soft Actor-Critic (SAC) Agent."""
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or {}
        
        # Hyperparameters
        self.lr_actor = self.config.get('lr_actor', 3e-4)
        self.lr_critic = self.config.get('lr_critic', 3e-4)
        self.lr_alpha = self.config.get('lr_alpha', 3e-4)
        self.gamma = self.config.get('gamma', 0.99)
        self.tau = self.config.get('tau', 0.005)
        self.alpha = self.config.get('alpha', 0.2)
        self.auto_entropy = self.config.get('auto_entropy', True)
        
        # Networks
        self.actor = Actor(state_dim, action_dim)
        self.critic1 = Critic(state_dim, action_dim)
        self.critic2 = Critic(state_dim, action_dim)
        self.critic1_target = Critic(state_dim, action_dim)
        self.critic2_target = Critic(state_dim, action_dim)
        
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.lr_critic)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.lr_critic)
        
        if self.auto_entropy:
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr_alpha)
        
        self.total_iterations = 0
        
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Select action with exploration."""
        state = torch.FloatTensor(state).unsqueeze(0)
        
        if evaluate:
            action = self.actor(state).detach().numpy()[0]
            log_prob = torch.zeros(1)
        else:
            action = self.actor(state).detach().numpy()[0]
            # Add exploration noise
            action += np.random.normal(0, 0.1, size=action.shape)
            action = np.clip(action, -1, 1)
            log_prob = torch.zeros(1)
            
        return action, log_prob.numpy()
    
    def train(self, replay_buffer, batch_size: int = 100) -> Dict:
        """Train the SAC agent."""
        if len(replay_buffer) < batch_size:
            return {}
            
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        
        # Convert to tensors
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor(done).unsqueeze(1)
        
        # Critic loss
        with torch.no_grad():
            next_action, next_log_prob = self.select_action(next_state.numpy(), evaluate=True)
            next_action = torch.FloatTensor(next_action)
            next_log_prob = torch.FloatTensor(next_log_prob)
            
            target_q1 = self.critic1_target(next_state, next_action)
            target_q2 = self.critic2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = reward + (1 - done) * self.gamma * target_q
        
        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)
        
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        # Update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Actor loss
        new_action, new_log_prob = self.select_action(state.numpy(), evaluate=True)
        new_action = torch.FloatTensor(new_action)
        new_log_prob = torch.FloatTensor(new_log_prob)
        
        q1_new = self.critic1(state, new_action)
        q2_new = self.critic2(state, new_action)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.alpha * new_log_prob - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Alpha loss (if auto entropy)
        if self.auto_entropy:
            alpha_loss = -(self.log_alpha * (new_log_prob + self.config.get('target_entropy', -self.action_dim))).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
        
        # Update target networks
        self._update_target_networks()
        
        self.total_iterations += 1
        
        return {
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item() if self.auto_entropy else 0,
            'alpha': self.alpha.item()
        }
    
    def _update_target_networks(self):
        """Soft update target networks."""
        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save(self, path: str):
        """Save the model."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'critic1_target_state_dict': self.critic1_target.state_dict(),
            'critic2_target_state_dict': self.critic2_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict() if self.auto_entropy else None,
            'log_alpha': self.log_alpha if self.auto_entropy else None,
            'total_iterations': self.total_iterations
        }, path)
    
    def load(self, path: str):
        """Load the model."""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target_state_dict'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
        
        if self.auto_entropy and checkpoint['alpha_optimizer_state_dict']:
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
            self.log_alpha = checkpoint['log_alpha']
            self.alpha = self.log_alpha.exp()
            
        self.total_iterations = checkpoint['total_iterations']

class ReplayBuffer:
    """Experience replay buffer for RL agents."""
    
    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self.buffer = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        if len(self.buffer) < self.max_size:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.max_size
        
    def sample(self, batch_size: int) -> Tuple:
        """Sample batch of experiences."""
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in batch])
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
        
    def __len__(self):
        return len(self.buffer)

class TradingEnvironment:
    """Trading environment for RL agents."""
    
    def __init__(self, data: pd.DataFrame, initial_balance: float = 100000, 
                 transaction_cost: float = 0.001):
        self.data = data
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.reset()
        
    def reset(self):
        """Reset environment."""
        self.balance = self.initial_balance
        self.position = 0
        self.current_step = 0
        self.total_trades = 0
        self.portfolio_value = self.initial_balance
        
        return self._get_state()
    
    def step(self, action: np.ndarray) -> Tuple:
        """Take action and return next state, reward, done."""
        # Action is [-1, 1] representing position size
        target_position = action[0]  # Scale to position size
        
        # Calculate transaction cost
        position_change = target_position - self.position
        transaction_cost = abs(position_change) * self.transaction_cost * self.portfolio_value
        
        # Update position
        self.position = target_position
        
        # Calculate portfolio value
        current_price = self.data.iloc[self.current_step]['Close']
        self.portfolio_value = self.balance + self.position * current_price - transaction_cost
        
        # Calculate reward (daily return)
        if self.current_step > 0:
            prev_price = self.data.iloc[self.current_step - 1]['Close']
            daily_return = (current_price - prev_price) / prev_price
            reward = self.position * daily_return - transaction_cost / self.portfolio_value
        else:
            reward = 0
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        return self._get_state(), reward, done
    
    def _get_state(self) -> np.ndarray:
        """Get current state representation."""
        if self.current_step >= len(self.data):
            return np.zeros(10)
            
        current_data = self.data.iloc[self.current_step]
        
        # State features: price, volume, technical indicators, position, portfolio value
        state = np.array([
            current_data['Close'] / 1000,  # Normalized price
            current_data['Volume'] / 1000000,  # Normalized volume
            current_data.get('RSI', 50) / 100,  # RSI
            current_data.get('MA20', current_data['Close']) / current_data['Close'] - 1,  # MA ratio
            current_data.get('MA50', current_data['Close']) / current_data['Close'] - 1,  # MA ratio
            current_data.get('Volatility', 0.02) * 100,  # Volatility
            self.position,  # Current position
            self.portfolio_value / self.initial_balance - 1,  # Portfolio return
            self.current_step / len(self.data),  # Time progress
            self.total_trades / 100  # Trade frequency
        ])
        
        return state.astype(np.float32) 