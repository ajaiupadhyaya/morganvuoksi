"""
Reinforcement Learning model using PPO algorithm for portfolio optimization.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from typing import Dict, List, Tuple
import pandas as pd
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

class ActorCritic(nn.Module):
    """Actor-Critic network for PPO."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Bound actions to [-1, 1]
        )
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            module.bias.data.zero_()
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through both networks."""
        return self.actor(state), self.critic(state)

class PPO:
    """Proximal Policy Optimization algorithm for portfolio optimization."""
    def __init__(self, config: Dict):
        self.config = config
        self.state_dim = config.get('state_dim', 10)
        self.action_dim = config.get('action_dim', 1)
        self.hidden_dim = config.get('hidden_dim', 64)
        self.lr = config.get('learning_rate', 3e-4)
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_ratio = config.get('clip_ratio', 0.2)
        self.target_kl = config.get('target_kl', 0.01)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.value_coef = config.get('value_coef', 0.5)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.normalize_advantage = config.get('normalize_advantage', True)
        
        # Initialize networks
        self.actor_critic = ActorCritic(self.state_dim, self.action_dim, self.hidden_dim)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.lr)
        
        # Training state
        self.epoch = 0
        self.best_reward = float('-inf')
    
    def compute_gae(self, rewards: List[float], values: List[float], 
                   next_value: float, dones: List[bool]) -> Tuple[List[float], List[float]]:
        """Compute Generalized Advantage Estimation."""
        advantages = []
        returns = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value_t * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        return advantages, returns
    
    def update(self, states: torch.Tensor, actions: torch.Tensor, 
              old_log_probs: torch.Tensor, advantages: torch.Tensor,
              returns: torch.Tensor) -> Dict[str, float]:
        """Update policy and value networks."""
        # Normalize advantages
        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(self.config.get('n_epochs', 10)):
            # Forward pass
            action_pred, value_pred = self.actor_critic(states)
            
            # Compute loss
            ratio = torch.exp(action_pred - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            value_loss = 0.5 * (returns - value_pred).pow(2).mean()
            entropy_loss = -self.entropy_coef * action_pred.mean()
            
            total_loss = actor_loss + self.value_coef * value_loss + entropy_loss
            
            # Optimize
            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            # Early stopping
            kl = (old_log_probs - action_pred).mean()
            if kl > 1.5 * self.target_kl:
                break
        
        return {
            'actor_loss': actor_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item(),
            'kl': kl.item()
        }
    
    def train(self, env, n_episodes: int) -> List[float]:
        """Train the agent."""
        episode_rewards = []
        
        for episode in range(n_episodes):
            state = env.reset()
            episode_reward = 0
            states, actions, rewards, values, dones = [], [], [], [], []
            
            while True:
                # Get action
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    action, value = self.actor_critic(state_tensor)
                
                # Take action
                next_state, reward, done, _ = env.step(action.numpy()[0])
                
                # Store transition
                states.append(state)
                actions.append(action.numpy()[0])
                rewards.append(reward)
                values.append(value.item())
                dones.append(done)
                
                state = next_state
                episode_reward += reward
                
                if done:
                    # Compute GAE
                    next_value = 0 if done else value.item()
                    advantages, returns = self.compute_gae(rewards, values, next_value, dones)
                    
                    # Update networks
                    states_tensor = torch.FloatTensor(states)
                    actions_tensor = torch.FloatTensor(actions)
                    old_log_probs = torch.FloatTensor(actions)
                    advantages_tensor = torch.FloatTensor(advantages)
                    returns_tensor = torch.FloatTensor(returns)
                    
                    metrics = self.update(states_tensor, actions_tensor, old_log_probs,
                                        advantages_tensor, returns_tensor)
                    
                    # Log progress
                    logger.info(f"Episode {episode + 1}/{n_episodes}")
                    logger.info(f"Reward: {episode_reward:.2f}")
                    logger.info(f"Loss: {metrics['total_loss']:.4f}")
                    
                    episode_rewards.append(episode_reward)
                    break
        
        return episode_rewards
    
    def predict(self, state: np.ndarray) -> np.ndarray:
        """Get action for given state."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, _ = self.actor_critic(state_tensor)
        return action.numpy()[0]
    
    def save(self, path: str):
        """Save model weights."""
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'best_reward': self.best_reward
        }, path)
    
    def load(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path)
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_reward = checkpoint['best_reward'] 