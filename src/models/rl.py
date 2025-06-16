"""
Reinforcement Learning models for portfolio optimization.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple
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
            nn.Softmax(dim=-1)
        )
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        action_probs = self.actor(state)
        state_value = self.critic(state)
        return action_probs, state_value

class PPO:
    """Proximal Policy Optimization for portfolio management."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.state_dim = config.get('state_dim', 10)
        self.action_dim = config.get('action_dim', 5)
        self.hidden_dim = config.get('hidden_dim', 64)
        self.lr = config.get('learning_rate', 3e-4)
        self.gamma = config.get('gamma', 0.99)
        self.eps_clip = config.get('eps_clip', 0.2)
        self.K_epochs = config.get('K_epochs', 10)
        
        self.policy = ActorCritic(self.state_dim, self.action_dim, self.hidden_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.MseLoss = nn.MSELoss()
    
    def select_action(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """Select action based on current state."""
        state = torch.FloatTensor(state)
        action_probs, _ = self.policy(state)
        action = torch.multinomial(action_probs, 1).item()
        return action, action_probs[action].item()
    
    def update(self, memory: List[Tuple]) -> float:
        """Update policy using PPO."""
        # Extract memory
        states = torch.FloatTensor([x[0] for x in memory])
        actions = torch.LongTensor([x[1] for x in memory])
        old_probs = torch.FloatTensor([x[2] for x in memory])
        rewards = torch.FloatTensor([x[3] for x in memory])
        next_states = torch.FloatTensor([x[4] for x in memory])
        dones = torch.FloatTensor([x[5] for x in memory])
        
        # Calculate advantages
        with torch.no_grad():
            _, state_values = self.policy(states)
            _, next_state_values = self.policy(next_states)
            advantages = rewards + self.gamma * next_state_values * (1 - dones) - state_values
        
        # Optimize policy for K epochs
        total_loss = 0
        for _ in range(self.K_epochs):
            # Get current action probabilities and state values
            action_probs, state_values = self.policy(states)
            
            # Calculate ratios
            ratios = action_probs.gather(1, actions.unsqueeze(1)) / old_probs.unsqueeze(1)
            
            # Calculate surrogate losses
            surr1 = ratios * advantages.unsqueeze(1)
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages.unsqueeze(1)
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Calculate value loss
            value_loss = self.MseLoss(state_values, rewards + self.gamma * next_state_values * (1 - dones))
            
            # Total loss
            loss = policy_loss + 0.5 * value_loss
            total_loss += loss.item()
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        return total_loss / self.K_epochs
    
    def save(self, path: str) -> None:
        """Save model."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
        logger.info(f"Saved PPO model to {path}")
    
    def load(self, path: str) -> None:
        """Load model."""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Loaded PPO model from {path}") 
