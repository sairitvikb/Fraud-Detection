import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
import logging
from collections import deque
import random

logger = logging.getLogger(__name__)


class RLAgent(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.target_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.update_target_network()
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.99
        self.batch_size = 64
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def act(self, state: torch.Tensor, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randint(0, self.q_network(state).size(-1) - 1)
        q_values = self.q_network(state)
        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, optimizer):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.stack([s for s, _, _, _, _ in batch])
        actions = torch.tensor([a for _, a, _, _, _ in batch])
        rewards = torch.tensor([r for _, _, r, _, _ in batch])
        next_states = torch.stack([s for _, _, _, s, _ in batch])
        dones = torch.tensor([d for _, _, _, _, d in batch])
        
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_network(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = F.mse_loss(current_q.squeeze(), target_q)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class ActorCriticAgent(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        action_probs = self.actor(state)
        value = self.critic(state)
        return action_probs, value


class PPOAgent(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.clip_epsilon = 0.2
    
    def get_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        action_logits = self.actor(state)
        action_dist = torch.distributions.Categorical(logits=action_logits)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action, log_prob
    
    def evaluate(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        action_logits = self.actor(states)
        action_dist = torch.distributions.Categorical(logits=action_logits)
        log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy()
        values = self.critic(states).squeeze()
        return log_probs, values, entropy


class FraudRLEnvironment:
    def __init__(self, transaction_data: torch.Tensor):
        self.transaction_data = transaction_data
        self.current_idx = 0
        self.state = None
        self.reset()
    
    def reset(self) -> torch.Tensor:
        self.current_idx = 0
        self.state = self.transaction_data[self.current_idx]
        return self.state
    
    def step(self, action: int) -> Tuple[torch.Tensor, float, bool]:
        is_fraud = self.transaction_data[self.current_idx, -1].item()
        reward = 1.0 if (action == 1 and is_fraud == 1) or (action == 0 and is_fraud == 0) else -1.0
        
        self.current_idx += 1
        done = self.current_idx >= len(self.transaction_data)
        
        if not done:
            self.state = self.transaction_data[self.current_idx]
        
        return self.state, reward, done
