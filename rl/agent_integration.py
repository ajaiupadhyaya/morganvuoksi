"""Bridge RL agents with the execution layer."""
from __future__ import annotations

from typing import Any

import numpy as np

from src.models.rl import PPO
from src.execution.options_engine import OptionsEngine


class RLTradingBridge:
    def __init__(self, env: Any, paper: bool = True) -> None:
        self.env = env
        self.engine = OptionsEngine(paper=paper)
        self.agent = PPO({'state_dim': env.state_dim, 'action_dim': env.action_dim})

    def run_episode(self, train: bool = True) -> float:
        state = self.env.reset()
        done = False
        total_reward = 0.0
        memory = []
        while not done:
            action, prob = self.agent.select_action(state)
            next_state, reward, done, info = self.env.step(action)
            if train:
                memory.append((state, action, prob, reward, next_state, float(done)))
            total_reward += reward
            state = next_state
        if train:
            self.agent.update(memory)
        return total_reward

    def execute_signal(self, symbol: str, strike: float, expiry: str, direction: str) -> None:
        self.engine.enter_trade(symbol, direction, strike, expiry)

