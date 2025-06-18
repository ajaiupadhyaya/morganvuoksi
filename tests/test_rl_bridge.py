class DummyEnv:
    def __init__(self):
        self.state_dim = 4
        self.action_dim = 2
        self.current = 0

    def reset(self):
        self.current = 0
        return [0.0]*self.state_dim

    def step(self, action):
        self.current += 1
        reward = 1.0
        done = self.current >= 2
        return [0.0]*self.state_dim, reward, done, {}

import pytest
torch = pytest.importorskip("torch", reason="requires torch")
from rl.agent_integration import RLTradingBridge


def test_rl_bridge_run():
    env = DummyEnv()
    bridge = RLTradingBridge(env)
    reward = bridge.run_episode(train=False)
    assert reward > 0
