"""Smoke test for DQN training: run 2 episodes and verify everything connects."""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "ares-sim"))

import torch
import torch.optim as optim

from config.seeds import get_seed_1
from core.tick import TickEngine
from agents.Commander.DQNCommander import DQNCommander
from agents.RL.DQN.network import DQNNetwork
from agents.RL.DQN.encoder import ObservationEncoder
from agents.RL.DQN.epsilon import EpsilonScheduler
from agents.RL.DQN.replay_buffer import ReplayBuffer
from agents.RL.DQN.trainer import Trainer

NUM_ZONES = 5

def _build_action_tensor(action_indices):
    t = torch.zeros(NUM_ZONES, dtype=torch.long)
    for zone_id, action_idx in action_indices.items():
        t[zone_id - 1] = action_idx
    return t

print("=== DQN Training Smoke Test ===")
print()

# Create components
state_dim = 12
hidden_dim = 128
online_net = DQNNetwork(state_dim, hidden_dim)
target_net = DQNNetwork(state_dim, hidden_dim)
target_net.load_state_dict(online_net.state_dict())

encoder = ObservationEncoder()
replay_buffer = ReplayBuffer(capacity=10_000)
optimizer = optim.Adam(online_net.parameters(), lr=1e-3)

trainer = Trainer(
    online_net=online_net, target_net=target_net,
    optimizer=optimizer, replay_buffer=replay_buffer,
    batch_size=8, gamma=0.99, sync_every_k_steps=100,
)

blue_eps = EpsilonScheduler(start=1.0, end=0.05, decay_steps=1000)
red_eps = EpsilonScheduler(start=1.0, end=0.05, decay_steps=1000)
blue_agent = DQNCommander(network=online_net, encoder=encoder, epsilon_scheduler=blue_eps)
red_agent = DQNCommander(network=online_net, encoder=encoder, epsilon_scheduler=red_eps)

print("[OK] All components created successfully")

# Run 2 short episodes
for ep in range(1, 3):
    state = get_seed_1()
    engine = TickEngine(state, blue_commander=blue_agent, red_commander=red_agent)

    ticks = 0
    while not engine.done and ticks < 20:
        result = engine.step()
        ticks += 1

        blue_state_t = encoder.encode(result.blue_obs)
        blue_action_t = _build_action_tensor(blue_agent.last_action_indices)
        red_state_t = encoder.encode(result.red_obs)
        red_action_t = _build_action_tensor(red_agent.last_action_indices)

        if result.done:
            blue_next_t = torch.zeros_like(blue_state_t)
            red_next_t = torch.zeros_like(red_state_t)
        else:
            blue_next_t = encoder.encode(result.next_blue_obs)
            red_next_t = encoder.encode(result.next_red_obs)

        replay_buffer.push(blue_state_t, blue_action_t, result.blue_reward, blue_next_t, result.done)
        replay_buffer.push(red_state_t, red_action_t, result.red_reward, red_next_t, result.done)

        train_result = trainer.train_step()
        if train_result is not None:
            loss, _, _ = train_result

    print(f"  Episode {ep}: {ticks} ticks, buffer={len(replay_buffer)}, "
          f"done={result.done}, blue_reward_last={result.blue_reward}")

print()

# Verify trainer ran
print(f"Trainer steps completed: {trainer.step_count}")
print(f"Replay buffer size: {len(replay_buffer)}")

# Verify a training step returns a loss
train_result = trainer.train_step()
if train_result is not None:
    loss, zone_losses, zone_q_stats = train_result
else:
    loss = None
print(f"Training loss: {loss}")
assert loss is not None, "Expected a loss value!"
print()
print("[OK] DQN training pipeline works end-to-end")
