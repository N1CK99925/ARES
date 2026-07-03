"""
DQN self-play training script for ARES.

This is the orchestration layer: it creates the environment, networks,
replay buffer, trainer, and commanders, then runs episodes by calling
engine.step() in a loop. The training script — not the commander and
not the simulator — owns the learning process.
"""

import sys
import pathlib
import argparse

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "ares-sim"))

import torch
import torch.optim as optim

from config.seeds import get_seed_1, get_seed_2, get_seed_3, get_seed_4
from core.tick import TickEngine
from agents.Commander.DQNCommander import DQNCommander
from agents.RL.DQN.network import DQNNetwork
from agents.RL.DQN.encoder import ObservationEncoder
from agents.RL.DQN.epsilon import EpsilonScheduler
from agents.RL.DQN.replay_buffer import ReplayBuffer
from agents.RL.DQN.trainer import Trainer
import random
NUM_ZONES = 5


def _build_action_tensor(action_indices: dict[int, int]) -> torch.Tensor:
    """Convert zone → action_idx dict to a fixed-size tensor of shape (NUM_ZONES,).

    Each position corresponds to one zone (index 0 → zone 1, etc.).
    Zones with no units (absent from the dict) default to action 0 (hold).
    """
    t = torch.zeros(NUM_ZONES, dtype=torch.long)
    for zone_id, action_idx in action_indices.items():
        t[zone_id - 1] = action_idx
    return t


def run_training(resume_from: str | None = None):
    # ── 1. Hyperparameters ──────────────────────────────────────────────
    state_dim = 12            # Must match ObservationEncoder feature count
    hidden_dim = 128
    batch_size = 32
    gamma = 0.99
    lr = 1e-3
    episodes = 2000
    max_ticks_per_episode = 1000
    replay_capacity = 50_000
    sync_every_k_steps = 100
    max_grad_norm = 10.0
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay_steps = 10_000

    # ── 2. Create networks ──────────────────────────────────────────────
    online_net = DQNNetwork(state_dim, hidden_dim)
    target_net = DQNNetwork(state_dim, hidden_dim)
    target_net.load_state_dict(online_net.state_dict())

    start_episode = 1
    if resume_from is not None:
        state_dict = torch.load(resume_from, map_location=torch.device("cpu"))
        online_net.load_state_dict(state_dict)
        target_net.load_state_dict(state_dict)
        # Infer start episode from filename like checkpoint_dqn_ep_500.pt
        stem = pathlib.Path(resume_from).stem
        if stem.startswith("checkpoint_dqn_ep_"):
            start_episode = int(stem.rsplit("_", 1)[-1]) + 1
        print(f"Resumed weights from {resume_from}, starting at episode {start_episode}")

    # ── 3. Create shared infrastructure ─────────────────────────────────
    encoder = ObservationEncoder()
    replay_buffer = ReplayBuffer(capacity=replay_capacity)
    optimizer = optim.Adam(online_net.parameters(), lr=lr)

    trainer = Trainer(
        online_net=online_net,
        target_net=target_net,
        optimizer=optimizer,
        replay_buffer=replay_buffer,
        batch_size=batch_size,
        gamma=gamma,
        sync_every_k_steps=sync_every_k_steps,
        max_grad_norm=max_grad_norm,
    )

    # ── 4. Create commanders (both share the SAME online network) ───────
    blue_eps = EpsilonScheduler(start=epsilon_start, end=epsilon_end, decay_steps=epsilon_decay_steps)
    red_eps = EpsilonScheduler(start=epsilon_start, end=epsilon_end, decay_steps=epsilon_decay_steps)

    blue_agent = DQNCommander(network=online_net, encoder=encoder, epsilon_scheduler=blue_eps)
    red_agent = DQNCommander(network=online_net, encoder=encoder, epsilon_scheduler=red_eps)

    if resume_from is not None:
        # Epsilon already at floor after prior training; keep exploration minimal.
        blue_agent.tick_counter = epsilon_decay_steps
        red_agent.tick_counter = epsilon_decay_steps
    
    seeds = [get_seed_1(), get_seed_2(), get_seed_3(), get_seed_4()]
    for episode in range(start_episode, episodes + 1):
        seed_fn = random.choice(seeds)
        initial_state = seed_fn
        engine = TickEngine(initial_state, blue_commander=blue_agent, red_commander=red_agent)

        ticks_this_episode = 0
        episode_reward_blue = 0.0
        last_loss = None
        last_zone_losses = None
        last_zone_q_stats = None

        while not engine.done and ticks_this_episode < max_ticks_per_episode:
            result = engine.step()
            ticks_this_episode += 1

            # ── Encode observations and build action tensors ────────────
            blue_state_t = encoder.encode(result.blue_obs)
            blue_action_t = _build_action_tensor(blue_agent.last_action_indices)

            red_state_t = encoder.encode(result.red_obs)
            red_action_t = _build_action_tensor(red_agent.last_action_indices)

            if result.done:
                # Terminal state: next obs is a zero tensor
                blue_next_t = torch.zeros_like(blue_state_t)
                red_next_t = torch.zeros_like(red_state_t)
            else:
                blue_next_t = encoder.encode(result.next_blue_obs)
                red_next_t = encoder.encode(result.next_red_obs)

            # ── Push both sides' transitions into shared replay ─────────
            replay_buffer.push(blue_state_t, blue_action_t, result.blue_reward, blue_next_t, result.done)
            replay_buffer.push(red_state_t, red_action_t, result.red_reward, red_next_t, result.done)

            episode_reward_blue += result.blue_reward

            # ── Train after each step ───────────────────────────────────
            train_result = trainer.train_step()
            if train_result is not None:
                last_loss, last_zone_losses, last_zone_q_stats = train_result

        # ── Episode reporting ───────────────────────────────────────────
        if episode % 10 == 0:
            current_eps = blue_eps.get_epsilon(blue_agent.tick_counter)
            zone_loss_str = "N/A"
            if last_zone_losses is not None:
                zone_loss_str = ", ".join(f"Z{z}:{l:.2f}" for z, l in sorted(last_zone_losses.items()))

            q_mean_str = "N/A"
            q_max_str = "N/A"
            if last_zone_q_stats is not None:
                q_mean_str = ", ".join(
                    f"Z{z}:{s['mean']:.1f}" for z, s in sorted(last_zone_q_stats.items())
                )
                q_max_str = ", ".join(
                    f"Z{z}:{s['max']:.1f}" for z, s in sorted(last_zone_q_stats.items())
                )

            print(
                f"Episode {episode:4d} | "
                f"Ticks: {ticks_this_episode:4d} | "
                f"Blue Reward: {episode_reward_blue:+8.2f} | "
                f"Loss: {last_loss if last_loss is not None else 'N/A':>6.2f} "
                f"({zone_loss_str}) | "
                f"Q-mean: [{q_mean_str}] | "
                f"Q-max: [{q_max_str}] | "
                f"Epsilon: {current_eps:.3f} | "
                f"Buffer: {len(replay_buffer)}"
            )

        # ── Periodic checkpoint ─────────────────────────────────────────
        if episode in {1500, 2000}:
            ckpt_path = f"checkpoint_dqn_ep_{episode}.pt"
            blue_agent.save(ckpt_path)
            print(f"  -> Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARES DQN self-play training")
    parser.add_argument(
        "--resume",
        type=str,
        default="checkpoint_dqn_ep_500.pt",
        help="Checkpoint to resume from (default: checkpoint_dqn_ep_500.pt). Pass '' to train from scratch.",
    )
    args = parser.parse_args()
    resume_from = args.resume or None
    run_training(resume_from=resume_from)
