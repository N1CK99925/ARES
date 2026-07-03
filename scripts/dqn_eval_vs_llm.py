"""
Evaluation harness for DQN vs LLM on a new seed 5.

Run from the repo root:
    python scripts/dqn_eval_vs_llm.py

This script evaluates checkpoints sequentially and stores a JSON result file
under the repo's `results/` directory.
"""

import sys
import pathlib
import json
import math
import os
import time
from collections import Counter
from datetime import datetime

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "ares-sim"))

import torch

from config.settings import Settings
from config.seeds import get_seed_5
from core.tick import TickEngine
from core.state import Side, ZoneControl
from agents.Commander.DQNCommander import DQNCommander
from agents.Commander.LLMCommander import LLMCommander
from agents.RL.DQN.network import DQNNetwork
from agents.RL.DQN.encoder import ObservationEncoder
from agents.RL.DQN.epsilon import EpsilonScheduler


def wilson_score(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0

    denominator = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    deviation = math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    lower = (centre - z * deviation) / denominator
    upper = (centre + z * deviation) / denominator
    return max(0, lower), min(1, upper)


def ensure_checkpoint_1(checkpoint_path: str) -> None:
    if os.path.exists(checkpoint_path):
        return

    print(f"Checkpoint {checkpoint_path} not found. Creating an initial untrained checkpoint.")
    network = DQNNetwork(state_dim=12, hidden_dim=128)
    torch.save(network.state_dict(), checkpoint_path)


def evaluate_checkpoint(checkpoint_path: str, n_episodes: int, model_name: str) -> dict:
    print("\n" + "=" * 60)
    print(f"Evaluating checkpoint: {checkpoint_path}")
    print("=" * 60)

    if checkpoint_path.endswith("checkpoint_dqn_ep_1.pt"):
        ensure_checkpoint_1(checkpoint_path)

    network = DQNNetwork(state_dim=12, hidden_dim=128)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    network.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    network.eval()

    encoder = ObservationEncoder()
    greedy_eps = EpsilonScheduler(start=0.0, end=0.0, decay_steps=1)

    wins = 0
    win_types = Counter()
    blue_wins = 0
    blue_games = 0
    red_wins = 0
    red_games = 0

    for episode in range(1, n_episodes + 1):
        dqn_agent = DQNCommander(network=network, encoder=encoder, epsilon_scheduler=greedy_eps)
        llm_agent = LLMCommander(model=model_name)

        dqn_is_blue = episode % 2 == 1
        blue = dqn_agent if dqn_is_blue else llm_agent
        red = llm_agent if dqn_is_blue else dqn_agent

        engine = TickEngine(get_seed_5(), blue_commander=blue, red_commander=red)

        ticks = 0
        while not engine.done and ticks < 10:
            with torch.no_grad():
                result = engine.step()
            ticks += 1
            time.sleep(10)

        final_state = engine.state
        if final_state.battle_winner is None:
            dqn_won = False
            win_type = "timeout"
        else:
            dqn_won = (
                final_state.battle_winner == Side.BLUE and dqn_is_blue
            ) or (
                final_state.battle_winner == Side.RED and not dqn_is_blue
            )

            zones = final_state.zones
            zone_3 = next((z for z in zones if z.zone_id == 3), None)
            total_red = sum(z.red_units for z in zones)
            total_blue = sum(z.blue_units for z in zones)

            if zone_3 and zone_3.side_control in (ZoneControl.RED, ZoneControl.BLUE):
                win_type = "zone3_hold"
            elif total_red == 0 or total_blue == 0:
                win_type = "elimination"
            else:
                win_type = "unknown"

        if dqn_won:
            wins += 1

        win_types[win_type] += 1

        if dqn_is_blue:
            blue_games += 1
            if dqn_won:
                blue_wins += 1
        else:
            red_games += 1
            if dqn_won:
                red_wins += 1

        print(
            f"Episode {episode:2d} | "
            f"Role: {'BLUE' if dqn_is_blue else 'RED'} | "
            f"Winner: {final_state.battle_winner} | "
            f"Ticks: {ticks} | "
            f"DQN Win: {dqn_won}"
        )

    rate = wins / n_episodes
    low, high = wilson_score(rate, n_episodes)

    summary = {
        "checkpoint": checkpoint_path,
        "exists": os.path.exists(checkpoint_path),
        "episodes": n_episodes,
        "wins": wins,
        "games": n_episodes,
        "win_rate": rate,
        "win_rate_ci": [low, high],
        "blue_wins": blue_wins,
        "blue_games": blue_games,
        "red_wins": red_wins,
        "red_games": red_games,
        "win_types": dict(win_types),
    }

    print(f"DQN Win Rate: {rate:.1%} [{low:.1%}, {high:.1%}]")
    print(f"As Blue: {blue_wins}/{blue_games}")
    print(f"As Red: {red_wins}/{red_games}")

    return summary


def main():
    settings = Settings()
    model_name = settings.model_1

    checkpoint_paths = [
        "checkpoint_dqn_ep_1.pt",
        "checkpoint_dqn_ep_100.pt",
        "checkpoint_dqn_ep_200.pt",
        "checkpoint_dqn_ep_500.pt",
        "checkpoint_dqn_ep_1000.pt",
        "checkpoint_dqn_ep_1500.pt",
        "checkpoint_dqn_ep_2000.pt",
    ]

    results = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "seed": "seed_5",
        "checkpoints": [],
    }

    results_dir = pathlib.Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "dqn_vs_llm_seed5_results.json"

    n_episodes = 1

    for checkpoint_path in checkpoint_paths:
        try:
            summary = evaluate_checkpoint(checkpoint_path, n_episodes, model_name)
        except Exception as exc:
            summary = {
                "checkpoint": checkpoint_path,
                "error": str(exc),
            }
            print(f"Skipping {checkpoint_path}: {exc}")

        results["checkpoints"].append(summary)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
