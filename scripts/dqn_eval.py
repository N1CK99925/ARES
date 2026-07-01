"""
Evaluation harness for ARES DQN.
Runs a trained checkpoint against a baseline (RandomCommander) and reports metrics.
"""

import sys
import pathlib
import math
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "ares-sim"))

import torch
from config.seeds import get_seed_1
from core.tick import TickEngine
from core.state import Side, ZoneControl
from agents.Commander.DQNCommander import DQNCommander
from agents.Commander.RandomCommander import RandomCommander
from agents.RL.DQN.network import DQNNetwork
from agents.RL.DQN.encoder import ObservationEncoder
from agents.RL.DQN.epsilon import EpsilonScheduler

def wilson_score(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Calculate 95% Wilson score interval for binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    denominator = 1 + z**2 / n
    centre_adjusted_probability = p + z**2 / (2 * n)
    adjusted_standard_deviation = math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    lower = (centre_adjusted_probability - z * adjusted_standard_deviation) / denominator
    upper = (centre_adjusted_probability + z * adjusted_standard_deviation) / denominator
    return max(0.0, lower), min(1.0, upper)

def run_evaluation(checkpoint_path: str, n_episodes: int = 100, trace_mode: bool = False):
    print(f"=== ARES DQN Evaluation Harness ===")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Episodes: {n_episodes}")
    print(f"Trace Mode: {trace_mode}")
    print(f"Opponent: RandomCommander")
    print("=" * 35)

    # Load the network
    state_dim = 12
    hidden_dim = 128
    network = DQNNetwork(state_dim, hidden_dim)
    try:
        network.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    except FileNotFoundError:
        print(f"[!] Warning: Checkpoint {checkpoint_path} not found. Evaluating with random weights.")
    network.eval()

    encoder = ObservationEncoder()
    # Greedy scheduler (always 0)
    greedy_eps = EpsilonScheduler(start=0.0, end=0.0, decay_steps=1)
    
    dqn_agent = DQNCommander(network=network, encoder=encoder, epsilon_scheduler=greedy_eps)
    random_agent = RandomCommander()

    # Metrics
    dqn_wins = 0
    win_types = Counter()
    blue_wins = 0
    blue_games = 0
    red_wins = 0
    red_games = 0
    
    for episode in range(1, n_episodes + 1):
        # Alternate sides
        dqn_is_blue = (episode % 2 == 1)
        blue_commander = dqn_agent if dqn_is_blue else random_agent
        red_commander = random_agent if dqn_is_blue else dqn_agent

        state = get_seed_1()
        engine = TickEngine(state, blue_commander=blue_commander, red_commander=red_commander)
        
        if trace_mode:
            print(f"\n--- Episode {episode} Trace (DQN is {'BLUE' if dqn_is_blue else 'RED'}) ---")
        
        ticks = 0
        total_dqn_reward = 0.0
        
        while not engine.done and ticks < 1000:
            result = engine.step()
            ticks += 1
            
            dqn_obs = result.blue_obs if dqn_is_blue else result.red_obs
            dqn_action = result.blue_action if dqn_is_blue else result.red_action
            dqn_reward = result.blue_reward if dqn_is_blue else result.red_reward
            total_dqn_reward += dqn_reward
            
            if trace_mode:
                print(f"[Tick {ticks}] DQN Reward: {dqn_reward:+.1f}")
                print(f"  Obs (Friendly Units): {dqn_obs.own_unit_per_zone}")
                print(f"  Chosen Actions: {dqn_agent.last_action_indices}")
                print(f"  Q-Values: {dqn_agent.last_q_values}")
                print()
                
        # Determine episode results
        final_state = engine.state
        
        if final_state.battle_winner is None:
            win_type = "timeout"
            dqn_won = False
        else:
            if (final_state.battle_winner == Side.BLUE and dqn_is_blue) or \
               (final_state.battle_winner == Side.RED and not dqn_is_blue):
                dqn_won = True
                dqn_wins += 1
            else:
                dqn_won = False
                
            # Figure out win type
            zone_3 = next((z for z in final_state.zones if z.zone_id == 3), None)
            total_red = sum(z.red_units for z in final_state.zones)
            total_blue = sum(z.blue_units for z in final_state.zones)
            
            if total_red == 0 or total_blue == 0:
                win_type = "elimination"
            elif zone_3 and zone_3.side_control in (ZoneControl.RED, ZoneControl.BLUE):
                win_type = "zone3_hold"
            else:
                win_type = "unknown"
                
        win_types[win_type] += 1
        
        if dqn_is_blue:
            blue_games += 1
            if dqn_won: blue_wins += 1
        else:
            red_games += 1
            if dqn_won: red_wins += 1

        if not trace_mode and episode % 10 == 0:
            print(f"Completed {episode}/{n_episodes} episodes...")

    # Aggregate and report
    p_hat = dqn_wins / n_episodes
    ci_lower, ci_upper = wilson_score(p_hat, n_episodes)
    
    blue_p = blue_wins / blue_games if blue_games > 0 else 0.0
    red_p = red_wins / red_games if red_games > 0 else 0.0
    
    print("\n" + "=" * 35)
    print("EVALUATION RESULTS")
    print("=" * 35)
    print(f"DQN Win Rate: {p_hat:.1%} (95% CI: [{ci_lower:.1%}, {ci_upper:.1%}])")
    print(f"  As Blue:    {blue_p:.1%} ({blue_wins}/{blue_games})")
    print(f"  As Red:     {red_p:.1%} ({red_wins}/{red_games})")
    print("\nWin Types:")
    for wt, count in win_types.items():
        print(f"  {wt}: {count} ({count/n_episodes:.1%})")
    print("=" * 35)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ARES DQN Eval Harness")
    parser.add_argument("--ckpt", type=str, default="checkpoint_dqn_ep_50.pt", help="Path to checkpoint file")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to run")
    parser.add_argument("--trace", action="store_true", help="Run a single verbose episode")
    args = parser.parse_args()

    run_evaluation(
        checkpoint_path=args.ckpt,
        n_episodes=1 if args.trace else args.episodes,
        trace_mode=args.trace
    )
