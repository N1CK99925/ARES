"""
Evaluation harness for ARES DQN.
Runs a trained checkpoint against RandomCommander across multiple seeds.
Reports per-seed metrics + pooled evaluation metrics.
"""

import sys
import pathlib
import math
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "ares-sim"))

import torch

from config.seeds import (
    get_seed_1,
    get_seed_2,
    get_seed_3,
    get_seed_4
)

from core.tick import TickEngine
from core.state import Side, ZoneControl

from agents.Commander.DQNCommander import DQNCommander
from agents.Commander.RandomCommander import RandomCommander

from agents.RL.DQN.network import DQNNetwork
from agents.RL.DQN.encoder import ObservationEncoder
from agents.RL.DQN.epsilon import EpsilonScheduler


def wilson_score(
    p: float,
    n: int,
    z: float = 1.96
) -> tuple[float, float]:
    """
    Calculate 95% Wilson score interval.
    """

    if n == 0:
        return 0.0, 0.0

    denominator = 1 + z**2 / n

    centre = (
        p + z**2 / (2 * n)
    )

    deviation = math.sqrt(
        (p * (1 - p) + z**2 / (4 * n)) / n
    )

    lower = (
        centre - z * deviation
    ) / denominator

    upper = (
        centre + z * deviation
    ) / denominator

    return max(0, lower), min(1, upper)



def evaluate_seed(
    seed_name,
    seed_fn,
    network,
    n_episodes,
    trace_mode=False
):

    print("\n" + "-" * 35)
    print(f"Evaluating {seed_name}")
    print("-" * 35)


    wins = 0
    win_types = Counter()

    blue_wins = 0
    blue_games = 0

    red_wins = 0
    red_games = 0


    encoder = ObservationEncoder()

    greedy_eps = EpsilonScheduler(
        start=0.0,
        end=0.0,
        decay_steps=1
    )


    for episode in range(1, n_episodes + 1):

        # Fresh commander each episode
        dqn_agent = DQNCommander(
            network=network,
            encoder=encoder,
            epsilon_scheduler=greedy_eps
        )

        random_agent = RandomCommander()


        dqn_is_blue = episode % 2 == 1


        blue = (
            dqn_agent
            if dqn_is_blue
            else random_agent
        )

        red = (
            random_agent
            if dqn_is_blue
            else dqn_agent
        )


        engine = TickEngine(
            seed_fn(),
            blue_commander=blue,
            red_commander=red
        )


        if trace_mode:
            print(
                f"\nEpisode {episode} "
                f"DQN={'BLUE' if dqn_is_blue else 'RED'}"
            )


        ticks = 0


        while (
            not engine.done
            and ticks < 1000
        ):

            with torch.no_grad():
                result = engine.step()

            ticks += 1


            if trace_mode:

                obs = (
                    result.blue_obs
                    if dqn_is_blue
                    else result.red_obs
                )

                print(
                    f"Tick {ticks}"
                )

                print(
                    "Units:",
                    obs.own_unit_per_zone
                )

                print(
                    "Actions:",
                    dqn_agent.last_action_indices
                )


        final_state = engine.state


        if final_state.battle_winner is None:

            dqn_won = False
            win_type = "timeout"

        else:

            dqn_won = (
                final_state.battle_winner == Side.BLUE
                and dqn_is_blue
            ) or (
                final_state.battle_winner == Side.RED
                and not dqn_is_blue
            )


            zones = final_state.zones


            zone_3 = next(
                (
                    z
                    for z in zones
                    if z.zone_id == 3
                ),
                None
            )


            total_red = sum(
                z.red_units
                for z in zones
            )

            total_blue = sum(
                z.blue_units
                for z in zones
            )


            if (
                zone_3
                and zone_3.side_control
                in (
                    ZoneControl.RED,
                    ZoneControl.BLUE
                )
            ):

                win_type = "zone3_hold"


            elif (
                total_red == 0
                or total_blue == 0
            ):

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



    rate = wins / n_episodes

    low, high = wilson_score(
        rate,
        n_episodes
    )


    print(
        f"DQN Win Rate: {rate:.1%}"
        f" [{low:.1%}, {high:.1%}]"
    )


    print(
        f"As Blue: {blue_wins}/{blue_games}"
    )

    print(
        f"As Red: {red_wins}/{red_games}"
    )


    print("Win Types:")

    for k, v in win_types.items():

        print(
            f"  {k}: {v}"
            f" ({v/n_episodes:.1%})"
        )


    return {
        "wins": wins,
        "games": n_episodes,
        "win_types": win_types
    }




def run_evaluation(
    checkpoint_path,
    n_episodes=100,
    trace_mode=False
):

    print("=== ARES DQN Evaluation ===")
    print(
        f"Checkpoint: {checkpoint_path}"
    )


    state_dim = 12
    hidden_dim = 128


    network = DQNNetwork(
        state_dim,
        hidden_dim
    )


    try:

        network.load_state_dict(
            torch.load(
                checkpoint_path,
                map_location="cpu"
            )
        )

    except FileNotFoundError:

        print(
            "Checkpoint missing. "
            "Using random weights."
        )


    network.eval()



    seed_pool = [
        ("seed_1", get_seed_1),
        ("seed_2", get_seed_2),
        ("seed_3", get_seed_3),
        ("seed_4", get_seed_4),
    ]


    total_wins = 0
    total_games = 0

    total_win_types = Counter()



    for name, seed_fn in seed_pool:

        result = evaluate_seed(
            name,
            seed_fn,
            network,
            n_episodes,
            trace_mode
        )


        total_wins += result["wins"]

        total_games += result["games"]

        total_win_types.update(
            result["win_types"]
        )



    rate = total_wins / total_games


    low, high = wilson_score(
        rate,
        total_games
    )


    print("\n" + "=" * 35)

    print("POOLED RESULTS")

    print("=" * 35)


    print(
        f"Overall Win Rate: {rate:.1%}"
        f" [{low:.1%}, {high:.1%}]"
    )


    print(
        f"Games: {total_wins}/{total_games}"
    )


    print("Win Types:")


    for k, v in total_win_types.items():

        print(
            f"  {k}: {v}"
            f" ({v/total_games:.1%})"
        )



if __name__ == "__main__":

    import argparse


    parser = argparse.ArgumentParser(
        description="ARES DQN Eval Harness"
    )


    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoint_dqn_ep_50.pt"
    )


    parser.add_argument(
        "--episodes",
        type=int,
        default=100
    )


    parser.add_argument(
        "--trace",
        action="store_true"
    )


    args = parser.parse_args()


    run_evaluation(
        checkpoint_path=args.ckpt,
        n_episodes=(
            1
            if args.trace
            else args.episodes
        ),
        trace_mode=args.trace
    )
