import random
import torch
from agents.Commander.BaseCommander import BaseCommander
from agents.RL.DQN.encoder import ObservationEncoder
from agents.RL.DQN.network import DQNNetwork
from agents.RL.DQN.epsilon import EpsilonScheduler

from agents.models import ActionType, Actions, CommanderAction, CommanderDecision, CommanderMemory
from core.state import CommanderObs


class DQNCommander(BaseCommander):
    """Pure policy commander: receives observation, returns decision.

    Owns no replay buffer, trainer, or optimizer.
    The network, encoder, and epsilon scheduler are injected from outside.
    """

    def __init__(
        self,
        network: DQNNetwork,
        encoder: ObservationEncoder,
        epsilon_scheduler: EpsilonScheduler,
    ):
        super().__init__()
        self.network = network
        self.encoder = encoder
        self.epsilon_scheduler = epsilon_scheduler
        self.tick_counter = 0   # used to compute epsilon — increments once per decide() call
        self.side = None
        self.last_action_indices: dict[int, int] = {}
        self.last_q_values: dict[str, list[float]] = {}

    def _get_action_for_zone(self, zone_id, unit_count, legal_targets, q_values_for_zone, epsilon):
        if random.random() < epsilon:
            choices = ["hold"] + legal_targets
            choice = random.choice(choices)
            action_idx = choices.index(choice)
            if choice == "hold":
                action_type = "hold"
                target_zone = zone_id
            else:
                action_type = "move"
                target_zone = choice
        else:
            action_type, target_zone = self.network.get_action_for_zone(
                zone_id, q_values_for_zone, unit_count
            )
            if action_type == "hold":
                action_idx = 0
            else:
                action_idx = self.network.adjacency[zone_id].index(target_zone) + 1

        units_to_move = unit_count if action_type == "move" else 0
        return action_idx, CommanderAction(
            side=self.side,
            source_zone=zone_id,
            target_zone=target_zone,
            units_to_move=units_to_move,
            action_type=ActionType.HOLD if action_type == "hold" else ActionType.MOVE,
    )

    def decide(self, obs: CommanderObs, memory):
        self.side = obs.side
        encoded = self.encoder.encode(obs)
        with torch.no_grad():
            q_values = self.network(encoded)   # dict: zone_1..zone_5
            # Store raw Q-values for trace/eval purposes (convert to float lists)
            self.last_q_values = {
                k: v.squeeze(0).tolist() for k, v in q_values.items()
            }

        epsilon = self.epsilon_scheduler.get_epsilon(self.tick_counter)
        self.tick_counter += 1

        actions = []
        action_indices = {}
        for zone_id, unit_count in obs.own_unit_per_zone.items():
            if unit_count <= 0:
                continue
            legal_targets = obs.legal_targets_per_zone.get(zone_id, [])
            q_values_for_zone = q_values[f"zone_{zone_id}"]
            idx, action = self._get_action_for_zone(zone_id, unit_count, legal_targets, q_values_for_zone, epsilon)
            actions.append(action)
            action_indices[zone_id] = idx
        self.last_action_indices = action_indices
        memory_out = CommanderMemory(
            current_objective=memory.current_objective,
            last_action_summary=str(actions),
            tick_of_last_strategy_changed=memory.tick_of_last_strategy_changed
)

        return CommanderDecision(actions=Actions(actions=actions), memory=memory_out)

    def save(self, path: str):
        """Saves weights to disk."""
        torch.save(self.network.state_dict(), path)

    def load(self, path: str):
        """Loads weights from disk."""
        self.network.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        self.network.eval()  
