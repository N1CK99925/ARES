import random
import torch
from agents.Commander.BaseCommander import BaseCommander
from agents.RL.DQN.encoder import ObservationEncoder
from agents.RL.DQN.network import DQNNetwork
from agents.RL.DQN.epsilon import EpsilonScheduler
from agents.models import ActionType, Actions, CommanderAction, CommanderDecision, CommanderMemory
from core.state import CommanderObs


class DQNCommander(BaseCommander):
    def __init__(self, state_dim: int, hidden_dim: int = 128,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.05,
                 epsilon_decay_steps: int = 10_000):
        super().__init__()
        self.network = DQNNetwork(state_dim, hidden_dim)
        self.encoder = ObservationEncoder()
        self.epsilon_scheduler = EpsilonScheduler(epsilon_start, epsilon_end, epsilon_decay_steps)
        self.tick_counter = 0   # used to compute epsilon — increments once per decide() call
        self.side = None

    def _get_action_for_zone(self, zone_id, unit_count, legal_targets, q_values_for_zone, epsilon):
        if random.random() < epsilon:
            # TODO: exploration branch
            #   legal indices: 0 (hold, always) + 1..len(legal_targets) (move, only if unit_count > 0)
            #   random.choice among legal indices
            #   decode index -> ("hold", zone_id) or ("move", self.network.adjacency[zone_id][idx-1])
            action_type, target_zone = ...
        else:
            # exploitation: just delegate to the network's own decode method
            action_type, target_zone = self.network.get_action_for_zone(zone_id, q_values_for_zone, unit_count)

        units_to_move = unit_count if action_type == "move" else 0
        return CommanderAction(
            side=self.side, source_zone=zone_id, target_zone=target_zone,
            units_to_move=units_to_move,
            action_type=ActionType.HOLD if action_type == "hold" else ActionType.MOVE,
        )

    def decide(self, obs: CommanderObs, memory):
        self.side = obs.side
        encoded = self.encoder.encode(obs)
        with torch.no_grad():
            q_values = self.network(encoded)   # dict: zone_1..zone_5

        epsilon = self.epsilon_scheduler.get_epsilon(self.tick_counter)
        self.tick_counter += 1

        actions = []
        for zone_id, unit_count in obs.own_unit_per_zone.items():
            if unit_count <= 0:
                continue
            legal_targets = obs.legal_targets_per_zone.get(zone_id, [])
            q_values_for_zone = q_values[f"zone_{zone_id}"]
            actions.append(self._get_action_for_zone(zone_id, unit_count, legal_targets, q_values_for_zone, epsilon, ))

        # TODO: build CommanderMemory (current_objective, last_action_summary, tick_of_last_strategy_changed)
        memory_out = ...

        return CommanderDecision(actions=Actions(actions=actions), memory=memory_out)

    def train_step(self):
        pass
    def remember(self):
        pass
    def save(self):
        pass
    def load(self):
        pass
