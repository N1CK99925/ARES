import random
import torch
from agents.Commander.BaseCommander import BaseCommander
from agents.RL.DQN.encoder import ObservationEncoder
from agents.RL.DQN.network import DQNNetwork
from agents.RL.DQN.epsilon import EpsilonScheduler
from agents.RL.DQN.replay_buffer import ReplayBuffer
from agents.models import ActionType, Actions, CommanderAction, CommanderDecision, CommanderMemory
from core.state import CommanderObs
from agents.RL.DQN.trainer import Trainer

class DQNCommander(BaseCommander):
    def __init__(self, state_dim: int, hidden_dim: int = 128,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.05,
                 epsilon_decay_steps: int = 10_000, trainer = None):
        super().__init__()
        self.network = DQNNetwork(state_dim, hidden_dim)
        self.encoder = ObservationEncoder()
        self.epsilon_scheduler = EpsilonScheduler(epsilon_start, epsilon_end, epsilon_decay_steps)
        self.tick_counter = 0   # used to compute epsilon — increments once per decide() call
        self.side = None
        self.last_action_indices = {}
        self.replay_buffer = ReplayBuffer(capacity=1000)
        self.trainer = Trainer
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

    def train_step(self):
        """
    Triggers optimization step only if replay buffer contains enough data.
    """
        if self.trainer is not None and len(self.replay_buffer) >= self.trainer.batch_size:
            return self.trainer.train_step() 
        return None
    def remember(self,obs, action_indices, reward, next_obs, done):
        self.replay_buffer.push(
            obs=obs,
            action=action_indices,
            reward=reward, 
            next_obs=next_obs, 
            done=done)
        
    def save(self, path: str):
        """Saves weights to disk."""
        torch.save(self.network.state_dict(), path)

    def load(self, path: str):
        """Loads weights from disk."""
        self.network.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        self.network.eval()  
