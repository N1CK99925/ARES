import torch
import torch.nn as nn
from config.settings import ADJACENCY   # {1:[2], 2:[1,3], 3:[2,4], 4:[3,5], 5:[4]}

class DQNNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Zone heads – each output size = 1 (hold) + len(adjacent zones)
        # We'll store the adjacency for each zone to know which index maps to which target
        self.adjacency = ADJACENCY
        self.num_zones = len(ADJACENCY)   # 5

        self.zone_heads = nn.ModuleDict()
        for zone_id in range(1, self.num_zones + 1):
            out_size = 1 + len(ADJACENCY[zone_id])
            self.zone_heads[f"zone_{zone_id}"] = nn.Linear(hidden_dim, out_size)
            

    def forward(self, state: torch.Tensor):
        """
        state shape: (batch_size, state_dim)
        Returns dict: {
            "zone_1": tensor (batch, out_size_1),
            ...
            "zone_5": tensor (batch, out_size_5)
        }
        """
        features = self.encoder(state)
        outputs = {}
        for zone_id in range(1, self.num_zones + 1):
            outputs[f"zone_{zone_id}"] = self.zone_heads[f"zone_{zone_id}"](features)
        return outputs

    def get_action_for_zone(self, zone_id: int, q_values: torch.Tensor, unit_count: int) -> tuple[str, int]:
        """
        Decode action for a single zone, with masking if zone has no units.

        Args:
            zone_id: zone ID (1..5)
            q_values: tensor shape (1, out_size) for this zone
            unit_count: number of units in this zone (from observation)

        Returns:
            (action_type, target_zone) where action_type is "hold" or "move"
        """
    # Clone and remove batch dimension → shape (out_size,)
        q = q_values.clone().squeeze(0)
        if unit_count == 0:
            mask = torch.zeros_like(q, dtype=torch.bool)
            mask[1:] = True
            q = q.masked_fill(mask, float('-inf'))
        idx = int(q.argmax(dim=0).item())

        if idx == 0:
            return ("hold", zone_id)
        else:
            target = self.adjacency[zone_id][idx - 1]
            return ("move", target)
