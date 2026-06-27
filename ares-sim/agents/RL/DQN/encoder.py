import torch

from core.state import CommanderObs, Side


class ObservationEncoder:
    def encode(self, obs: CommanderObs) -> torch.Tensor:
        features = []

        # Side
        features.append(0.0 if obs.side == Side.RED else 1.0)

        # Current tick
        features.append(float(obs.current_tick))

        # Friendly units in each zone
        for zone_id in sorted(obs.own_unit_per_zone.keys()):
            features.append(float(obs.own_unit_per_zone[zone_id]))

        # Resources
        features.append(float(obs.own_fuel))
        features.append(float(obs.own_weapons_remaining))

        # Enemy information
        features.append(
            float(obs.enemy_last_known_unit_count)
            if obs.enemy_last_known_unit_count is not None
            else -1.0
        )

        features.append(
            float(obs.enemy_last_known_zone)
            if obs.enemy_last_known_zone is not None
            else -1.0
        )

        features.append(
            float(obs.how_many_ticks_ago_enemy_last_seen)
            if obs.how_many_ticks_ago_enemy_last_seen is not None
            else -1.0
        )

        return torch.tensor(
         features,
         dtype=torch.float32,
).unsqueeze(0)
