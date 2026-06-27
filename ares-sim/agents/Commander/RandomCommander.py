import random
from typing import Any

from agents.Commander.BaseCommander import BaseCommander
from agents.models import (
    ActionType,
    Actions,
    CommanderAction,
    CommanderDecision,
    CommanderMemory,
)
from core.state import CommanderObs


class RandomCommander(BaseCommander):
    """A commander that takes random actions each tick."""

    def __init__(self) -> None:
        super().__init__()

    def decide(self, obs: CommanderObs, memory: Any) -> CommanderDecision:
        actions: list[CommanderAction] = []

        for zone_id, units in obs.own_unit_per_zone.items():
            if units <= 0:
                continue

            # 50% chance to move, 50% chance to hold
            if random.random() < 0.5:
                legal_targets = obs.legal_targets_per_zone.get(zone_id, [])
                if not legal_targets:
                    # No valid targets – hold
                    actions.append(
                        CommanderAction(
                            side=obs.side,
                            source_zone=zone_id,
                            target_zone=zone_id,
                            units_to_move=0,
                            action_type=ActionType.HOLD,
                        )
                    )
                else:
                    target = random.choice(legal_targets)
                    move_units = random.randint(1, units)
                    actions.append(
                        CommanderAction(
                            side=obs.side,
                            source_zone=zone_id,
                            target_zone=target,
                            units_to_move=move_units,
                            action_type=ActionType.MOVE,
                        )
                    )
            else:
                # Hold
                actions.append(
                    CommanderAction(
                        side=obs.side,
                        source_zone=zone_id,
                        target_zone=zone_id,
                        units_to_move=0,
                        action_type=ActionType.HOLD,
                    )
                )

        memory_out = CommanderMemory(
            current_objective="Random probing",
            last_action_summary=f"Random actions for tick {obs.current_tick}",
            tick_of_last_strategy_changed=obs.current_tick,
        )

        self.last_call_outcome = "success"
        self.last_error = None

        return CommanderDecision(actions=Actions(actions=actions), memory=memory_out)
