from core.state import BattleState, Side, ZoneControl, ZoneSnapshot
from core.obs import build_obs
from agents.commander import ActionType, Commander
from core.attrition import calculate_attrition
from core.zones import determine_zone_control, update_zone_3_ticks
from core.outcomes import check_win_condition
from config.settings import ADJACENCY
from typing import NamedTuple
import logging


logger = logging.getLogger(__name__)


class Delta(NamedTuple):
    """Represents a unit/fuel change to apply to a zone."""
    zone_id: int
    side: Side
    unit_delta: int
    fuel_delta: int
    weapons_delta: int


class TickEngine:
    def __init__(self, state: BattleState, blue_commander, red_commander):
        self.state = state
        self.blue = blue_commander
        self.red = red_commander

    def _compute_attacker_controls_flank(self, action_side: Side, target_zone_id: int) -> bool:
        """
        Determine if the attacker controls the flank by checking adjacent zones.
        Attacker controls flank if they control adjacent zones.
        """
        adjacent_zone_ids = ADJACENCY.get(target_zone_id, [])
        
        for adj_zone_id in adjacent_zone_ids:
            adj_zone = next(
                (z for z in self.state.zones if z.zone_id == adj_zone_id),
                None,
            )
            if adj_zone is None:
                continue
            
            # Check if attacker controls this adjacent zone
            if action_side == Side.BLUE:
                if adj_zone.side_control == ZoneControl.BLUE:
                    return True
            else:  # RED
                if adj_zone.side_control == ZoneControl.RED:
                    return True
        
        return False

    def run(self, max_ticks) -> BattleState:
        while self.state.is_engagement_active and self.state.current_tick < max_ticks:
            # Collect actions from both sides
            blue_obs = build_obs(self.state, Side.BLUE)
            red_obs = build_obs(self.state, Side.RED)

            blue_actions = self.blue.decide(blue_obs)
            red_actions = self.red.decide(red_obs)
            all_actions = blue_actions.actions + red_actions.actions

            pending_deltas = []
            actions_taken_per_side = {Side.BLUE: 0, Side.RED: 0}

            # Process all actions without mutating state
            for action in all_actions:
                if action.action_type == ActionType.HOLD:
                    continue

                actions_taken_per_side[action.side] += 1

                # Snapshot zone state (read-only)
                source_zone_snapshot = next(
                    (z for z in self.state.zones if z.zone_id == action.source_zone),
                    None,
                )
                target_zone_snapshot = next(
                    (z for z in self.state.zones if z.zone_id == action.target_zone),
                    None,
                )
                if source_zone_snapshot is None or target_zone_snapshot is None:
                    continue

                # Verify attacker has sufficient units in source zone
                if action.side == Side.BLUE:
                    if source_zone_snapshot.blue_units < action.units_to_move:
                        continue
                else:  # RED
                    if source_zone_snapshot.red_units < action.units_to_move:
                        continue

                # Get attacker and defender units from snapshot
                attacker_units = action.units_to_move
                defender_units = (
                    target_zone_snapshot.red_units
                    if action.side == Side.BLUE
                    else target_zone_snapshot.blue_units
                )
                
                # Get attacker fuel
                attacker_fuel = (
                    self.state.blue_fuel
                    if action.side == Side.BLUE
                    else self.state.red_fuel
                )

                # Compute attacker_controls_flank from actual adjacency/zone state
                attacker_controls_flank = self._compute_attacker_controls_flank(
                    action.side, action.target_zone
                )

                # Compute attacker_controls_zone_3
                attacker_controls_zone_3 = (
                    target_zone_snapshot.zone_id == 3
                    and (
                        (action.side == Side.BLUE and target_zone_snapshot.side_control == ZoneControl.BLUE)
                        or (action.side == Side.RED and target_zone_snapshot.side_control == ZoneControl.RED)
                    )
                )

                # Calculate attrition
                attrition_result = calculate_attrition(
                    attacker_units=attacker_units,
                    defender_units=defender_units,
                    attacker_fuel=attacker_fuel,
                    attacker_controls_flank=attacker_controls_flank,
                    attacker_controls_zone_3=attacker_controls_zone_3,
                )

                # Classic wargame movement: units relocate from source to target
                attacker_survivors = attacker_units - attrition_result.attacker_losses
                opponent_side = Side.RED if action.side == Side.BLUE else Side.BLUE

                # Delta 1: Remove units from source zone
                pending_deltas.append(
                    Delta(
                        zone_id=action.source_zone,
                        side=action.side,
                        unit_delta=-attacker_units,
                        fuel_delta=-attrition_result.attacker_fuel_penalty,
                        weapons_delta=-attrition_result.attacker_weapons_consumed,
                    )
                )

                # Delta 2: Add surviving attacker units to target zone
                pending_deltas.append(
                    Delta(
                        zone_id=action.target_zone,
                        side=action.side,
                        unit_delta=attacker_survivors,
                        fuel_delta=0,
                        weapons_delta=0,
                    )
                )

                # Delta 3: Apply defender losses to target zone
                pending_deltas.append(
                    Delta(
                        zone_id=action.target_zone,
                        side=opponent_side,
                        unit_delta=-attrition_result.defender_losses,
                        fuel_delta=0,
                        weapons_delta=0,
                    )
                )

            # Apply all pending_deltas in batch (simultaneous resolution)
            prev_zone_3_control = next(
                (z.side_control for z in self.state.zones if z.zone_id == 3),
                None,
            )
            self.state = self._apply_deltas(pending_deltas)

            # Log tick result
            logger.info(
                f"Tick {self.state.current_tick}: "
                f"Blue actions={actions_taken_per_side[Side.BLUE]}, "
                f"Red actions={actions_taken_per_side[Side.RED]}, "
                f"Deltas applied={len(pending_deltas)}, "
                f"Zone control: {[(z.zone_id, z.side_control) for z in self.state.zones]}"
            )

            # Update zone 3 tick counter
            zone_3 = next((z for z in self.state.zones if z.zone_id == 3), None)
            if zone_3:
                new_consecutive_ticks = update_zone_3_ticks(
                    current_control=zone_3.side_control,
                    previous_control=prev_zone_3_control,
                    consecutive_ticks=self.state.zone_3_consecutive_ticks,
                )
                self.state = self.state.model_copy(
                    update={"zone_3_consecutive_ticks": new_consecutive_ticks}
                )
            
            # Check win condition
            check_win_condition(state=self.state)
            
            # Advance tick
            self.state = self.state.model_copy(
                update={"current_tick": self.state.current_tick + 1}
            )

        return self.state

    def _apply_deltas(self, deltas: list[Delta]) -> BattleState:
        """Apply all pending deltas to state in one batch."""
        if not deltas:
            return self.state

        # Group deltas by zone for efficient batch update
        zone_updates = {}
        blue_fuel_total = self.state.blue_fuel
        red_fuel_total = self.state.red_fuel
        blue_weapons_total = self.state.blue_weapons_remaining
        red_weapons_total = self.state.red_weapons_remaining

        for delta in deltas:
            if delta.zone_id not in zone_updates:
                zone_updates[delta.zone_id] = {Side.BLUE: 0, Side.RED: 0}
            
            zone_updates[delta.zone_id][delta.side] += delta.unit_delta

            # Apply fuel deltas
            if delta.side == Side.BLUE:
                blue_fuel_total += delta.fuel_delta
                blue_weapons_total += delta.weapons_delta
            else:
                red_fuel_total += delta.fuel_delta
                red_weapons_total += delta.weapons_delta

        # Rebuild zones with updated unit counts
        updated_zones = []
        for zone in self.state.zones:
            if zone.zone_id in zone_updates:
                blue_units = max(0, zone.blue_units + zone_updates[zone.zone_id][Side.BLUE])
                red_units = max(0, zone.red_units + zone_updates[zone.zone_id][Side.RED])
                
                # Recalculate zone control based on updated units
                new_control = determine_zone_control(blue_units, red_units)
                
                updated_zones.append(
                    zone.model_copy(
                        update={
                            "blue_units": blue_units,
                            "red_units": red_units,
                            "side_control": new_control,
                        }
                    )
                )
            else:
                updated_zones.append(zone)

        # Return updated state with all deltas applied
        return self.state.model_copy(
            update={
                "zones": updated_zones,
                "blue_fuel": max(0, blue_fuel_total),
                "red_fuel": max(0, red_fuel_total),
                "blue_weapons_remaining": max(0, blue_weapons_total),
                "red_weapons_remaining": max(0, red_weapons_total),
            }
        )





