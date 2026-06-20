import pathlib
from core.state import BattleState, Side, ZoneControl

from core.obs import build_obs
from agents.models import ActionType, TickLogEntry
from agents.Commander.BaseCommander import BaseCommander
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
    def __init__(
        self,
        state: BattleState,
        blue_commander: BaseCommander,
        red_commander: BaseCommander,
        log_path: str | pathlib.Path | None = None,
    ):
        self.state = state
        self.blue = blue_commander
        self.red = red_commander
        self.log_path = pathlib.Path(log_path) if log_path else None
        self.tick_log: list[TickLogEntry] = []
        # Enemy memory: {Side -> {zone, units, tick_seen}}
        # Persists across ticks; updated only when enemy visible in controlled zones
        self.enemy_memory: dict[Side, dict[str, int | None]] = {
            Side.BLUE: {"zone": None, "units": None, "tick_seen": None},
            Side.RED: {"zone": None, "units": None, "tick_seen": None},
        }

    def _get_visible_zones_for_side(self, side: Side) -> set[int]:
        """Compute zones visible to a side based on control.
        
        A zone is visible if:
        - The side controls it directly, OR
        - The side controls an adjacent zone
        
        This represents scouting/intelligence presence.
        """
        visible = set()
        
        for zone in self.state.zones:
            # Check if we control this zone
            if side == Side.BLUE and zone.side_control == ZoneControl.BLUE:
                visible.add(zone.zone_id)
            elif side == Side.RED and zone.side_control == ZoneControl.RED:
                visible.add(zone.zone_id)
            else:
                # Check if we control an adjacent zone
                adjacent_zone_ids = ADJACENCY.get(zone.zone_id, [])
                for adj_id in adjacent_zone_ids:
                    adj_zone = next(
                        (z for z in self.state.zones if z.zone_id == adj_id),
                        None,
                    )
                    if adj_zone is None:
                        continue
                    
                    if side == Side.BLUE and adj_zone.side_control == ZoneControl.BLUE:
                        visible.add(zone.zone_id)
                        break
                    elif side == Side.RED and adj_zone.side_control == ZoneControl.RED:
                        visible.add(zone.zone_id)
                        break
        
        return visible
    
    def _update_enemy_memory(self, side: Side) -> None:
        """Update memory for a side based on current visibility.
        
        If enemy is visible in any visible zone, update memory with the
        closest enemy zone and unit count. If not visible, memory persists
        (this is the decay mechanic: stale info remains until refreshed).
        """
        visible_zones = self._get_visible_zones_for_side(side)
        
        # Find enemy in any visible zone
        # enemy_side = Side.RED if side == Side.BLUE else Side.BLUE
        
        closest_enemy_zone = None
        enemy_units_in_closest = None
        min_distance_to_ref = float("inf")
        ref_zone = 2 if side == Side.BLUE else 4
        
        for zone in self.state.zones:
            if zone.zone_id not in visible_zones:
                continue
            
            enemy_count = (
                zone.red_units if side == Side.BLUE else zone.blue_units
            )
            
            if enemy_count > 0:
                distance = abs(zone.zone_id - ref_zone)
                if distance < min_distance_to_ref:
                    min_distance_to_ref = distance
                    closest_enemy_zone = zone.zone_id
                    enemy_units_in_closest = enemy_count
        
        # If enemy found in visible zones, update memory
        if closest_enemy_zone is not None:
            self.enemy_memory[side] = {
                "zone": closest_enemy_zone,
                "units": enemy_units_in_closest,
                "tick_seen": self.state.current_tick,
            }
        # else: memory unchanged (stale sighting persists)

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

    def run(self, max_ticks: int) -> BattleState:
        from agents.models import CommanderMemory

        # Seed initial memory for each side — commanders update and return
        # new memory each tick; we thread it forward.
        blue_memory = CommanderMemory(
            current_objective="",
            last_action_summary="",
            tick_of_last_strategy_changed=None,
        )
        red_memory = CommanderMemory(
            current_objective="",
            last_action_summary="",
            tick_of_last_strategy_changed=None,
        )

        while self.state.is_engagement_active and self.state.current_tick < max_ticks:
            # Update enemy memory for both sides based on current zone control
            self._update_enemy_memory(Side.BLUE)
            self._update_enemy_memory(Side.RED)

            # Collect decisions from both sides (obs + memory in, decision + new memory out)
            blue_obs = build_obs(self.state, Side.BLUE, self.enemy_memory[Side.BLUE])
            red_obs = build_obs(self.state, Side.RED, self.enemy_memory[Side.RED])

            blue_decision = self.blue.decide(blue_obs, blue_memory)
            blue_outcome = self.blue.last_call_outcome
            red_decision = self.red.decide(red_obs, red_memory)
            red_outcome = self.red.last_call_outcome

            # Thread updated memory forward to the next tick
            blue_memory = blue_decision.memory
            red_memory = red_decision.memory

            # Build log entries for this tick (before actions are resolved)
            for side, decision, outcome in [
                (Side.BLUE, blue_decision, blue_outcome),
                (Side.RED,  red_decision,  red_outcome),
            ]:
                self.tick_log.append(TickLogEntry(
                    tick=self.state.current_tick,
                    side=side,
                    call_outcome=outcome,
                    actions=decision.actions,
                    current_objective=decision.memory.current_objective,
                    last_action_summary=decision.memory.last_action_summary,
                    tick_of_last_strategy_changed=decision.memory.tick_of_last_strategy_changed,
                ))

            # BUG FIX: .actions on a CommanderDecision is an Actions model;
            # .actions.actions is the underlying list[CommanderAction].
            all_actions = (
                blue_decision.actions.actions + red_decision.actions.actions
            )

            pending_deltas = []
            actions_taken_per_side = {Side.BLUE: 0, Side.RED: 0}
            actions_rejected_per_side = {Side.BLUE: 0, Side.RED: 0}
            cumulative_outflow = {}  # Track (zone_id, side) -> total units drained this tick

            # Process all actions without mutating state
            for action in all_actions:
                if action.action_type == ActionType.HOLD:
                    continue

                # Snapshot zone state (read-only)
                source_zone_snapshot = next(
                    (z for z in self.state.zones if z.zone_id == action.source_zone),
                    None,
                )
                target_zone_snapshot = next(
                    (z for z in self.state.zones if z.zone_id == action.target_zone),
                    None,
                )
                allowed_targets = ADJACENCY.get(action.source_zone,[])
                if source_zone_snapshot is None or target_zone_snapshot is None:
                    actions_rejected_per_side[action.side] += 1
                    continue
                if action.target_zone not in allowed_targets:
                    actions_rejected_per_side[action.side] += 1
                    continue

                # Verify attacker has sufficient units in source zone (including cumulative outflow this tick)
                source_key = (action.source_zone, action.side)
                already_drained = cumulative_outflow.get(source_key, 0)
                
                if action.side == Side.BLUE:
                    available_units = source_zone_snapshot.blue_units - already_drained
                    if available_units < action.units_to_move:
                        actions_rejected_per_side[action.side] += 1
                        continue
                else:  # RED
                    available_units = source_zone_snapshot.red_units - already_drained
                    if available_units < action.units_to_move:
                        actions_rejected_per_side[action.side] += 1
                        continue

                # Action passes validation — increment counter only now
                actions_taken_per_side[action.side] += 1
                cumulative_outflow[source_key] = already_drained + action.units_to_move

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
                f"Blue actions executed={actions_taken_per_side[Side.BLUE]}, rejected={actions_rejected_per_side[Side.BLUE]}, "
                f"Red actions executed={actions_taken_per_side[Side.RED]}, rejected={actions_rejected_per_side[Side.RED]}, "
                f"Deltas applied={len(pending_deltas)}, "
                f"Zone control: {[(z.zone_id, z.side_control) for z in self.state.zones]}"
            )

            # Update zone 3 tick counter
            zone_3 = next((z for z in self.state.zones if z.zone_id == 3), None)
            if zone_3:
                new_consecutive_ticks = update_zone_3_ticks(
                    current_control=zone_3.side_control,
                    previous_control=prev_zone_3_control or zone_3.side_control,
                    consecutive_ticks=self.state.zone_3_consecutive_ticks,
                )
                self.state = self.state.model_copy(
                    update={"zone_3_consecutive_ticks": new_consecutive_ticks}
                )
            
            # Check win condition
            self.state = check_win_condition(state=self.state)
            
            # Advance tick
            self.state = self.state.model_copy(
                update={"current_tick": self.state.current_tick + 1}
            )

        # Write tick log as JSONL if a path was provided
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("w", encoding="utf-8") as fh:
                for entry in self.tick_log:
                    fh.write(entry.model_dump_json() + "\n")

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
                new_control = determine_zone_control(red_units, blue_units)
                
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





