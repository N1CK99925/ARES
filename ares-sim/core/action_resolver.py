from typing import NamedTuple

from config.settings import ADJACENCY
from core.attrition import calculate_attrition
from core.state import BattleState, Side, ZoneControl
from agents.models import ActionType


class Delta(NamedTuple):
    """Represents a unit/fuel change to apply to a zone."""
    zone_id: int
    side: Side
    unit_delta: int
    fuel_delta: int
    weapons_delta: int


def compute_attacker_controls_flank(state: BattleState, action_side: Side, target_zone_id: int) -> bool:
    """
    Determine if the attacker controls the flank by checking adjacent zones.
    Attacker controls flank if they control adjacent zones.
    """
    adjacent_zone_ids = ADJACENCY.get(target_zone_id, [])

    for adj_zone_id in adjacent_zone_ids:
        adj_zone = next(
            (z for z in state.zones if z.zone_id == adj_zone_id),
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


def resolve_actions(state: BattleState, all_actions: list) -> tuple:
    """
    Validate all actions and build deltas without mutating state.
    
    Returns:
        pending_deltas: list of Delta objects to apply
        actions_taken: dict of {Side: count}
        actions_rejected: dict of {Side: count}
    """
    pending_deltas = []
    actions_taken = {Side.BLUE: 0, Side.RED: 0}
    actions_rejected = {Side.BLUE: 0, Side.RED: 0}
    cumulative_outflow = {}  # (zone_id, side) -> total units drained this tick

    for action in all_actions:
        if action.action_type == ActionType.HOLD:
            continue

        # Snapshot zone state (read-only)
        source_zone_snapshot = next(
            (z for z in state.zones if z.zone_id == action.source_zone),
            None,
        )
        target_zone_snapshot = next(
            (z for z in state.zones if z.zone_id == action.target_zone),
            None,
        )
        allowed_targets = ADJACENCY.get(action.source_zone, [])
        if source_zone_snapshot is None or target_zone_snapshot is None:
            actions_rejected[action.side] += 1
            continue
        if action.target_zone not in allowed_targets:
            actions_rejected[action.side] += 1
            continue

        # Verify attacker has sufficient units in source zone (including cumulative outflow this tick)
        source_key = (action.source_zone, action.side)
        already_drained = cumulative_outflow.get(source_key, 0)

        if action.side == Side.BLUE:
            available_units = source_zone_snapshot.blue_units - already_drained
            if available_units < action.units_to_move:
                actions_rejected[action.side] += 1
                continue
        else:  # RED
            available_units = source_zone_snapshot.red_units - already_drained
            if available_units < action.units_to_move:
                actions_rejected[action.side] += 1
                continue

        # Action passes validation — increment counter only now
        actions_taken[action.side] += 1
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
            state.blue_fuel
            if action.side == Side.BLUE
            else state.red_fuel
        )

        # Compute attacker_controls_flank from actual adjacency/zone state
        attacker_controls_flank = compute_attacker_controls_flank(
            state, action.side, action.target_zone
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

    return pending_deltas, actions_taken, actions_rejected