from core.state import BattleState, Side, ZoneControl
from core.zones import determine_zone_control
from core.action_resolver import Delta


def apply_deltas(state: BattleState, deltas: list[Delta]) -> BattleState:
    """Apply all pending deltas to state in one batch."""
    if not deltas:
        return state

    # Group deltas by zone for efficient batch update
    zone_updates = {}
    blue_fuel_total = state.blue_fuel
    red_fuel_total = state.red_fuel
    blue_weapons_total = state.blue_weapons_remaining
    red_weapons_total = state.red_weapons_remaining

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
    for zone in state.zones:
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
    return state.model_copy(
        update={
            "zones": updated_zones,
            "blue_fuel": max(0, blue_fuel_total),
            "red_fuel": max(0, red_fuel_total),
            "blue_weapons_remaining": max(0, blue_weapons_total),
            "red_weapons_remaining": max(0, red_weapons_total),
        }
    )