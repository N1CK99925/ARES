from config.settings import ADJACENCY
from core.state import BattleState, Side, ZoneControl


def get_visible_zones_for_side(state: BattleState, side: Side) -> set[int]:
    """
    Compute zones visible to a side based on control.
    
    A zone is visible if:
    - The side controls it directly, OR
    - The side controls an adjacent zone
    
    This represents scouting/intelligence presence.
    """
    visible = set()

    for zone in state.zones:
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
                    (z for z in state.zones if z.zone_id == adj_id),
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


def update_enemy_memory(state: BattleState, enemy_memory: dict, side: Side) -> None:
    """
    Update memory for a side based on current visibility.
    
    If enemy is visible in any visible zone, update memory with the
    closest enemy zone and unit count. If not visible, memory persists
    (this is the decay mechanic: stale info remains until refreshed).
    """
    visible_zones = get_visible_zones_for_side(state, side)

    closest_enemy_zone = None
    enemy_units_in_closest = None
    min_distance_to_ref = float("inf")
    ref_zone = 2 if side == Side.BLUE else 4

    for zone in state.zones:
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
        enemy_memory[side] = {
            "zone": closest_enemy_zone,
            "units": enemy_units_in_closest,
            "tick_seen": state.current_tick,
        }
    # else: memory unchanged (stale sighting persists)