from core.state import BattleState, CommanderObs, Side


def build_obs(state: BattleState, side: Side) -> CommanderObs:
    if side == Side.BLUE:
        own_units = {z.zone_id: z.blue_units for z in state.zones}
        own_fuel = state.blue_fuel
        own_weapons = state.blue_weapons_remaining
        last_enemy_known_zone = None
        last_enemy_known_units = None

    else:
        own_units = {z.zone_id: z.red_units for z in state.zones}
        own_fuel = state.red_fuel
        own_weapons = state.red_weapons_remaining
        last_enemy_known_zone = None
        last_enemy_known_units = None

    # find enemy zones
    # sort by distance from reference zone (2 for BLUE, 4 for RED)
    # if found: set last_enemy_known_zone and last_enemy_known_units from closest
    # if not found: leave as None
    ref_zone = 2 if side == Side.BLUE else 4
    enemy_candidates: dict[int, int] = {}
    for z in state.zones:
        enemy_count = z.red_units if side == Side.BLUE else z.blue_units
        if enemy_count > 0:
            enemy_candidates[z.zone_id] = enemy_count

    if enemy_candidates:
        closest_zone = min(enemy_candidates.keys(), key=lambda zid: abs(zid - ref_zone))
        last_enemy_known_zone = closest_zone
        last_enemy_known_units = enemy_candidates[closest_zone]

    how_many_ticks_ago_enemy_last_seen = None

    return CommanderObs(
        side=side,
        current_tick=state.current_tick,
        own_unit_per_zone=own_units,
        own_fuel=own_fuel,
        own_weapons_remaining=own_weapons,
        enemy_last_known_unit_count=last_enemy_known_units,
        enemy_last_known_zone=last_enemy_known_zone,
        how_many_ticks_ago_enemy_last_seen=how_many_ticks_ago_enemy_last_seen,
    )




    


