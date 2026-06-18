from core.state import BattleState, CommanderObs, Side


def build_obs(
    state: BattleState,
    side: Side,
    enemy_memory: dict[str, int | None],
) -> CommanderObs:
    """Build a commander observation from current state and persistent memory.
    
    Args:
        state: Ground truth battle state
        side: Which side this observation is for (BLUE or RED)
        enemy_memory: Persistent memory dict with keys:
            - "zone": int | None (last known enemy zone)
            - "units": int | None (last known enemy unit count)
            - "tick_seen": int | None (tick when enemy was last observed)
    
    Returns:
        CommanderObs with partial observability based on memory, not ground truth.
    """
    if side == Side.BLUE:
        own_units = {z.zone_id: z.blue_units for z in state.zones}
        own_fuel = state.blue_fuel
        own_weapons = state.blue_weapons_remaining
    else:
        own_units = {z.zone_id: z.red_units for z in state.zones}
        own_fuel = state.red_fuel
        own_weapons = state.red_weapons_remaining

    # Extract memory; if any key missing, treat as no sighting
    last_enemy_known_zone = enemy_memory.get("zone")
    last_enemy_known_units = enemy_memory.get("units")
    tick_seen = enemy_memory.get("tick_seen")
    
    # Compute decay: how many ticks since last sighting?
    if tick_seen is not None:
        how_many_ticks_ago_enemy_last_seen = state.current_tick - tick_seen
    else:
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




    


