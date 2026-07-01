from core.state import Side, ZoneControl
def compute_reward(prev_state, post_state):
    """
    prev_state, post_state: BattleState snapshots, before/after this tick's deltas
    returns: float, reward from Blue's perspective (Red = -this)
    """
    reward = 0.0

    # Game-ending condition
    if post_state.battle_winner is not None:
        reward += 10.0 if post_state.battle_winner == Side.BLUE else -10.0
        return reward

    prev_streak = prev_state.zone_3_consecutive_ticks
    post_streak = post_state.zone_3_consecutive_ticks

    prev_zone_3 = next((z for z in prev_state.zones if z.zone_id == 3), None)
    post_zone_3 = next((z for z in post_state.zones if z.zone_id == 3), None)

    # Streak progression / break logic (Blue's perspective)
    if post_streak > prev_streak:
        if post_zone_3 and post_zone_3.side_control == ZoneControl.BLUE:
            reward += 1.0
        elif post_zone_3 and post_zone_3.side_control == ZoneControl.RED:
            reward -= 1.0
    elif post_streak < prev_streak:
        if prev_zone_3 and prev_zone_3.side_control == ZoneControl.BLUE:
            reward -= float(prev_streak)
        elif prev_zone_3 and prev_zone_3.side_control == ZoneControl.RED:
            reward += float(prev_streak)

    return reward