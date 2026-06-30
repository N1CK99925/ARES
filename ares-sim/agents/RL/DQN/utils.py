from core.state import Side
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

    # Streak progression / break logic
    if post_streak > prev_streak:
        reward += 1.0
    elif post_streak < prev_streak:
        reward -= float(prev_streak)

    return reward