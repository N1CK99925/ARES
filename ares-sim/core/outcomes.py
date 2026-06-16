from core.state import BattleState, Side, ZoneControl

from config.settings import ZONE_3_WIN_THRESHOLD




def check_win_condition(state: BattleState) -> BattleState:
    zone3 = next(
            zone for zone in state.zones
            if zone.zone_id == 3             
            )
    if ( zone3.side_control in (
        ZoneControl.RED, ZoneControl.BLUE
        )
        and state.zone_3_consecutive_ticks >= ZONE_3_WIN_THRESHOLD):

        winner = (
                Side.RED
                if zone3.side_control == ZoneControl.RED
                else Side.BLUE
                )
        return state.model_copy(
                update={
                    "is_engagement_active" : False,
                    "battle_winner" : winner
            
            })
    total_red_units = sum(
        zone.red_units
        for zone in state.zones
    )

    total_blue_units = sum(
        zone.blue_units
        for zone in state.zones
    )

    if total_red_units == 0:
        return state.model_copy(
            update={
                "is_engagement_active": False,
                "battle_winner": Side.BLUE,
            }
        )

    if total_blue_units == 0:
        return state.model_copy(
            update={
                "is_engagement_active": False,
                "battle_winner": Side.RED,
            }
        )

    # No win condition met
    return state

    
