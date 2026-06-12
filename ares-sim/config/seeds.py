
from core.state import BattleState,ZoneSnapshot,ZoneControl

def get_seed_1():
    zone_1 = ZoneSnapshot(
        zone_id=1,
        red_units=0,
        blue_units=100,
        side_control=ZoneControl.BLUE
        )
    zone_2 = ZoneSnapshot(
        zone_id=2,
        red_units=0,
        blue_units=50,
        side_control=ZoneControl.BLUE
        )
    zone_3 = ZoneSnapshot(
        zone_id=3,
        red_units=0,
        blue_units=0,
        side_control=ZoneControl.NEUTRAL
        )
    zone_4 = ZoneSnapshot(
        zone_id=4,
        red_units=50,
        blue_units=0,
        side_control=ZoneControl.RED
        )

    zone_5 = ZoneSnapshot(
        zone_id=5,
        red_units=100,
        blue_units=0,
        side_control=ZoneControl.RED
        )
    zones = [zone_1, zone_2,zone_3,zone_4,zone_5]
    # 150 everything because 150 units
    battle_state = BattleState(
            current_tick=0,
            zones = zones,
            red_fuel=150,
            blue_fuel=150,
            red_weapons_remaining=150,
            blue_weapons_remaining=150,
            is_engagement_active=True,
            battle_winner=None,
            zone_3_consecutive_ticks=0
            )
    return battle_state






def get_seed_2():
   raise NotImplementedError


def get_seed_3():
    raise NotImplementedError

def get_seed_4():
    raise NotImplementedError

def get_seed_5():
    raise NotImplementedError

