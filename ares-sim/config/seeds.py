from core.state import BattleState, ZoneSnapshot, ZoneControl

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
    zones = [zone_1, zone_2, zone_3, zone_4, zone_5]
    battle_state = BattleState(
        current_tick=0,
        zones=zones,
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
    """
    Case 2: Contested Center Start
    Zone 3 already has 20 units from both sides instead of being empty/neutral.
    Tests if the agent fights for zone-3 control from turn 1.
    """
    zones = [
        ZoneSnapshot(zone_id=1, red_units=0, blue_units=100, side_control=ZoneControl.BLUE),
        ZoneSnapshot(zone_id=2, red_units=0, blue_units=50, side_control=ZoneControl.BLUE),
        ZoneSnapshot(zone_id=3, red_units=20, blue_units=20, side_control=ZoneControl.NEUTRAL),
        ZoneSnapshot(zone_id=4, red_units=50, blue_units=0, side_control=ZoneControl.RED),
        ZoneSnapshot(zone_id=5, red_units=100, blue_units=0, side_control=ZoneControl.RED),
    ]
    return BattleState(
        current_tick=0,
        zones=zones,
        red_fuel=190,  
        blue_fuel=190,
        red_weapons_remaining=190,
        blue_weapons_remaining=190,
        is_engagement_active=True,
        battle_winner=None,
        zone_3_consecutive_ticks=0
    )


def get_seed_3():
    """
    Case 3: Asymmetric Fuel/Weapons
    Normal unit distribution matching seed_1, but Blue is under heavy resource pressure.
    Tests if the agent adapts strategy or blindly executes a script.
    """
    zones = [
        ZoneSnapshot(zone_id=1, red_units=0, blue_units=100, side_control=ZoneControl.BLUE),
        ZoneSnapshot(zone_id=2, red_units=0, blue_units=50, side_control=ZoneControl.BLUE),
        ZoneSnapshot(zone_id=3, red_units=0, blue_units=0, side_control=ZoneControl.NEUTRAL),
        ZoneSnapshot(zone_id=4, red_units=50, blue_units=0, side_control=ZoneControl.RED),
        ZoneSnapshot(zone_id=5, red_units=100, blue_units=0, side_control=ZoneControl.RED),
    ]
    return BattleState(
        current_tick=0,
        zones=zones,
        red_fuel=150,
        blue_fuel=50,  
        red_weapons_remaining=150,
        blue_weapons_remaining=50,  
        is_engagement_active=True,
        battle_winner=None,
        zone_3_consecutive_ticks=0
    )


def get_seed_4():
    """
    Case 4: Reversed Flank Strength
    Mirrors seed_1's flank unit distribution (Zone 1/2 swapped with Zone 4/5).
    Tests generalization ("push weak flank") vs hardcoding ("zone_1 is always strong").
    """
    zones = [
        ZoneSnapshot(zone_id=1, red_units=0, blue_units=50, side_control=ZoneControl.BLUE),    
        ZoneSnapshot(zone_id=2, red_units=0, blue_units=100, side_control=ZoneControl.BLUE),    
        ZoneSnapshot(zone_id=3, red_units=0, blue_units=0, side_control=ZoneControl.NEUTRAL),
        ZoneSnapshot(zone_id=4, red_units=100, blue_units=0, side_control=ZoneControl.RED),   
        ZoneSnapshot(zone_id=5, red_units=50, blue_units=0, side_control=ZoneControl.RED),     
    ]
    return BattleState(
        current_tick=0,
        zones=zones,
        red_fuel=150,
        blue_fuel=150,
        red_weapons_remaining=150,
        blue_weapons_remaining=150,
        is_engagement_active=True,
        battle_winner=None,
        zone_3_consecutive_ticks=0
    )
