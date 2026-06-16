
from core.state import ZoneControl
from config.settings import ZONE_CONTEST_RATIO


def determine_zone_control(
    red_units: int,
    blue_units: int,
) -> ZoneControl:
    # No forces present
    if red_units == 0 and blue_units == 0:
        return ZoneControl.NEUTRAL

    # One side completely absent
    if red_units == 0:
        return ZoneControl.BLUE

    if blue_units == 0:
        return ZoneControl.RED

    # Relative force comparison
    smaller_force = min(red_units, blue_units)
    larger_force = max(red_units, blue_units)

    ratio = smaller_force / larger_force

    # Forces are close enough that neither side has control
    if ratio >= ZONE_CONTEST_RATIO:
        return ZoneControl.CONTESTED

    # Red has clear superiority
    if red_units > blue_units:
        return ZoneControl.RED

    # Blue has clear superiority
    return ZoneControl.BLUE

def update_zone_3_ticks(current_control: ZoneControl , previous_control: ZoneControl
                        ,consecutive_ticks: int) -> int:
    
    if current_control in (
    ZoneControl.NEUTRAL,
    ZoneControl.CONTESTED):
        return 0
    if current_control ==previous_control:
        return consecutive_ticks + 1
    return 1
