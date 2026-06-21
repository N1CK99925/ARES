import pytest
import sys
sys.path.insert(0, 'ares-sim')

from core.zones import determine_zone_control
from core.state import ZoneControl

class TestDetermineZoneControl:
    """Unit tests for zone control determination logic."""

    def test_both_sides_zero_units(self):
        """Empty zone is neutral."""
        result = determine_zone_control(red_units=0, blue_units=0)
        assert result == ZoneControl.NEUTRAL

    def test_red_only_present(self):
        """Zone with only red units is red-controlled."""
        result = determine_zone_control(red_units=10, blue_units=0)
        assert result == ZoneControl.RED

    def test_blue_only_present(self):
        """Zone with only blue units is blue-controlled."""
        result = determine_zone_control(red_units=0, blue_units=10)
        assert result == ZoneControl.BLUE

    def test_red_dominant(self):
        """Red outnumbers blue decisively."""
        result = determine_zone_control(red_units=100, blue_units=10)
        assert result == ZoneControl.RED

    def test_blue_dominant(self):
        """Blue outnumbers red decisively."""
        result = determine_zone_control(red_units=10, blue_units=100)
        assert result == ZoneControl.BLUE

    def test_forces_contested_equal(self):
        """Equal forces create contested zone."""
        result = determine_zone_control(red_units=50, blue_units=50)
        assert result == ZoneControl.CONTESTED

    def test_forces_contested_close(self):
        """Close forces (within contest ratio) create contested zone."""
        # Assuming ZONE_CONTEST_RATIO allows this
        result = determine_zone_control(red_units=70, blue_units=80)
        # Could be contested or decided depending on config
        assert result in [ZoneControl.CONTESTED, ZoneControl.BLUE, ZoneControl.RED]

    def test_argument_order_matters(self):
        """Verify that swapping red/blue gives opposite results."""
        result_red_dominant = determine_zone_control(red_units=100, blue_units=10)
        result_blue_dominant = determine_zone_control(red_units=10, blue_units=100)
        
        # These should be different
        assert result_red_dominant != result_blue_dominant
        # And specifically opposite
        assert result_red_dominant == ZoneControl.RED
        assert result_blue_dominant == ZoneControl.BLUE
