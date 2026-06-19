#!/usr/bin/env python3
"""
Sanity test: Run 200 ticks and verify win condition termination logic.
Track which win path fires: zone-3 hold, elimination, or timeout.
"""

import sys
sys.path.insert(0, 'ares-sim')

from core.tick import TickEngine
from config.seeds import get_seed_1
from agents.commander import Commander
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(name)s] %(message)s'
)
logger = logging.getLogger('TEST')

# Dummy commander that always holds
class DummyCommander:
    def __init__(self, side):
        self.side = side
    
    def decide(self, obs):
        from agents.commander import Actions, CommanderAction, ActionType
        # Always hold (no-op actions)
        return Actions(actions=[])

def run_test():
    """Run 200 ticks and report results."""
    state = get_seed_1()
    engine = TickEngine(state, DummyCommander('BLUE'), DummyCommander('RED'))
    
    logger.info("=" * 70)
    logger.info(f"Starting 200-tick simulation with ZONE_3_WIN_THRESHOLD=6")
    logger.info("=" * 70)
    logger.info(f"Initial state: zone_3_consecutive_ticks=0, is_engagement_active=True")
    logger.info("")
    
    final_state = engine.run(max_ticks=200)
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("FINAL RESULTS")
    logger.info("=" * 70)
    logger.info(f"Ticks executed: {final_state.current_tick}")
    logger.info(f"is_engagement_active: {final_state.is_engagement_active}")
    logger.info(f"battle_winner: {final_state.battle_winner}")
    logger.info(f"zone_3_consecutive_ticks: {final_state.zone_3_consecutive_ticks}")
    
    # Determine win condition type
    if final_state.battle_winner is not None:
        if final_state.current_tick < 200:
            # Early termination
            zone_3 = next(z for z in final_state.zones if z.zone_id == 3)
            
            total_red = sum(z.red_units for z in final_state.zones)
            total_blue = sum(z.blue_units for z in final_state.zones)
            
            if total_red == 0 or total_blue == 0:
                logger.info(f"✓ WIN CONDITION: ELIMINATION (Red={total_red}, Blue={total_blue})")
            elif zone_3.side_control in ('red', 'blue'):
                logger.info(f"✓ WIN CONDITION: ZONE-3 HOLD (control={zone_3.side_control}, consecutive_ticks={final_state.zone_3_consecutive_ticks})")
            else:
                logger.info(f"⚠ WIN CONDITION: Unknown (but battle ended)")
        else:
            logger.info(f"⚠ TIMEOUT: Reached max_ticks=200 but battle_winner is {final_state.battle_winner}")
    else:
        if final_state.current_tick >= 200:
            logger.info(f"⚠ TIMEOUT: Ran full 200 ticks with no winner")
            total_red = sum(z.red_units for z in final_state.zones)
            total_blue = sum(z.blue_units for z in final_state.zones)
            logger.info(f"   Final units: Red={total_red}, Blue={total_blue}")
            zone_3 = next(z for z in final_state.zones if z.zone_id == 3)
            logger.info(f"   Zone 3 control: {zone_3.side_control} (consecutive_ticks={final_state.zone_3_consecutive_ticks})")
        else:
            logger.info(f"⚠ ERROR: Terminated early but no winner set")
    
    logger.info("")
    logger.info("Final zone state:")
    for z in final_state.zones:
        logger.info(f"  Zone {z.zone_id}: red={z.red_units:3d}, blue={z.blue_units:3d}, control={z.side_control:10s}")

if __name__ == '__main__':
    run_test()
