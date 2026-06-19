#!/usr/bin/env python3
import sys
sys.path.insert(0, 'ares-sim')

from core.tick import TickEngine
from config.seeds import get_seed_1
from agents.commander import Actions
import logging

logging.basicConfig(level=logging.INFO, format='[%(name)s] %(message)s')
logger = logging.getLogger('TEST')

class DummyCommander:
    def decide(self, obs):
        return Actions(actions=[])

state = get_seed_1()
engine = TickEngine(state, DummyCommander(), DummyCommander())

logger.info("=" * 70)
logger.info("Starting 200-tick simulation with ZONE_3_WIN_THRESHOLD=6")
logger.info("=" * 70)

final_state = engine.run(max_ticks=200)

logger.info("")
logger.info("=" * 70)
logger.info("FINAL RESULTS")
logger.info("=" * 70)
logger.info(f"Ticks executed: {final_state.current_tick}")
logger.info(f"is_engagement_active: {final_state.is_engagement_active}")
logger.info(f"battle_winner: {final_state.battle_winner}")
logger.info(f"zone_3_consecutive_ticks: {final_state.zone_3_consecutive_ticks}")

zone_3 = next(z for z in final_state.zones if z.zone_id == 3)
total_red = sum(z.red_units for z in final_state.zones)
total_blue = sum(z.blue_units for z in final_state.zones)

logger.info(f"Zone 3 control: {zone_3.side_control}")
logger.info(f"Total units - Red: {total_red}, Blue: {total_blue}")

if final_state.battle_winner:
    if total_red == 0 or total_blue == 0:
        logger.info("✓ WIN: ELIMINATION")
    elif final_state.zone_3_consecutive_ticks >= 6:
        logger.info("✓ WIN: ZONE-3 HOLD")
else:
    logger.info(f"⚠ NO WINNER (ticks={final_state.current_tick})")
