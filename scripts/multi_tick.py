import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "ares-sim"))

from config.settings import settings
from config.seeds import get_seed_1
from agents.Commander.LLMCommander import LLMCommander
from core.tick import TickEngine


state = get_seed_1()


blue_commander = LLMCommander(model=settings.model_1)
red_commander  = LLMCommander(model=settings.model_1)
logpath = pathlib.Path(__file__).parent.parent / "logs" /"multi_tick_run_jsonl"   

engine = TickEngine(state,blue_commander,red_commander,logpath)
max_ticks = 15 
final_state = engine.run(max_ticks)

print(f"Final tick: {final_state.current_tick}")
print(f"Battle winner: {final_state.battle_winner}")
print(f"Log written to: {logpath}")




