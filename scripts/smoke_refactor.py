"""Quick smoke test: verify step()/run() backward compat with DummyCommander."""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "ares-sim"))

from config.seeds import get_seed_1
from core.tick import TickEngine
from agents.models import Actions, CommanderDecision, CommanderMemory


class DummyCommander:
    def __init__(self):
        self.last_call_outcome = "success"

    def decide(self, obs, memory=None):
        if memory is None:
            memory = CommanderMemory(current_objective="", last_action_summary="", tick_of_last_strategy_changed=None)
        return CommanderDecision(actions=Actions(actions=[]), memory=memory)


# Test 1: run() still works (backward compat)
print("=" * 60)
print("Test 1: run() backward compatibility")
state = get_seed_1()
engine = TickEngine(state, DummyCommander(), DummyCommander())
final = engine.run(max_ticks=10)
print(f"  Final tick: {final.current_tick}")
print(f"  Active: {final.is_engagement_active}")
print(f"  Winner: {final.battle_winner}")
print("  [OK] PASSED")

# Test 2: step() returns StepResult
print()
print("Test 2: step() returns StepResult")
state = get_seed_1()
engine = TickEngine(state, DummyCommander(), DummyCommander())
result = engine.step()
print(f"  Type: {type(result).__name__}")
print(f"  blue_obs side: {result.blue_obs.side}")
print(f"  red_obs side: {result.red_obs.side}")
print(f"  blue_reward: {result.blue_reward}")
print(f"  red_reward: {result.red_reward}")
print(f"  done: {result.done}")
print(f"  next_blue_obs: {'present' if result.next_blue_obs else 'None'}")
print(f"  info keys: {list(result.info.keys())}")
print("  [OK] PASSED")

# Test 3: step() loop matches run()
print()
print("Test 3: step() loop equivalence")
state = get_seed_1()
engine = TickEngine(state, DummyCommander(), DummyCommander())
ticks = 0
while not engine.done and ticks < 10:
    engine.step()
    ticks += 1
print(f"  Ticks executed: {ticks}")
print(f"  Final tick: {engine.state.current_tick}")
assert engine.state.current_tick == final.current_tick, "MISMATCH!"
print(f"  Matches run() result: OK")
print("  [OK] PASSED")

# Test 4: done property
print()
print("Test 4: engine.done property")
state = get_seed_1()
engine = TickEngine(state, DummyCommander(), DummyCommander())
print(f"  Before any steps: done={engine.done}")
assert not engine.done
engine.run(max_ticks=10)
print(f"  After run(10):    done={engine.done}")
print("  [OK] PASSED")

print()
print("=" * 60)
print("ALL TESTS PASSED")
