"""
Smoke test: single LLMCommander.decide() call.

Run from the repo root:
    python scripts/smoke_llm.py

Reads GROQ_API_KEY and MODEL_1 from .env via pydantic_settings.
"""

import sys
import pathlib

# Add ares-sim to path so bare package imports (core, agents, config) resolve
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "ares-sim"))

from config.settings import Settings
from config.seeds import get_seed_1
from core.obs import build_obs
from core.state import Side
from agents.models import CommanderMemory
from agents.Commander.LLMCommander import LLMCommander

state = get_seed_1()

# Same empty enemy memory TickEngine starts with on tick 0
enemy_memory = {"zone": None, "units": None, "tick_seen": None}

obs = build_obs(state, Side.BLUE, enemy_memory)

memory = CommanderMemory(
    current_objective="No prior objective",
    last_action_summary="N/A",
    tick_of_last_strategy_changed=None,
)

settings = Settings()  # loads .env automatically
commander = LLMCommander(model=settings.model_1)

print(f"Model : {settings.model_1}")
print(f"Obs   : {obs.model_dump()}")
print()

result = commander.decide(obs, memory)

print("=== RESULT ===")
print(f"call_outcome : {commander.last_call_outcome}")
print(f"actions      : {result.actions.model_dump()}")
print(f"memory       : {result.memory.model_dump()}")
