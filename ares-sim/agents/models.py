from pydantic import BaseModel
from enum import Enum
from typing import Literal
from core.state import Side
from core.state import BattleState, CommanderObs, ZoneSnapshot 


class ActionType(str, Enum):
    HOLD= "hold"
    MOVE= "move"
class CommanderAction(BaseModel):
   side : Side
   source_zone:int
   target_zone:int
   units_to_move:int
   action_type : ActionType
class Actions(BaseModel):
    actions : list[CommanderAction]


class CommanderMemory(BaseModel):
    current_objective: str
    last_action_summary: str
    tick_of_last_strategy_changed: int | None

class CommanderDecision(BaseModel):
    actions : Actions
    memory : CommanderMemory


CallOutcome = Literal[
    "success",
    "retry_success",
    "hold_fallback_api",
    "hold_fallback_validation",
]


class TickLogEntry(BaseModel):
    tick: int
    side: Side
    call_outcome: CallOutcome
    actions: Actions
    current_objective: str
    last_action_summary: str
    tick_of_last_strategy_changed: int | None
    error_details: str | None = None



class FullTickSnapshot(BaseModel):
    tick: int
    is_engagement_active: bool
    battle_winner: Side | None

    # 1. GROUND TRUTH (What actually happened)
    zones: list[ZoneSnapshot]
    red_fuel: int
    blue_fuel: int
    red_weapons_remaining: int
    blue_weapons_remaining: int
    zone_3_consecutive_ticks: int

    # 2. PARTIAL OBSERVATIONS (What each side *thinks* is happening)
    blue_observation: CommanderObs
    red_observation: CommanderObs

    # 3. DECISIONS MADE (Their orders)
    blue_decision: CommanderDecision
    red_decision: CommanderDecision

    # 4. RESOLUTION METRICS (How many orders actually worked)
    blue_actions_taken: int
    blue_actions_rejected: int
    red_actions_taken: int
    red_actions_rejected: int
    total_deltas_applied: int
    # Optional: List the actual deltas if you want extreme verbosity
    # applied_deltas: list[Delta] 
