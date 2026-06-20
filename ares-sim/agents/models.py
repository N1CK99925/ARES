from pydantic import BaseModel
from enum import Enum
from typing import Literal
from core.state import Side
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
