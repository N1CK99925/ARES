from abc import ABC
from abc import abstractmethod
from typing import Any
from core.state import CommanderObs
from agents.models import CommanderDecision, CallOutcome

class BaseCommander(ABC):

    def __init__(self) -> None:
        
        self.last_call_outcome: CallOutcome = "success"

    @abstractmethod
    def decide(self, obs: CommanderObs, memory: Any) -> CommanderDecision:
        pass


