from abc import ABC
from abc import abstractmethod
from typing import Any
from core.state import CommanderObs
from agents.models import  CommanderDecision

class BaseCommander(ABC):

    
    @abstractmethod
    def decide(self, obs: CommanderObs, memory: Any)-> CommanderDecision:
        pass


