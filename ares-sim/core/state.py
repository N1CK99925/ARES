

from pydantic import BaseModel
from pydantic import Field
from enum import Enum
class Side(str, Enum):
    RED = "red"
    BLUE = "blue"

class ZoneSnapshot(BaseModel):
    pass

class BattleState(BaseModel):
    pass

class CommanderOrbs(BaseModel):
    pass




