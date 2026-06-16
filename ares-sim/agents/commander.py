from enum import Enum
from litellm.exceptions import APIError
from litellm.types.utils import ModelResponse
import litellm
from pydantic import BaseModel
from core.state import Side,CommanderObs 
from litellm import completion
from pathlib import Path
from typing import cast

CommanderPrompt = Path(
        "prompts/commander_prompt.md"
        ).read_text()
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


litellm.enable_json_schema_validation = True
# litellm.mock_response




class Commander:
    def __init__(self, model : str):
        self.model = model

    def decide(self, obs : CommanderObs)->Actions:
        messages = [
                {"role": "system","content": CommanderPrompt},
                {"role": "user", "content": obs.model_dump_json()}
                ]
        try:
            response : ModelResponse =cast(ModelResponse, completion(
            model=self.model,
            messages=messages,
            response_format=Actions,
            stream=False
            )
                                           )
        except APIError:
            try:
                retry_message = messages + [
                          {
            "role": "user",
            "content": (
                "Your previous response failed validation. "
                "Return a valid Actions object and obey the schema."
            ),
    }
                        ]
                response = cast (ModelResponse, completion(
                    model=self.model,
                    messages=retry_message,
                    response_format=Actions,
                    stream=False
                    )
                )
            except Exception:
             # Hold
              hold_actions = [
                CommanderAction(
                    side=obs.side,
                    source_zone=zone_id,
                    target_zone=zone_id,
                    units_to_move=0,
                    action_type=ActionType.HOLD,
                    )
                for zone_id, units in obs.own_unit_per_zone.items()
                if units > 0


                ]
              return Actions(actions=hold_actions) 

        actions = response.choices[0].message.content
        if actions is None:
                raise ValueError("Empty response")
        return Actions.model_validate_json(actions)
