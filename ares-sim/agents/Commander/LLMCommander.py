import json

from litellm.exceptions import APIError
from litellm.types.utils import ModelResponse
import litellm
from core.state import CommanderObs 
from litellm import completion
from pathlib import Path
from typing import cast
from agents.models import Actions,ActionType,CommanderAction, CommanderDecision, CommanderMemory
from agents.Commander.BaseCommander import BaseCommander
CommanderPrompt = Path(
        "prompts/commander_prompt.md"
        ).read_text()


litellm.enable_json_schema_validation = True
# litellm.mock_response




class LLMCommander(BaseCommander):
    def __init__(self, model : str):
        self.model = model
    
    def decide(self, obs : CommanderObs, memory: CommanderMemory  )->CommanderDecision:
        messages = [
                {"role": "system","content": CommanderPrompt},
                {"role": "user", "content": json.dumps(
                    {
                        "observation" : obs.model_dump(),
                        "memory" : memory.model_dump() 
                        }
                    )} 
                ]
        try:
            response : ModelResponse =cast(ModelResponse, completion(
            model=self.model,
            messages=messages,
            response_format=CommanderDecision,
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
                    response_format=CommanderDecision,
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
                commander_memory_on_hold  = CommanderMemory(
                          current_objective="Hold",
                          last_action_summary="Held all positions due to llm failure Prompt",
                          tick_of_last_strategy_changed=memory.tick_of_last_strategy_changed
                          # How do i write this here     
                          
                     )
                
                return CommanderDecision(actions=Actions(actions=hold_actions),memory=commander_memory_on_hold)
          

        response_content = response.choices[0].message.content
        if response_content is None:
                raise ValueError("Empty response")
        return CommanderDecision.model_validate_json(response_content)

