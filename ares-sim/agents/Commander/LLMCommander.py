import json
from pathlib import Path
from typing import cast

import litellm
from litellm import completion
from litellm.types.utils import ModelResponse

from core.state import CommanderObs
from agents.models import (
    Actions,
    ActionType,
    CommanderAction,
    CommanderDecision,
    CommanderMemory,
)
from agents.Commander.BaseCommander import BaseCommander


CommanderPrompt = Path(
    "prompts/commander_prompt.md"
).read_text()

litellm.enable_json_schema_validation = True


class LLMCommander(BaseCommander):
    def __init__(self, model: str):
        super().__init__()
        self.model = model

    def _hold_decision(
        self,
        obs: CommanderObs,
        memory: CommanderMemory,
        reason: str = "LLM failure",
    ) -> CommanderDecision:

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

        commander_memory = CommanderMemory(
            current_objective="Hold",
            last_action_summary=reason,
            tick_of_last_strategy_changed=memory.tick_of_last_strategy_changed,
        )

        return CommanderDecision(
            actions=Actions(actions=hold_actions),
            memory=commander_memory,
        )

    def _parse_response(
        self,
        response: ModelResponse,
    ) -> CommanderDecision:

        response_content = response.choices[0].message.content

        if response_content is None:
            raise ValueError("Empty response")

        return CommanderDecision.model_validate_json(response_content)

    def decide(
        self,
        obs: CommanderObs,
        memory: CommanderMemory,
    ) -> CommanderDecision:

        messages = [
            {
                "role": "system",
                "content": CommanderPrompt,
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "observation": obs.model_dump(),
                        "memory": memory.model_dump(),
                    }
                ),
            },
        ]

        try:
            response = cast(
                ModelResponse,
                completion(
                    model=self.model,
                    messages=messages,
                    response_format=CommanderDecision,
                    stream=False,
                ),
            )
            decision = self._parse_response(response)
            self.last_call_outcome = "success"
            return decision
        except Exception:
            pass

        retry_messages = messages + [
            {
                "role": "user",
                "content": (
                    "Your previous response failed validation. "
                    "Return a valid CommanderDecision and obey the schema."
                ),
            }
        ]

        try:
            response = cast(
                ModelResponse,
                completion(
                    model=self.model,
                    messages=retry_messages,
                    response_format=CommanderDecision,
                    stream=False,
                ),
            )
        except Exception:
            # API call itself failed on retry.
            self.last_call_outcome = "hold_fallback_api"
            return self._hold_decision(
                obs,
                memory,
                "Held all positions due to LLM API failure",
            )

        try:
            decision = self._parse_response(response)
            self.last_call_outcome = "retry_success"
            return decision
        except Exception:
            # Retry API call succeeded but response failed schema validation.
            self.last_call_outcome = "hold_fallback_validation"
            return self._hold_decision(
                obs,
                memory,
                "Held all positions due to schema validation failure",
            )
