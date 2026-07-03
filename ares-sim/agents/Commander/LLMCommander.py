import json
from pathlib import Path
from typing import cast
from datetime import datetime
import time
import os


import litellm

os.environ["LITELLM_LOG"] = "INFO"

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


CommanderPrompt = (
    Path(__file__).parent.parent.parent / "prompts" / "commander_prompt.md"
).read_text()


litellm.enable_json_schema_validation = True


class LLMCommander(BaseCommander):
    _last_request_time: float | None = None
    _request_timestamps: list[float] = []
    _daily_count_date: str | None = None
    _daily_count: int = 0
    _max_requests_per_minute = 2
    _min_request_interval_seconds = 30.0
    _daily_request_limit = 1000

    def __init__(self, model: str):
        super().__init__()
        self.model = model

    @classmethod
    def _reset_daily_count_if_needed(cls) -> None:
        today = datetime.utcnow().date().isoformat()
        if cls._daily_count_date != today:
            cls._daily_count_date = today
            cls._daily_count = 0
            cls._request_timestamps = []
            cls._last_request_time = None

    @classmethod
    def _wait_for_request_slot(cls) -> None:
        cls._reset_daily_count_if_needed()
        now = time.monotonic()
        cls._request_timestamps = [
            t for t in cls._request_timestamps if now - t < 60
        ]

        if cls._daily_count >= cls._daily_request_limit:
            raise RuntimeError(
                f"Daily LLM request limit reached ({cls._daily_request_limit})."
            )

        if len(cls._request_timestamps) >= cls._max_requests_per_minute:
            oldest = cls._request_timestamps[0]
            wait_seconds = 60 - (now - oldest) + 0.1
            time.sleep(wait_seconds)
            now = time.monotonic()
            cls._request_timestamps = [
                t for t in cls._request_timestamps if now - t < 60
            ]

        if cls._last_request_time is not None:
            elapsed = now - cls._last_request_time
            if elapsed < cls._min_request_interval_seconds:
                time.sleep(cls._min_request_interval_seconds - elapsed)

    @classmethod
    def _record_request(cls) -> None:
        cls._reset_daily_count_if_needed()
        now = time.monotonic()
        cls._request_timestamps.append(now)
        cls._daily_count += 1
        cls._last_request_time = now
        cls._request_timestamps = [
            t for t in cls._request_timestamps if now - t < 60
        ]

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

        data = json.loads(response_content)
        if isinstance(data.get("actions"), list):
            data["actions"] = {"actions": data["actions"]}

        return CommanderDecision.model_validate(data)

    def decide(
        self,
        obs: CommanderObs,
        memory: CommanderMemory,
    ) -> CommanderDecision:

        prompt = CommanderPrompt.replace("{SIDE}", obs.side.value.upper())
        
        system_content = prompt + "\n\n" + (
            "You must respond ONLY with a JSON object. No explanation, no markdown blocks, just raw JSON.\n"
            "The output JSON must conform to the following schema structure:\n"
            "{\n"
            "  \"actions\": [\n"
            "    {\n"
            "      \"side\": \"red\" | \"blue\",\n"
            "      \"source_zone\": int,\n"
            "      \"target_zone\": int,\n"
            "      \"units_to_move\": int,\n"
            "      \"action_type\": \"hold\" | \"move\"\n"
            "    }\n"
            "  ],\n"
            "  \"memory\": {\n"
            "    \"current_objective\": str,\n"
            "    \"last_action_summary\": str,\n"
            "    \"tick_of_last_strategy_changed\": int | null\n"
            "  }\n"
            "}\n"
        )

        messages = [
            {
                "role": "system",
                "content": system_content,
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
            self.__class__._wait_for_request_slot()
            self.__class__._record_request()
            response = cast(
                ModelResponse,
                completion(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    stream=False,
                    max_retries=0,
                    reasoning_effort="low",
                    max_tokens=600
                ),
            )
            decision = self._parse_response(response)
            self.last_call_outcome = "success"
            return decision
        except Exception as e:
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {e}")
            self.last_error = f"{type(e).__name__}: {e}"
            time.sleep(5)
        retry_messages = messages + [
            {
                "role": "user",
                "content": (
                    "Your previous response failed validation. "
                    "Return a valid CommanderDecision and obey the schema structure."
                ),
            }
        ]

        try:
            self.__class__._wait_for_request_slot()
            self.__class__._record_request()
            response = cast(
                ModelResponse,
                completion(
                    model=self.model,
                    messages=retry_messages,
                    response_format={"type": "json_object"},
                    stream=False,
                    max_retries=0,
                    max_tokens=600,
                    reasoning_effort="low"
                ),
            )
        except Exception as e:
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {e}")
            self.last_error = f"{type(e).__name__}: {e}"
                    
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
        except Exception as e:
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {e}")
            self.last_error = f"{type(e).__name__}: {e}"
            
            self.last_call_outcome = "hold_fallback_validation"
            return self._hold_decision(
                obs,
                memory,
                "Held all positions due to schema validation failure",
            )
