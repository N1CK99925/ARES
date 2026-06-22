import pathlib
import time
import logging

from core.state import BattleState, Side, ZoneControl
from core.obs import build_obs
from agents.models import ActionType, TickLogEntry
from agents.Commander.BaseCommander import BaseCommander
from core.outcomes import check_win_condition
from core.zones import update_zone_3_ticks
from core.intel import get_visible_zones_for_side, update_enemy_memory
from core.action_resolver import resolve_actions
from core.state_updater import apply_deltas

logger = logging.getLogger(__name__)

TPM_LIMIT = 6000
SAFETY_MARGIN = 0.8
TOKENS_PER_CALL = 2100  
CALLS_PER_TICK = 2 
SLEEP_SECONDS = 60 * (TOKENS_PER_CALL * CALLS_PER_TICK) / (TPM_LIMIT * SAFETY_MARGIN)


class TickEngine:
    def __init__(
        self,
        state: BattleState,
        blue_commander: BaseCommander,
        red_commander: BaseCommander,
        log_path: str | pathlib.Path | None = None,
    ):
        self.state = state
        self.blue = blue_commander
        self.red = red_commander
        self.log_path = pathlib.Path(log_path) if log_path else None
        self.tick_log: list[TickLogEntry] = []
        # Enemy memory: {Side -> {zone, units, tick_seen}}
        # Persists across ticks; updated only when enemy visible in controlled zones
        self.enemy_memory: dict[Side, dict[str, int | None]] = {
            Side.BLUE: {"zone": None, "units": None, "tick_seen": None},
            Side.RED: {"zone": None, "units": None, "tick_seen": None},
        }

    def run(self, max_ticks: int) -> BattleState:
        from agents.models import CommanderMemory

        # Initialize/truncate the log file if a path is provided
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_path.open("w", encoding="utf-8").close()

        # Seed initial memory for each side — commanders update and return
        # new memory each tick; we thread it forward.
        blue_memory = CommanderMemory(
            current_objective="",
            last_action_summary="",
            tick_of_last_strategy_changed=None,
        )
        red_memory = CommanderMemory(
            current_objective="",
            last_action_summary="",
            tick_of_last_strategy_changed=None,
        )

        while self.state.is_engagement_active and self.state.current_tick < max_ticks:
            # Update enemy memory for both sides based on current zone control
            update_enemy_memory(self.state, self.enemy_memory, Side.BLUE)
            update_enemy_memory(self.state, self.enemy_memory, Side.RED)

            # Collect decisions from both sides (obs + memory in, decision + new memory out)
            blue_obs = build_obs(self.state, Side.BLUE, self.enemy_memory[Side.BLUE])
            red_obs = build_obs(self.state, Side.RED, self.enemy_memory[Side.RED])

            blue_decision = self.blue.decide(blue_obs, blue_memory)
            blue_outcome = self.blue.last_call_outcome
            red_decision = self.red.decide(red_obs, red_memory)
            red_outcome = self.red.last_call_outcome

            # Thread updated memory forward to the next tick
            blue_memory = blue_decision.memory
            red_memory = red_decision.memory

            time.sleep(SLEEP_SECONDS)

            # Build log entries for this tick (before actions are resolved)
            for side, decision, outcome, commander in [
                (Side.BLUE, blue_decision, blue_outcome, self.blue),
                (Side.RED,  red_decision,  red_outcome, self.red),
            ]:
                entry = TickLogEntry(
                    tick=self.state.current_tick,
                    side=side,
                    call_outcome=outcome,
                    actions=decision.actions,
                    current_objective=decision.memory.current_objective,
                    last_action_summary=decision.memory.last_action_summary,
                    tick_of_last_strategy_changed=decision.memory.tick_of_last_strategy_changed,
                    error_details=getattr(commander, "last_error", None),
                )
                self.tick_log.append(entry)
                if self.log_path:
                    with self.log_path.open("a", encoding="utf-8") as fh:
                        fh.write(entry.model_dump_json() + "\n")

            # BUG FIX: .actions on a CommanderDecision is an Actions model;
            # .actions.actions is the underlying list[CommanderAction].
            all_actions = (
                blue_decision.actions.actions + red_decision.actions.actions
            )

            # Resolve all actions and collect deltas (no state mutation yet)
            pending_deltas, actions_taken, actions_rejected = resolve_actions(
                self.state, all_actions
            )

            # Save previous zone 3 control before applying deltas
            prev_zone_3_control = next(
                (z.side_control for z in self.state.zones if z.zone_id == 3),
                None,
            )

            # Apply all deltas in one batch
            self.state = apply_deltas(self.state, pending_deltas)

            # Log tick result
            logger.info(
                f"Tick {self.state.current_tick}: "
                f"Blue actions executed={actions_taken[Side.BLUE]}, rejected={actions_rejected[Side.BLUE]}, "
                f"Red actions executed={actions_taken[Side.RED]}, rejected={actions_rejected[Side.RED]}, "
                f"Deltas applied={len(pending_deltas)}, "
                f"Zone control: {[(z.zone_id, z.side_control) for z in self.state.zones]}"
            )

            # Update zone 3 tick counter
            zone_3 = next((z for z in self.state.zones if z.zone_id == 3), None)
            if zone_3:
                new_consecutive_ticks = update_zone_3_ticks(
                    current_control=zone_3.side_control,
                    previous_control=prev_zone_3_control or zone_3.side_control,
                    consecutive_ticks=self.state.zone_3_consecutive_ticks,
                )
                self.state = self.state.model_copy(
                    update={"zone_3_consecutive_ticks": new_consecutive_ticks}
                )
            
            # Check win condition
            self.state = check_win_condition(state=self.state)
            
            # Advance tick
            self.state = self.state.model_copy(
                update={"current_tick": self.state.current_tick + 1}
            )

        return self.state