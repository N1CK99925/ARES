import pathlib
import time
import logging

from core.state import BattleState, Side, ZoneControl
from core.obs import build_obs
from agents.models import ActionType  ,TickLogEntry
from agents.Commander.BaseCommander import BaseCommander
from core.outcomes import check_win_condition
from core.zones import update_zone_3_ticks
from core.intel import update_enemy_memory
from core.action_resolver import resolve_actions
from core.state_updater import apply_deltas
from agents.models import FullTickSnapshot

logger = logging.getLogger(__name__)
TPM_LIMIT = 8000
SAFETY_MARGIN = 0.8
TOKENS_PER_CALL = 2328 + 900  # input + conservative output ceiling at low effort ≈ 3228
CALLS_PER_TICK = 2

SLEEP_SECONDS = 60 * (TOKENS_PER_CALL * CALLS_PER_TICK) / (TPM_LIMIT * SAFETY_MARGIN)
# = 60 * 6456 / 6400 ≈ 60.5s/tick
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
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_file_handle = self.log_path.open("w", encoding="utf-8")
        else:
            self.log_file_handle = None
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
            if self.state.current_tick % 2 == 0:
                turn_order = [
                    (Side.BLUE, self.blue, blue_obs),
                    (Side.RED, self.red, red_obs),
                ]
            else:
                turn_order = [
                    (Side.RED, self.red, red_obs),
                    (Side.BLUE, self.blue, blue_obs),
                ]

            results = {}
            for side, commander, obs in turn_order:
                mem = blue_memory if side == Side.BLUE else red_memory
                results[side] = (commander.decide(obs, mem), commander.last_call_outcome)
                time.sleep(5)  # small gap between same-tick calls

            blue_decision, blue_outcome = results[Side.BLUE]
            red_decision, red_outcome = results[Side.RED]
            # Thread updated memory forward to the next tick
            blue_memory = blue_decision.memory
            red_memory = red_decision.memory

            time.sleep(SLEEP_SECONDS)

            # Build log entries for this tick (before actions are resolved)
            

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


            if self.log_file_handle:
                snapshot = FullTickSnapshot(
                    tick=self.state.current_tick,
                    is_engagement_active=self.state.is_engagement_active,
                    battle_winner=self.state.battle_winner,
                    zones=self.state.zones, # Ground truth
                    red_fuel=self.state.red_fuel,
                    blue_fuel=self.state.blue_fuel,
                    red_weapons_remaining=self.state.red_weapons_remaining,
                    blue_weapons_remaining=self.state.blue_weapons_remaining,
                    zone_3_consecutive_ticks=self.state.zone_3_consecutive_ticks,
                    blue_observation=blue_obs,  # What Blue saw
                    red_observation=red_obs,    # What Red saw
                    blue_decision=blue_decision, # What Blue ordered
                    red_decision=red_decision,   # What Red ordered
                    blue_actions_taken=actions_taken[Side.BLUE],
                    blue_actions_rejected=actions_rejected[Side.BLUE],
                    red_actions_taken=actions_taken[Side.RED],
                    red_actions_rejected=actions_rejected[Side.RED],
                    total_deltas_applied=len(pending_deltas),
                )
                # Write as a single JSON line
                self.log_file_handle.write(snapshot.model_dump_json() + "\n")
                self.log_file_handle.flush()  # Ensure it's written to disk immediately

            # Human-readable console logging (very useful for watching it live)
            logger.info(
                f"Tick {self.state.current_tick}: "
                f"Blue: {actions_taken[Side.BLUE]} taken, {actions_rejected[Side.BLUE]} rej | "
                f"Red: {actions_taken[Side.RED]} taken, {actions_rejected[Side.RED]} rej | "
                f"Z3 Control: {zone_3.side_control if zone_3 else 'N/A'} ({self.state.zone_3_consecutive_ticks})"
            )

            
            # Advance tick
            self.state = self.state.model_copy(
                update={"current_tick": self.state.current_tick + 1}
            )

        if self.log_file_handle:
            self.log_file_handle.close()

        return self.state
