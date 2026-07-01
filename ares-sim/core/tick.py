import pathlib
import logging
from dataclasses import dataclass, field
from typing import Callable

from core.state import BattleState, Side, ZoneControl
from core.obs import build_obs
from agents.models import ActionType, TickLogEntry, CommanderMemory, CommanderDecision
from agents.Commander.BaseCommander import BaseCommander
from core.outcomes import check_win_condition
from core.zones import update_zone_3_ticks
from core.intel import update_enemy_memory
from core.action_resolver import resolve_actions
from core.state_updater import apply_deltas
from agents.models import FullTickSnapshot, CommanderObs
from agents.RL.DQN.utils import compute_reward

logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    """Everything an RL algorithm needs after one simulation tick."""
    blue_obs: CommanderObs          # Observation Blue received this tick
    red_obs: CommanderObs           # Observation Red received this tick
    blue_action: CommanderDecision  # Action Blue chose
    red_action: CommanderDecision   # Action Red chose
    state: BattleState              # Updated state after this tick
    blue_reward: float              # Reward for Blue this tick
    red_reward: float               # Reward for Red this tick
    done: bool                      # Whether the episode ended
    next_blue_obs: "CommanderObs | None"  # Next tick's Blue observation (None if done)
    next_red_obs: "CommanderObs | None"   # Next tick's Red observation (None if done)
    info: dict = field(default_factory=dict)  # Extra metadata


class TickEngine:
    def __init__(
        self,
        state: BattleState,
        blue_commander: BaseCommander,
        red_commander: BaseCommander,
        log_path: str | pathlib.Path | None = None,
        reward_fn: Callable[[BattleState, BattleState], float] | None = None,
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

        # Reward function (default: DQN compute_reward from Blue's perspective)
        self._reward_fn = reward_fn if reward_fn is not None else compute_reward

        # Enemy memory: {Side -> {zone, units, tick_seen}}
        # Persists across ticks; updated only when enemy visible in controlled zones
        self.enemy_memory: dict[Side, dict[str, int | None]] = {
            Side.BLUE: {"zone": None, "units": None, "tick_seen": None},
            Side.RED: {"zone": None, "units": None, "tick_seen": None},
        }

        # Commander memory — threaded forward across ticks
        self.blue_memory = CommanderMemory(
            current_objective="",
            last_action_summary="",
            tick_of_last_strategy_changed=None,
        )
        self.red_memory = CommanderMemory(
            current_objective="",
            last_action_summary="",
            tick_of_last_strategy_changed=None,
        )

    @property
    def done(self) -> bool:
        """Whether the episode has ended."""
        return not self.state.is_engagement_active

    def step(self) -> StepResult:
        """Execute a single simulation tick and return everything an RL loop needs.

        Raises RuntimeError if the episode has already ended.
        """
        if not self.state.is_engagement_active:
            raise RuntimeError("Episode has ended. Create a new TickEngine to start a new episode.")

        # --- 1. Update enemy memory for both sides ---
        update_enemy_memory(self.state, self.enemy_memory, Side.BLUE)
        update_enemy_memory(self.state, self.enemy_memory, Side.RED)

        # --- 2. Build observations ---
        blue_obs = build_obs(self.state, Side.BLUE, self.enemy_memory[Side.BLUE])
        red_obs = build_obs(self.state, Side.RED, self.enemy_memory[Side.RED])

        # --- 3. Collect decisions (alternating turn order for fairness) ---
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
            mem = self.blue_memory if side == Side.BLUE else self.red_memory
            results[side] = (commander.decide(obs, mem), commander.last_call_outcome)

        blue_decision, blue_outcome = results[Side.BLUE]
        red_decision, red_outcome = results[Side.RED]

        # Thread updated memory forward
        self.blue_memory = blue_decision.memory
        self.red_memory = red_decision.memory

        # --- 4. Resolve actions ---
        all_actions = blue_decision.actions.actions + red_decision.actions.actions
        pending_deltas, actions_taken, actions_rejected = resolve_actions(
            self.state, all_actions
        )

        # --- 5. Capture pre-delta state for reward computation ---
        prev_state = self.state

        # Save previous zone 3 control before applying deltas
        prev_zone_3_control = next(
            (z.side_control for z in self.state.zones if z.zone_id == 3),
            None,
        )

        # --- 6. Apply deltas ---
        self.state = apply_deltas(self.state, pending_deltas)

        # Log tick result
        logger.info(
            f"Tick {self.state.current_tick}: "
            f"Blue actions executed={actions_taken[Side.BLUE]}, rejected={actions_rejected[Side.BLUE]}, "
            f"Red actions executed={actions_taken[Side.RED]}, rejected={actions_rejected[Side.RED]}, "
            f"Deltas applied={len(pending_deltas)}, "
            f"Zone control: {[(z.zone_id, z.side_control) for z in self.state.zones]}"
        )

        # --- 7. Update zone 3 tick counter ---
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

        # --- 8. Check win condition ---
        self.state = check_win_condition(state=self.state)

        # --- 9. Compute rewards ---
        blue_reward = self._reward_fn(prev_state, self.state)
        red_reward = -blue_reward  # Zero-sum game

        # --- 10. Write log snapshot ---
        if self.log_file_handle:
            snapshot = FullTickSnapshot(
                tick=self.state.current_tick,
                is_engagement_active=self.state.is_engagement_active,
                battle_winner=self.state.battle_winner,
                zones=self.state.zones,
                red_fuel=self.state.red_fuel,
                blue_fuel=self.state.blue_fuel,
                red_weapons_remaining=self.state.red_weapons_remaining,
                blue_weapons_remaining=self.state.blue_weapons_remaining,
                zone_3_consecutive_ticks=self.state.zone_3_consecutive_ticks,
                blue_observation=blue_obs,
                red_observation=red_obs,
                blue_decision=blue_decision,
                red_decision=red_decision,
                blue_actions_taken=actions_taken[Side.BLUE],
                blue_actions_rejected=actions_rejected[Side.BLUE],
                red_actions_taken=actions_taken[Side.RED],
                red_actions_rejected=actions_rejected[Side.RED],
                total_deltas_applied=len(pending_deltas),
            )
            self.log_file_handle.write(snapshot.model_dump_json() + "\n")
            self.log_file_handle.flush()

        # Human-readable console logging
        logger.info(
            f"Tick {self.state.current_tick}: "
            f"Blue: {actions_taken[Side.BLUE]} taken, {actions_rejected[Side.BLUE]} rej | "
            f"Red: {actions_taken[Side.RED]} taken, {actions_rejected[Side.RED]} rej | "
            f"Z3 Control: {zone_3.side_control if zone_3 else 'N/A'} ({self.state.zone_3_consecutive_ticks})"
        )

        # --- 11. Advance tick ---
        self.state = self.state.model_copy(
            update={"current_tick": self.state.current_tick + 1}
        )

        # --- 12. Build next observations (or None if episode ended) ---
        episode_done = not self.state.is_engagement_active

        if not episode_done:
            # Pre-compute next observations for RL transition tuples
            update_enemy_memory(self.state, self.enemy_memory, Side.BLUE)
            update_enemy_memory(self.state, self.enemy_memory, Side.RED)
            next_blue_obs = build_obs(self.state, Side.BLUE, self.enemy_memory[Side.BLUE])
            next_red_obs = build_obs(self.state, Side.RED, self.enemy_memory[Side.RED])
        else:
            next_blue_obs = None
            next_red_obs = None

        return StepResult(
            blue_obs=blue_obs,
            red_obs=red_obs,
            blue_action=blue_decision,
            red_action=red_decision,
            state=self.state,
            blue_reward=blue_reward,
            red_reward=red_reward,
            done=episode_done,
            next_blue_obs=next_blue_obs,
            next_red_obs=next_red_obs,
            info={
                "actions_taken": actions_taken,
                "actions_rejected": actions_rejected,
                "deltas_applied": len(pending_deltas),
            },
        )

    def run(self, max_ticks: int) -> BattleState:
        """Run the simulation for up to max_ticks. Convenience wrapper around step().

        Maintains full backward compatibility with existing scripts.
        """
        while self.state.is_engagement_active and self.state.current_tick < max_ticks:
            self.step()

        if self.log_file_handle:
            self.log_file_handle.close()

        return self.state
