from core.state import BattleState, Side, ZoneControl
from core.obs import build_obs
from agents.commander import ActionType, Commander
from core.attrition import calculate_attrition
from core.zones import determine_zone_control, update_zone_3_ticks
from core.outcomes import check_win_condition
class TickEngine:
    def __init__(self, state:BattleState , blue_commander, red_commander):
        self.state = state
        self.blue = blue_commander
        self.red = red_commander



    def run(self, max_ticks)-> BattleState:
        while self.state.is_engagement_active and self.state.current_tick < max_ticks:
            blue_obs = build_obs(self.state, Side.BLUE)
            red_obs  = build_obs(self.state, Side.RED)

            blue_actions = self.blue.decide(blue_obs)
            red_actions = self.red.decide(red_obs)
            actions = blue_actions.actions + red_actions.actions

            for action in actions:
                if action.action_type == ActionType.HOLD:
                    continue

                target_zone_snapshot = next(
                    (
                        z for z in self.state.zones
                        if z.zone_id == action.target_zone
                    ),
                    None,
                )
                if target_zone_snapshot is None:
                    continue

                attacker_units = action.units_to_move
                defender_units = (
                    target_zone_snapshot.red_units
                    if action.side == Side.BLUE
                    else target_zone_snapshot.blue_units
                )
                attacker_fuel = (
                    self.state.blue_fuel
                    if action.side == Side.BLUE
                    else self.state.red_fuel
                )
                attacker_controls_zone_3 = (
                    target_zone_snapshot.zone_id == 3
                    and (
                        (action.side == Side.BLUE and target_zone_snapshot.side_control == ZoneControl.BLUE)
                        or (action.side == Side.RED and target_zone_snapshot.side_control == ZoneControl.RED)
                    )
                )
                attrition_result = calculate_attrition(
                    attacker_units=attacker_units,
                    defender_units=defender_units,
                    attacker_fuel=attacker_fuel,
                    attacker_controls_flank=False,
                    attacker_controls_zone_3=attacker_controls_zone_3,
                )

                # apply attrition_result to the relevant zone and state later
            

            


            update_zone_3_ticks(self.state)
            check_win_condition(state=self.state)
            self.state = self.state.model_copy(update={"current_tick": self.state.current_tick + 1})
            
        
        return self.state





