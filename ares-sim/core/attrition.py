from pydantic import BaseModel, ConfigDict
import math


class AttritionResult(BaseModel):
    attacker_losses: int
    defender_losses: int
    attacker_fuel_penalty: int
    attacker_weapons_consumed: int

    model_config = ConfigDict(frozen=True)


def calculate_attrition(
    attacker_units: int,
    defender_units: int,
    attacker_fuel: int,
    attacker_controls_flank: bool,
    attacker_controls_zone_3: bool,
) -> AttritionResult:
    # Base losses before any modifiers
    attacker_losses = defender_units * 0.05
    defender_losses = attacker_units * 0.06

    fuel_penalty = 0

    # Numerical advantage modifier
    force_ratio = min(
    math.sqrt(attacker_units / max(defender_units, 1)),
    2.0,
)

    # Larger attacking force inflicts proportionally more losses
    defender_losses *= force_ratio

    # Zone 3 control provides a positional advantage
    if attacker_controls_zone_3:
        defender_losses *= 1.2

    # Lack of flank control reduces combat effectiveness
    if not attacker_controls_flank:
        defender_losses *= 0.7
        fuel_penalty = 10

    # Fuel level impacts sortie effectiveness
    fuel_effectiveness = max(
    0.5,
    min(attacker_fuel / 150, 1.0),
)
    defender_losses *= fuel_effectiveness

    # Weapons expenditure scales with engagement activity
    attacker_weapons_consumed = round(attacker_units * 0.05)

    # Convert to integers and ensure losses never exceed available units
    attacker_losses = min(round(attacker_losses), attacker_units)
    defender_losses = min(round(defender_losses), defender_units)

    return AttritionResult(
        attacker_losses=attacker_losses,
        defender_losses=defender_losses,
        attacker_fuel_penalty=fuel_penalty,
        attacker_weapons_consumed=attacker_weapons_consumed,
    )
