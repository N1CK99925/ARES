from enum import Enum
from pydantic import BaseModel, Field , ConfigDict
class Side(str, Enum):
    RED = "red"
    BLUE = "blue"
        

class ZoneControl(str, Enum):
    RED = "red"
    BLUE = "blue"
    NEUTRAL = "neutral"
    CONTESTED = "contested"

class ZoneSnapshot(BaseModel):
    """
    Snapshot of a single zone at the current simulation tick.
    """

    zone_id: int = Field(ge=0)

    red_units: int = Field(
        ge=0,
        description="Number of red units currently present in the zone.",
    )

    blue_units: int = Field(
        ge=0,
        description="Number of blue units currently present in the zone.",
    )

    side_control: ZoneControl = Field(
        description="Current control state of the zone."
    )
    model_config = ConfigDict(frozen=True)


class BattleState(BaseModel):
    """
    Ground-truth simulation state maintained by the environment.

    Commanders should not directly access this model.
    Instead, they receive a CommanderOrbs generated from it.
    """

    current_tick: int = Field(ge=0)

    zones: list[ZoneSnapshot]

    red_fuel: int = Field(ge=0)
    blue_fuel: int = Field(ge=0)

    red_weapons_remaining: int = Field(ge=0)
    blue_weapons_remaining: int = Field(ge=0)

    is_engagement_active: bool

    battle_winner: Side | None = Field(
        default=None,
        description="Winning side if the battle has concluded.",
    )

    zone_3_consecutive_ticks:int = Field(
        default=0,
        ge=0,
        description=(
            "Consecutive ticks zone 3 has remained under its current control state. "),
    )
    model_config = ConfigDict(frozen=True)


class CommanderObs(BaseModel):
    """
    Partial observation provided to a commander agent.

    Represents what the commander currently knows about the battlefield,
    not the full simulation state.
    """

    side: Side

    current_tick: int = Field(ge=0)

    own_unit_per_zone: dict[int, int] = Field(
        description="Observed friendly unit count per zone."
    )

    own_fuel: int = Field(ge=0)

    own_weapons_remaining: int = Field(ge=0)

    enemy_last_known_unit_count: int | None = Field(
        default=None,
        ge=0,
        description="Most recently observed enemy unit count.",
    )

    enemy_last_known_zone: int | None = Field(
        default=None,
        ge=0,
        description="Zone where the enemy was most recently observed.",
    )

    how_many_ticks_ago_enemy_last_seen: int | None = Field(
        default=None,
        ge=0,
        description="Ticks elapsed since the last confirmed enemy sighting.",
    )
    model_config = ConfigDict(frozen=True)
    
    legal_targets_per_zone: dict[int, list[int]] = Field(
    default_factory=dict,
    description=(
        "For each zone you currently occupy, the exact list of zone_ids "
        "you are permitted to target with a move action. Use this directly "
        "as the source of truth for adjacency — do not infer adjacency yourself."
    ),
)

