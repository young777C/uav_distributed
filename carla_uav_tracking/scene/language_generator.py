"""Auto-generate natural language tracking instructions for multi-vehicle scenarios.

Generates descriptions that uniquely identify a TARGET vehicle among DISTRACTORS,
forcing the VLA to use both visual recognition AND language understanding.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

# Vehicle blueprint → visual description (color + type)
# CARLA vehicles have baked-in colors — these descriptions match the actual meshes
VEHICLE_DESCRIPTIONS = {
    "vehicle.audi.a2":                "silver Audi hatchback",
    "vehicle.audi.tt":                "red Audi TT coupe",
    "vehicle.audi.etron":             "white Audi e-tron SUV",
    "vehicle.tesla.model3":           "red Tesla Model 3 sedan",
    "vehicle.tesla.cybertruck":       "silver Tesla Cybertruck",
    "vehicle.bmw.grandtourer":        "black BMW coupe",
    "vehicle.mercedes.coupe":         "white Mercedes coupe",
    "vehicle.mercedes.coupe_2020":    "silver Mercedes coupe",
    "vehicle.mini.cooper_s":          "green Mini Cooper",
    "vehicle.mini.cooper_s_2021":     "yellow Mini Cooper",
    "vehicle.nissan.patrol":          "dark Nissan Patrol SUV",
    "vehicle.nissan.patrol_2021":     "white Nissan Patrol SUV",
    "vehicle.nissan.micra":           "blue Nissan Micra",
    "vehicle.jeep.wrangler_rubicon":   "orange Jeep Wrangler",
    "vehicle.ford.mustang":           "red Ford Mustang",
    "vehicle.ford.crown":             "yellow taxi cab",
    "vehicle.ford.ambulance":         "white ambulance van",
    "vehicle.chevrolet.impala":        "brown Chevrolet Impala",
    "vehicle.dodge.charger_2020":     "black Dodge Charger",
    "vehicle.dodge.charger_police":   "police cruiser",
    "vehicle.dodge.charger_police_2020": "police SUV",
    "vehicle.lincoln.mkz_2017":       "gray Lincoln sedan",
    "vehicle.lincoln.mkz_2020":       "white Lincoln sedan",
    "vehicle.toyota.prius":           "white Toyota Prius",
    "vehicle.citroen.c3":             "blue Citroen C3",
    "vehicle.seat.leon":              "gray Seat Leon",
    "vehicle.carlamotors.carlacola":  "blue delivery van",
    "vehicle.carlamotors.european_hgv": "white cargo truck",
    "vehicle.mercedes.sprinter":      "white Sprinter van",
    "vehicle.mitsubishi.fusorosa":    "white bus",
    "vehicle.volkswagen.t2":          "blue VW camper van",
    "vehicle.volkswagen.t2_2021":     "orange VW camper van",
    "vehicle.micro.microlino":        "tiny white microcar",
    "vehicle.carlamotors.firetruck":  "red fire truck",
    "vehicle.vespa.zx125":            "red Vespa scooter",
    "vehicle.yamaha.yzf":             "blue Yamaha motorcycle",
    "vehicle.harley-davidson.low_rider": "black Harley-Davidson",
    "vehicle.kawasaki.ninja":         "green Kawasaki Ninja",
    "vehicle.bh.crossbike":           "mountain bike",
    "vehicle.diamondback.century":    "road bicycle",
    "vehicle.gazelle.omafiets":       "city bicycle",
}

# Cars only (no bikes, no emergency vehicles) — suitable as target or distractor
CARS_ONLY = [k for k in VEHICLE_DESCRIPTIONS
             if "motorcycle" not in VEHICLE_DESCRIPTIONS[k].lower()
             and "bike" not in VEHICLE_DESCRIPTIONS[k].lower()
             and "bicycle" not in VEHICLE_DESCRIPTIONS[k].lower()
             and "vespa" not in VEHICLE_DESCRIPTIONS[k].lower()
             and "ambulance" not in VEHICLE_DESCRIPTIONS[k].lower()
             and "fire" not in VEHICLE_DESCRIPTIONS[k].lower()
             and "police" not in VEHICLE_DESCRIPTIONS[k].lower()]

# Templates for multi-vehicle scenarios (must identify specific target)
MULTI_TEMPLATES = [
    "Track the {target}. There are other vehicles on the road — do not confuse them.",
    "Your target is the {target}. Ignore the other traffic.",
    "Find and follow the {target} among the surrounding vehicles.",
    "Keep your eyes on the {target}. Multiple cars are nearby.",
    "The {target} is your objective. Stay locked on it despite nearby traffic.",
    "Among the vehicles below, track only the {target}.",
    "Monitor the {target}. There may be similar-looking cars nearby — focus on the correct one.",
    "Follow the {target} as it navigates through traffic.",
    "The {target} is moving through urban traffic. Maintain continuous tracking.",
]

# Templates for the scene context
CONTEXT_TEMPLATES = [
    "It is currently on a {road_type} heading {direction}.",
    "The vehicle is traveling {direction} in {lane_info}.",
    "It's driving {direction} on a multi-lane road.",
]


@dataclass
class LanguageConfig:
    templates: list[str] | None = None
    descriptions: dict[str, str] | None = None


class LanguageGenerator:
    """Generate diverse tracking instructions for multi-vehicle VLA training."""

    def __init__(self, config: LanguageConfig | None = None):
        self._cfg = config or LanguageConfig()
        self._templates = self._cfg.templates or MULTI_TEMPLATES
        self._descriptions = self._cfg.descriptions or VEHICLE_DESCRIPTIONS

    def generate(
        self,
        target_bp: str,
        distractor_bps: list[str] | None = None,
        direction: str = "",
        road_type: str = "urban road",
        lane_info: str = "the right lane",
    ) -> str:
        """Generate instruction to track a specific target among distractors."""
        target = self._descriptions.get(
            target_bp,
            target_bp.replace("vehicle.", "").replace(".", " "),
        )

        # Pick a template
        instruction = random.choice(self._templates).format(target=target)

        # Add context (50% of the time)
        if direction and random.random() < 0.5:
            context = random.choice(CONTEXT_TEMPLATES).format(
                road_type=road_type, direction=direction, lane_info=lane_info,
            )
            instruction += " " + context

        return instruction

    def describe_target(self, blueprint_name: str) -> str:
        return self._descriptions.get(
            blueprint_name,
            blueprint_name.replace("vehicle.", "").replace(".", " "),
        )

    def get_distractors(self, target_bp: str, count: int) -> list[str]:
        """Pick N vehicle BPs that are visually distinct from the target."""
        available = [b for b in CARS_ONLY if b != target_bp]
        return random.sample(available, min(count, len(available)))
