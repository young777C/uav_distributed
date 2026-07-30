"""Auto-generate natural language tracking instructions for multi-vehicle scenarios.

Generates descriptions that uniquely identify a TARGET among DISTRACTORS,
forcing the VLA to use both visual recognition AND language understanding.

Core design (design doc §5.1 / §6.3 / §7.4): every scene contains >=2 distractors
that are *visually similar* to the target, so appearance alone cannot disambiguate
and the language must carry the discriminating attribute. Two similarity strategies:

  - same_color_diff_shape : target + look-alikes share a COLOR, differ in MODEL
                            → language must name the model  ("the red Ford Mustang")
  - same_shape_diff_color : target + look-alikes share a MODEL, differ in COLOR
                            → language must name the color  ("the blue Model 3")

Target classes: cars, motorcycles, scooters/e-bikes (电动), bicycles (非机动车),
and pedestrians (行人).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Blueprint → visual description (color + type). CARLA meshes have baked colors.
# ---------------------------------------------------------------------------
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
    "vehicle.jeep.wrangler_rubicon":  "orange Jeep Wrangler",
    "vehicle.ford.mustang":           "red Ford Mustang",
    "vehicle.chevrolet.impala":       "brown Chevrolet Impala",
    "vehicle.dodge.charger_2020":     "black Dodge Charger",
    "vehicle.lincoln.mkz_2017":       "gray Lincoln sedan",
    "vehicle.lincoln.mkz_2020":       "white Lincoln sedan",
    "vehicle.toyota.prius":           "white Toyota Prius",
    "vehicle.citroen.c3":             "blue Citroen C3",
    "vehicle.seat.leon":              "gray Seat Leon",
    # vans / trucks / buses
    "vehicle.carlamotors.carlacola":  "blue delivery van",
    "vehicle.carlamotors.european_hgv": "white cargo truck",
    "vehicle.mercedes.sprinter":      "white Sprinter van",
    "vehicle.mitsubishi.fusorosa":    "white bus",
    "vehicle.volkswagen.t2":          "blue VW camper van",
    "vehicle.volkswagen.t2_2021":     "orange VW camper van",
    "vehicle.micro.microlino":        "tiny white microcar",
    # motorcycles (摩托)
    "vehicle.yamaha.yzf":             "blue Yamaha sport motorcycle",
    "vehicle.harley-davidson.low_rider": "black Harley-Davidson motorcycle",
    "vehicle.kawasaki.ninja":         "green Kawasaki Ninja motorcycle",
    # scooter / electric (电动)
    "vehicle.vespa.zx125":            "red Vespa scooter",
    # bicycles (非机动车)
    "vehicle.bh.crossbike":           "black mountain bike",
    "vehicle.diamondback.century":    "gray road bicycle",
    "vehicle.gazelle.omafiets":       "black city bicycle",
}

# ---------------------------------------------------------------------------
# Class taxonomy (drives target selection + same-class similarity)
# ---------------------------------------------------------------------------
CARS = [
    "vehicle.audi.a2", "vehicle.audi.tt", "vehicle.tesla.model3",
    "vehicle.bmw.grandtourer", "vehicle.mercedes.coupe", "vehicle.mercedes.coupe_2020",
    "vehicle.mini.cooper_s", "vehicle.mini.cooper_s_2021", "vehicle.nissan.micra",
    "vehicle.jeep.wrangler_rubicon", "vehicle.ford.mustang", "vehicle.chevrolet.impala",
    "vehicle.dodge.charger_2020", "vehicle.lincoln.mkz_2017", "vehicle.lincoln.mkz_2020",
    "vehicle.toyota.prius", "vehicle.citroen.c3", "vehicle.seat.leon",
    "vehicle.nissan.patrol", "vehicle.nissan.patrol_2021", "vehicle.audi.etron",
]
MOTORCYCLES = ["vehicle.yamaha.yzf", "vehicle.harley-davidson.low_rider", "vehicle.kawasaki.ninja"]
SCOOTERS = ["vehicle.vespa.zx125"]                    # 电动 / 踏板
BICYCLES = ["vehicle.bh.crossbike", "vehicle.diamondback.century", "vehicle.gazelle.omafiets"]  # 非机动车
PEDESTRIANS = [f"walker.pedestrian.{i:04d}" for i in range(1, 15)]  # 行人

CLASS_POOLS = {
    "car": CARS,
    "motorcycle": MOTORCYCLES,
    "scooter": SCOOTERS,
    "bicycle": BICYCLES,
    "pedestrian": PEDESTRIANS,
}
# classes whose 'color' attribute is reliably settable (all wheeled vehicles have it)
COLORABLE = set(CARS) | set(MOTORCYCLES) | set(SCOOTERS) | set(BICYCLES)
# classes driven by CARLA autopilot (vehicles) vs walker AI (pedestrians)
VEHICLE_CLASSES = {"car", "motorcycle", "scooter", "bicycle"}

# Back-compat alias used elsewhere in the codebase
CARS_ONLY = CARS

COLOR_WORDS = {
    "silver", "red", "white", "black", "green", "yellow", "dark", "light",
    "blue", "orange", "brown", "gray", "grey", "royal", "beige", "purple", "gold",
}
# 18 natural-language colours (UAV-Track VLA style) -> anchor RGB.
COLOR_ANCHORS = {
    "black": (15, 15, 18), "white": (240, 240, 240), "gray": (128, 128, 128),
    "silver": (190, 192, 195), "red": (180, 20, 20), "dark red": (110, 15, 15),
    "orange": (220, 100, 20), "yellow": (225, 205, 30), "green": (30, 140, 45),
    "dark green": (18, 70, 30), "blue": (30, 80, 200), "light blue": (90, 160, 220),
    "royal blue": (30, 55, 180), "dark blue": (18, 30, 90), "brown": (95, 60, 35),
    "beige": (200, 180, 140), "purple": (110, 50, 130), "gold": (200, 160, 40),
}
# name -> CARLA 'color' attribute string "R,G,B"
COLOR_RGB = {n: f"{r},{g},{b}" for n, (r, g, b) in COLOR_ANCHORS.items()}
PALETTE = list(COLOR_ANCHORS)
_ACHROMATIC = ("black", "white", "gray", "silver")


def rgb_to_name(rgb) -> str:
    """Map an RGB colour to the nearest of the 18 names (paper's method: filter
    black/white/gray by luminance first, then nearest chromatic anchor)."""
    if isinstance(rgb, str):
        r, g, b = (int(x) for x in rgb.split(","))
    else:
        r, g, b = rgb
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    if max(r, g, b) - min(r, g, b) < 25:          # achromatic
        if lum < 50:
            return "black"
        if lum > 205:
            return "white"
        return "gray" if lum < 150 else "silver"
    best, bestd = "gray", 1e18                     # nearest chromatic anchor
    for name, (ar, ag, ab) in COLOR_ANCHORS.items():
        if name in _ACHROMATIC:
            continue
        d = (r - ar) ** 2 + (g - ag) ** 2 + (b - ab) ** 2
        if d < bestd:
            best, bestd = name, d
    return best

# Instruction = [Verb] + [target attribute] + [distance constraint] (+ disambiguation),
# following UAV-Track VLA's structure. Verb + distance are paraphrase-augmented.
VERBS = ["Track", "Follow", "Keep tracking", "Focus on", "Pursue",
         "Keep an eye on", "Lock onto", "Stay locked on"]
DISTANCE_PHRASES = {
    "close":    ["at a close distance", "at close range", "nearby"],
    "suitable": ["at a suitable distance", "at a moderate distance"],
    "long":     ["at a long distance", "from afar", "from a long range"],
}
DISAMBIG_CLAUSES = [
    "other similar vehicles are nearby — follow the correct one",
    "do not confuse it with the look-alikes",
    "ignore the other similar-looking vehicles around it",
    "several look-alikes are on the road; stay on the right one",
    "do not be fooled by the vehicles that look alike",
]


def _norm_angle(a: float) -> float:
    return ((a + 180.0) % 360.0) - 180.0


def spatial_and_distance(ux, uy, uav_yaw_deg, tx, ty, uz=0.0, tz=0.0):
    """Return (spatial_phrase, 3D slant distance_m) of the target relative to the UAV
    camera heading — e.g. 'in the front-left'. Computed from actual geometry."""
    dx, dy = tx - ux, ty - uy
    yaw = math.radians(uav_yaw_deg)
    fwd = dx * math.cos(yaw) + dy * math.sin(yaw)        # along camera forward
    lat = -dx * math.sin(yaw) + dy * math.cos(yaw)       # + = to the UAV's right
    dist = math.hypot(math.hypot(dx, dy), uz - tz)       # 3D slant range
    if fwd < 0:
        return "behind you", dist
    ang = math.degrees(math.atan2(lat, fwd))
    if ang < -18:
        sp = "in the front-left"
    elif ang > 18:
        sp = "in the front-right"
    else:
        sp = "directly ahead"
    return sp, dist


def motion_phrase(tyaw, tspeed):
    """Describe the target's imminent motion from its first-seconds trajectory
    (accurate: derived post-hoc from the recorded path). tyaw/tspeed are arrays."""
    n = len(tyaw)
    if n < 6:
        return ""
    dyaw = _norm_angle(float(tyaw[-1]) - float(tyaw[0]))
    v0 = float(sum(tspeed[:3]) / 3.0)
    v1 = float(sum(tspeed[-3:]) / 3.0)
    if dyaw < -25:
        return "which is about to turn left"
    if dyaw > 25:
        return "which is about to turn right"
    if v0 > 5.0 and v1 < 1.0:
        return "which is about to stop"
    if v0 < 1.0 and v1 > 5.0:
        return "which is about to accelerate away"
    return ""


@dataclass
class LanguageConfig:
    templates: list[str] | None = None
    descriptions: dict[str, str] | None = None


class LanguageGenerator:
    """Compose multi-actor scenes + tracking instructions for VLA training."""

    def __init__(self, config: LanguageConfig | None = None):
        self._cfg = config or LanguageConfig()
        self._templates = self._cfg.templates
        self._descriptions = self._cfg.descriptions or VEHICLE_DESCRIPTIONS

    # ------------------------------------------------------------------
    # Descriptions
    # ------------------------------------------------------------------
    def _type_noun(self, bp: str) -> str:
        """Description with any leading color word stripped ('Tesla Model 3 sedan')."""
        d = self._descriptions.get(bp, bp.replace("vehicle.", "").replace(".", " "))
        head, _, tail = d.partition(" ")
        return tail if (head in COLOR_WORDS and tail) else d

    def describe(self, bp: str, color: str | None = None, cls: str = "car") -> str:
        if cls == "pedestrian":
            return "pedestrian"
        if color:
            return f"{color} {self._type_noun(bp)}"
        return self._descriptions.get(bp, bp.replace("vehicle.", "").replace(".", " "))

    def describe_target(self, blueprint_name: str) -> str:  # back-compat
        return self.describe(blueprint_name)

    # ------------------------------------------------------------------
    # Scene composition (the core anti-shortcut design)
    # ------------------------------------------------------------------
    def compose_scene(
        self,
        target_class: str,
        num_distractors: int,
        strategy: str = "same_color_diff_shape",
        min_similar: int = 2,
        rng=None,
    ) -> dict:
        """Return a full actor plan: 1 target + N distractors with >=min_similar look-alikes.

        Returns dict:
            {
              "target": {"bp","color","desc","cls"},
              "distractors": [{"bp","color","desc","cls","similar":bool}, ...],
              "strategy": str, "num_similar": int,
            }
        color is None when the blueprint's baked color is used (no override).
        """
        rng = rng or random

        # Pedestrians / small non-car classes: similarity is by CLASS (same kind),
        # since their color/appearance is not reliably controllable.
        if target_class not in {"car"} or strategy == "same_class":
            return self._compose_same_class(target_class, num_distractors, min_similar, rng)
        if strategy == "same_shape_diff_color":
            return self._compose_same_shape(num_distractors, min_similar, rng)
        if strategy == "distinct":
            return self._compose_distinct(target_class, num_distractors, rng)
        return self._compose_same_color(num_distractors, min_similar, rng)  # default

    def _compose_same_color(self, n: int, min_similar: int, rng) -> dict:
        color = rng.choice(PALETTE)
        target_bp = rng.choice(CARS)
        k = min(rng.randint(min_similar, min_similar + 1), n)
        # look-alikes: DIFFERENT models, all painted the same color
        others = [b for b in CARS if b != target_bp]
        rng.shuffle(others)
        distractors = [{"bp": b, "color": color, "cls": "car", "similar": True}
                       for b in others[:k]]
        # filler: random distinct cars/classes with baked colors
        distractors += self._fillers(n - k, exclude={target_bp}, rng=rng)
        for d in distractors:
            d["desc"] = self.describe(d["bp"], d.get("color"), d["cls"])
        return {
            "target": {"bp": target_bp, "color": color, "cls": "car",
                       "desc": self.describe(target_bp, color, "car")},
            "distractors": distractors, "strategy": "same_color_diff_shape",
            "num_similar": k,
        }

    def _compose_same_shape(self, n: int, min_similar: int, rng) -> dict:
        target_bp = rng.choice(CARS)
        colors = rng.sample(PALETTE, min(len(PALETTE), min_similar + 2))
        target_color = colors[0]
        k = min(rng.randint(min_similar, min_similar + 1), n)
        # look-alikes: SAME model, different colors
        distractors = [{"bp": target_bp, "color": colors[1 + i % (len(colors) - 1)],
                        "cls": "car", "similar": True} for i in range(k)]
        distractors += self._fillers(n - k, exclude={target_bp}, rng=rng)
        for d in distractors:
            d["desc"] = self.describe(d["bp"], d.get("color"), d["cls"])
        return {
            "target": {"bp": target_bp, "color": target_color, "cls": "car",
                       "desc": self.describe(target_bp, target_color, "car")},
            "distractors": distractors, "strategy": "same_shape_diff_color",
            "num_similar": k,
        }

    def _compose_distinct(self, target_class: str, n: int, rng) -> dict:
        target_bp = rng.choice(CLASS_POOLS[target_class])
        tcol = rng.choice(PALETTE) if target_class in VEHICLE_CLASSES else None
        distractors = self._fillers(n, exclude={target_bp}, rng=rng)
        for d in distractors:
            d["desc"] = self.describe(d["bp"], d.get("color"), d["cls"])
        return {
            "target": {"bp": target_bp, "color": tcol, "cls": target_class,
                       "desc": self.describe(target_bp, tcol, target_class)},
            "distractors": distractors, "strategy": "distinct", "num_similar": 0,
        }

    def _compose_same_class(self, target_class: str, n: int, min_similar: int, rng) -> dict:
        pool = CLASS_POOLS[target_class]
        target_bp = rng.choice(pool)
        k = min(max(min_similar, 2), n)
        # two-wheelers are colourable → disambiguate look-alikes by colour
        if target_class in {"motorcycle", "scooter", "bicycle"}:
            colors = rng.sample(PALETTE, min(len(PALETTE), k + 2))
            tcol = colors[0]
            distractors = [{"bp": rng.choice(pool),
                            "color": colors[1 + i % (len(colors) - 1)],
                            "cls": target_class, "similar": True} for i in range(k)]
        else:                                # pedestrians (ambient only) — no colour
            tcol = None
            distractors = [{"bp": rng.choice(pool), "color": None,
                            "cls": target_class, "similar": True} for _ in range(k)]
        distractors += self._fillers(n - k, exclude=set(), rng=rng)
        for d in distractors:
            d["desc"] = self.describe(d["bp"], d.get("color"), d["cls"])
        return {
            "target": {"bp": target_bp, "color": tcol, "cls": target_class,
                       "desc": self.describe(target_bp, tcol, target_class)},
            "distractors": distractors, "strategy": "same_class", "num_similar": k,
        }

    def _fillers(self, count: int, exclude: set, rng) -> list[dict]:
        out = []
        for _ in range(max(0, count)):
            b = rng.choice([c for c in CARS if c not in exclude])
            out.append({"bp": b, "color": rng.choice(PALETTE), "cls": "car", "similar": False})
        return out

    # ------------------------------------------------------------------
    # Instruction text
    # ------------------------------------------------------------------
    def build_instruction(self, target_desc: str, num_similar: int = 0,
                          spatial: str = "", motion: str = "",
                          distance_m: float | None = None,
                          disambig: bool = False, rng=None) -> str:
        """Assemble a referring instruction: [action verb] + the [colour+model object]
        + [spatial relation] + [motion intent] + [distance constraint].

        The disambiguation is carried by the scene (co-visible look-alikes) + the unique
        colour/model attribute, so no explicit 'don't confuse it' clause is added by
        default (disambig=False) — adding one is a redundant shortcut that muddies the
        language-necessity analysis (§7.4)."""
        rng = rng or random
        s = f"{rng.choice(VERBS)} the {target_desc}"
        clauses = [c for c in (spatial, motion) if c]
        if clauses:
            s += " " + ", ".join(clauses)
        if distance_m and distance_m > 0:
            d = int(round(distance_m / 5.0) * 5)
            s += f", maintaining a tracking distance of about {d} m"
        else:
            s += rng.choice([" at a suitable distance", " at a close distance"])
        if disambig and num_similar >= 2 and rng.random() < 0.9:
            s += " (" + rng.choice(DISAMBIG_CLAUSES) + ")"
        return s + "."

    # ------------------------------------------------------------------
    # Back-compat shims (older callers)
    # ------------------------------------------------------------------
    def generate(self, target_bp: str, distractor_bps: list[str] | None = None,
                 **kwargs) -> str:
        return f"{random.choice(VERBS)} the {self.describe(target_bp)} at a suitable distance."

    def get_distractors(self, target_bp: str, count: int) -> list[str]:
        available = [b for b in CARS if b != target_bp]
        return random.sample(available, min(count, len(available)))
