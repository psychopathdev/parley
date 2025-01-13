"""A rule-based grounder for the tabletop instruction templates.

The grammar Parley's synthetic dataset emits is intentionally small:

  <VERB> [the] [<COLOR>] [<SHAPE>] [to the <DIRECTION>]

where VERB ∈ {pick, place, push}, COLOR ∈ {red, blue, green, yellow},
SHAPE ∈ {cube, sphere, block, ball}, DIRECTION ∈ {left, right, center}.
This module walks the recognized tokens and fills the slots greedily.
Unknown tokens are skipped — the codec frontend can produce garbage
under heavy perturbation, and silent skipping lets the downstream
metric capture the *quality* of the parse on what survived.
"""

from __future__ import annotations

from parley.core.registry import registry
from parley.core.types import Grounding, Transcript
from parley.grounding.base import Grounder

VERBS = frozenset({"pick", "place", "push", "pickup", "move"})
COLORS = frozenset({"red", "blue", "green", "yellow", "orange", "purple"})
SHAPES = frozenset({"cube", "sphere", "block", "ball", "cylinder"})
DIRECTIONS = frozenset({"left", "right", "center", "forward", "back"})

# Map common variants to a canonical form.
_VERB_CANON = {"pickup": "pick", "move": "push"}
_DIR_CANON = {"forward": "front"}


@registry.grounding.register("rule_based")
class RuleBasedGrounder(Grounder):
    """Greedy slot-filler over the tabletop grammar."""

    name = "rule_based"

    def ground(self, transcript: Transcript) -> Grounding:
        tokens = list(transcript.tokens) or transcript.text.lower().split()
        verb: str | None = None
        color: str | None = None
        shape: str | None = None
        direction: str | None = None

        for t in tokens:
            t = t.lower()
            if verb is None and t in VERBS:
                verb = _VERB_CANON.get(t, t)
            elif color is None and t in COLORS:
                color = t
            elif shape is None and t in SHAPES:
                shape = t
            elif direction is None and t in DIRECTIONS:
                direction = _DIR_CANON.get(t, t)

        # A missing verb is a parse failure; emit a sentinel so downstream
        # metrics can count it without crashing.
        if verb is None:
            return Grounding(verb="<unknown>", confidence=0.0)

        # Target is "color shape" if both are known, else whichever exists.
        target = f"{color} {shape}" if color and shape else (color or shape)

        # Confidence: 0.25 per filled slot — purely indicative, not calibrated.
        filled = sum(x is not None for x in (verb, color, shape, direction))
        confidence = 0.25 * filled

        slots: dict[str, str] = {}
        if color:
            slots["color"] = color
        if shape:
            slots["shape"] = shape
        if direction:
            slots["direction"] = direction

        return Grounding(
            verb=verb,
            target=target,
            modifier=color,
            destination=direction,
            slots=slots,
            confidence=confidence,
        )
