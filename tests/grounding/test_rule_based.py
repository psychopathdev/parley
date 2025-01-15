"""Tests for the rule-based grounder."""

from __future__ import annotations

from parley.core.types import Grounding, Transcript
from parley.grounding import RuleBasedGrounder


def _ground(text: str) -> Grounding:
    return RuleBasedGrounder().ground(Transcript(text=text, tokens=tuple(text.split())))


def test_full_instruction() -> None:
    g = _ground("pick the red cube to the left")
    assert g.verb == "pick"
    assert g.target == "red cube"
    assert g.destination == "left"
    assert g.slots == {"color": "red", "shape": "cube", "direction": "left"}


def test_minimal_verb_only() -> None:
    g = _ground("pick")
    assert g.verb == "pick"
    assert g.target is None
    assert g.destination is None


def test_canonical_verb_form() -> None:
    g = _ground("pickup the blue sphere")
    assert g.verb == "pick"
    assert g.target == "blue sphere"


def test_unknown_returns_sentinel() -> None:
    g = _ground("foo bar baz")
    assert g.verb == "<unknown>"
    assert g.confidence == 0.0


def test_confidence_scales_with_slots() -> None:
    g0 = _ground("pick")
    g3 = _ground("pick the red cube to the left")
    assert g3.confidence is not None
    assert g0.confidence is not None
    assert g3.confidence > g0.confidence


def test_partial_target_with_color_only() -> None:
    g = _ground("pick the red")
    assert g.target == "red"
    assert g.slots == {"color": "red"}
