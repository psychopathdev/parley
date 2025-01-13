"""Grounders — extract structured intent from a transcript.

Parley ships a single deterministic, dependency-free parser
(:class:`RuleBasedGrounder`) covering the verbs and slot shapes used by
the synthetic tabletop tasks. It's intentionally simple: real grounding
is an open research problem, and a rule-based baseline is the right thing
for a benchmark *toolkit* — it gives a reproducible reference number that
real LLM-based grounders can be compared against.
"""

from __future__ import annotations

from parley.grounding.base import Grounder
from parley.grounding.rule_based import RuleBasedGrounder

__all__ = ["Grounder", "RuleBasedGrounder"]
