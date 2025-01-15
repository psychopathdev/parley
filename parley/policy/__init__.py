"""VLA policies — anything that maps (observation, history) -> action.

Parley ships three reference policies:

* :class:`ScriptedPolicy` — an oracle-ish policy that reads the parsed
  Grounding and the current scene, locates the target object, and emits
  a near-optimal action sequence. Provides the success-rate ceiling.

* :class:`NoisyPolicy` — wraps another policy and perturbs its actions.
  Useful for measuring metric sensitivity / failure-rate slope.

* :class:`RandomPolicy` — uniform random actions. The floor baseline.

Real VLA models (OpenVLA, Octo, π0) plug in via the same
:class:`VLAPolicy` protocol, with the runner translating their action
chunks into the env's action space.
"""

from __future__ import annotations

from parley.policy.base import VLAPolicy
from parley.policy.noisy import NoisyPolicy
from parley.policy.random import RandomPolicy
from parley.policy.scripted import ScriptedPolicy

__all__ = ["NoisyPolicy", "RandomPolicy", "ScriptedPolicy", "VLAPolicy"]
