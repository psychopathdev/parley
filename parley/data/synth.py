"""Procedural synthetic dataset generator.

The generator emits :class:`Episode` rows whose audio is the codec-encoded
form of the instruction text. Pairing this generator with the codec
frontend gives a *closed-loop* benchmark where:

  audio = codec.encode(instruction)
  recovered = codec.decode(perturb(audio))

so any non-zero WER is purely the result of the perturbation. This is
the toolkit's main credibility property — and it is *easy* to extend to
real audio later by swapping the generator.

Templates form a tiny grammar:

  pick the <COLOR> <SHAPE>
  pick the <COLOR> <SHAPE> to the <DIRECTION>
  place the <COLOR> <SHAPE> to the <DIRECTION>
  push the <COLOR> <SHAPE> to the <DIRECTION>

Scenes are 2-3 objects sampled from the cross-product of colors and shapes
so the target referred to by the instruction is unambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from parley.core.types import GoalSpec, Instruction, SceneSpec
from parley.data.schema import Episode
from parley.speech._codec import CodecConfig, encode

COLORS: tuple[str, ...] = ("red", "blue", "green", "yellow")
SHAPES: tuple[str, ...] = ("cube", "sphere", "block", "ball")
DIRECTIONS: tuple[str, ...] = ("left", "right", "center")
VERBS: tuple[str, ...] = ("pick", "place", "push")


@dataclass
class SynthConfig:
    """Knobs for :func:`generate_dataset`."""

    n_episodes: int = 32
    sample_rate: int = 16_000
    seed: int = 0
    objects_per_scene: int = 3
    include_directions: bool = True


def vocab_for(cfg: SynthConfig) -> tuple[str, ...]:
    """Closed lexicon used by both the synth audio encoder and the codec ASR.

    Exposed so callers (CLI, tests) can pass it into ``CodecSpeechFrontend``.
    """
    fillers = ("the", "to", "a", "and", "an")
    accent_subs = ("da", "ta", "uh", "ya", "gonna")
    disfluencies = ("uhm", "uh", "er", "like", "you")
    extras = ("know",)
    base = (*VERBS, *COLORS, *SHAPES, *DIRECTIONS, *fillers)
    # Include perturbation-emitted tokens in the closed vocab so the codec
    # can still encode/decode perturbed text deterministically.
    return tuple(dict.fromkeys((*base, *accent_subs, *disfluencies, *extras)))


def generate_dataset(cfg: SynthConfig) -> list[Episode]:
    """Build ``cfg.n_episodes`` deterministic synthetic :class:`Episode` rows."""

    rng = np.random.default_rng(cfg.seed)
    vocab = vocab_for(cfg)
    codec = CodecConfig(vocab=vocab, sample_rate=cfg.sample_rate)
    episodes: list[Episode] = []

    for ep_idx in range(cfg.n_episodes):
        # Build a scene with non-overlapping objects and a unique target.
        scene_objects, target_obj = _sample_scene(rng, cfg.objects_per_scene)
        verb = VERBS[int(rng.integers(0, len(VERBS)))]
        if verb == "pick":
            text = f"pick the {target_obj['color']} {target_obj['shape']}"
            destination: str | None = None
        else:
            # place/push always carry a destination so the success predicate
            # is satisfiable. include_directions=False degrades them to picks.
            if not cfg.include_directions:
                verb = "pick"
                text = f"pick the {target_obj['color']} {target_obj['shape']}"
                destination = None
            else:
                destination = DIRECTIONS[int(rng.integers(0, len(DIRECTIONS)))]
                text = (
                    f"{verb} the {target_obj['color']} {target_obj['shape']} to the {destination}"
                )
        instruction = Instruction(text=text, reference=text, language="en")
        audio = encode(text, codec)
        scene = SceneSpec(objects=tuple(scene_objects))
        goal = GoalSpec(
            verb=verb,
            target=f"{target_obj['color']} {target_obj['shape']}",
            destination=destination,
        )
        episodes.append(
            Episode(
                episode_id=f"ep-{ep_idx:05d}",
                instruction=instruction,
                audio=audio,
                sample_rate=cfg.sample_rate,
                scene=scene,
                goal=goal,
                metadata={"template_verb": verb, "vocab_size": len(vocab)},
            )
        )

    return episodes


def _sample_scene(
    rng: np.random.Generator,
    n_objects: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Sample ``n_objects`` distinct (color, shape) pairs at distinct positions.

    Distinctness ensures that "the red cube" is unambiguous in the scene.
    """
    if n_objects > len(COLORS) * len(SHAPES):
        raise ValueError(
            f"objects_per_scene={n_objects} exceeds {len(COLORS) * len(SHAPES)} "
            "unique (color, shape) pairs"
        )
    seen: set[tuple[str, str]] = set()
    objects: list[dict[str, object]] = []
    while len(objects) < n_objects:
        c = COLORS[int(rng.integers(0, len(COLORS)))]
        s = SHAPES[int(rng.integers(0, len(SHAPES)))]
        if (c, s) in seen:
            continue
        seen.add((c, s))
        # spread positions: x in [0.1, 0.9], y in [0.1, 0.9]
        pos = (float(rng.uniform(0.1, 0.9)), float(rng.uniform(0.1, 0.9)))
        objects.append({"id": f"o{len(objects)}", "color": c, "shape": s, "pos": pos})
    target = objects[int(rng.integers(0, len(objects)))]
    return objects, target
