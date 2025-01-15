"""On-disk dataset format: a jsonl index next to an npz audio blob.

Why two files: jsonl is human-diffable and streamable for index-only
operations (e.g. count episodes, list goals); npz keeps audio compact and
random-access. Pairing them lets ``parley run`` mmap audio for one
episode at a time without loading the whole dataset.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from parley.core.errors import ValidationError
from parley.core.types import GoalSpec, Instruction, SceneSpec
from parley.data.schema import Episode

_INDEX_SUFFIX = ".jsonl"
_AUDIO_SUFFIX = ".audio.npz"
_FORMAT_VERSION = 1


def save_episodes(episodes: Iterable[Episode], path: str | Path) -> None:
    """Write ``episodes`` to ``<path>`` (index) + ``<path>.audio.npz`` (audio)."""

    p = Path(path)
    if p.suffix != _INDEX_SUFFIX:
        p = p.with_suffix(_INDEX_SUFFIX)
    audio_path = p.with_name(p.stem + _AUDIO_SUFFIX)
    p.parent.mkdir(parents=True, exist_ok=True)
    audio_blob: dict[str, np.ndarray] = {}
    with p.open("w", encoding="utf-8") as f:
        for ep in episodes:
            audio_blob[ep.episode_id] = ep.audio
            f.write(
                json.dumps(
                    {
                        "format_version": _FORMAT_VERSION,
                        "episode_id": ep.episode_id,
                        "instruction": {
                            "text": ep.instruction.text,
                            "reference": ep.instruction.reference,
                            "language": ep.instruction.language,
                            "metadata": ep.instruction.metadata,
                        },
                        "sample_rate": ep.sample_rate,
                        "scene": {
                            "objects": list(ep.scene.objects),
                            "workspace": list(ep.scene.workspace),
                            "metadata": ep.scene.metadata,
                        },
                        "goal": {
                            "verb": ep.goal.verb,
                            "target": ep.goal.target,
                            "modifier": ep.goal.modifier,
                            "destination": ep.goal.destination,
                            "extras": ep.goal.extras,
                        },
                        "metadata": ep.metadata,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    np.savez_compressed(audio_path, **audio_blob)  # type: ignore[arg-type]


def _normalize_object(o: dict[str, Any]) -> dict[str, Any]:
    """Re-tuple ``pos`` (and any sequence fields) after a JSON round trip."""
    out = dict(o)
    pos = out.get("pos")
    if isinstance(pos, list):
        out["pos"] = tuple(pos)
    return out


def load_episodes(path: str | Path) -> list[Episode]:
    """Inverse of :func:`save_episodes`. Audio is fully read into memory.

    For very large datasets we'd want a streaming version; for the v0
    benchmark suite (a few thousand episodes max) eager loading is fine.
    """
    p = Path(path)
    if p.suffix != _INDEX_SUFFIX:
        p = p.with_suffix(_INDEX_SUFFIX)
    if not p.exists():
        raise FileNotFoundError(f"index not found: {p}")
    audio_path = p.with_name(p.stem + _AUDIO_SUFFIX)
    if not audio_path.exists():
        raise FileNotFoundError(f"audio blob not found: {audio_path}")
    audio_blob = np.load(audio_path)

    episodes: list[Episode] = []
    with p.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            try:
                row: dict[str, Any] = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"{p}:{lineno}: invalid JSON") from exc
            ep_id = row["episode_id"]
            if ep_id not in audio_blob.files:
                raise ValidationError(f"{p}:{lineno}: episode {ep_id!r} missing from audio blob")
            episodes.append(
                Episode(
                    episode_id=ep_id,
                    instruction=Instruction(
                        text=row["instruction"]["text"],
                        reference=row["instruction"].get("reference"),
                        language=row["instruction"].get("language", "en"),
                        metadata=row["instruction"].get("metadata", {}),
                    ),
                    audio=audio_blob[ep_id],
                    sample_rate=int(row["sample_rate"]),
                    scene=SceneSpec(
                        objects=tuple(_normalize_object(o) for o in row["scene"]["objects"]),
                        workspace=tuple(row["scene"].get("workspace", [0.0, 0.0, 1.0, 1.0])),
                        metadata=row["scene"].get("metadata", {}),
                    ),
                    goal=GoalSpec(
                        verb=row["goal"]["verb"],
                        target=row["goal"].get("target"),
                        modifier=row["goal"].get("modifier"),
                        destination=row["goal"].get("destination"),
                        extras=row["goal"].get("extras", {}),
                    ),
                    metadata=row.get("metadata", {}),
                )
            )
    return episodes
