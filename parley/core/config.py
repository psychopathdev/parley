"""Config models and YAML loader.

Configs are the user's contract with Parley: every benchmark run is fully
specified by a YAML file (plus a CLI seed override). We use pydantic v2 so
errors point at the offending key, and we wrap the loader to attach the
source path to error messages — debugging a 60-line YAML without that is
miserable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as _PydanticValidationError

from parley.core.errors import ConfigError


class _Strict(BaseModel):
    """Base for all config sections.

    ``extra='forbid'`` is a real footgun-catcher: it turns typos into
    immediate errors instead of silently-ignored fields.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


class PluginSpec(_Strict):
    """A typed reference to a registered plugin plus its kwargs."""

    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class DatasetConfig(_Strict):
    source: str = "synth"
    path: str | None = None
    episodes: int = 32
    params: dict[str, Any] = Field(default_factory=dict)


class PipelineConfig(_Strict):
    name: str
    speech: PluginSpec
    grounding: PluginSpec = PluginSpec(name="rule_based")
    policy: PluginSpec


class PerturbationGroup(_Strict):
    """A named bundle of perturbations applied as a single composite step."""

    name: str
    steps: list[PluginSpec] = Field(default_factory=list)


class EnvConfig(_Strict):
    name: str = "tabletop"
    params: dict[str, Any] = Field(default_factory=dict)


class RunnerConfig(_Strict):
    max_steps: int = 64
    workers: int = 1
    cache_dir: str | None = ".cache/parley"


class BenchmarkConfig(_Strict):
    """Top-level config for one ``parley run`` invocation."""

    name: str
    seed: int = 0
    dataset: DatasetConfig = DatasetConfig()
    env: EnvConfig = EnvConfig()
    pipelines: list[PipelineConfig]
    perturbations: list[PerturbationGroup] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=lambda: ["wer", "grounding_f1", "success_rate"])
    runner: RunnerConfig = RunnerConfig()
    output_dir: str = "runs"


def load_config(path: str | Path) -> BenchmarkConfig:
    """Load a YAML benchmark config and validate it.

    Raises :class:`ConfigError` (subclass of :class:`ParleyError`) with the
    source path embedded so users can find the offending file in a log.
    """

    p = Path(path)
    try:
        with p.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {p}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML parse error in {p}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"top-level YAML in {p} must be a mapping, got {type(raw).__name__}")

    try:
        return BenchmarkConfig(**raw)
    except _PydanticValidationError as exc:
        raise ConfigError(f"invalid config in {p}:\n{exc}") from exc


def dump_config(cfg: BenchmarkConfig) -> str:
    """Round-trip a config back to a stable YAML string.

    Used by the runner to snapshot the resolved config into each output
    directory, so a downstream reader can reproduce the run without the
    original file.
    """

    return yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False)
