"""Tests for the YAML config loader and pydantic models."""

from __future__ import annotations

from pathlib import Path

import pytest

from parley.core.config import (
    BenchmarkConfig,
    PipelineConfig,
    PluginSpec,
    dump_config,
    load_config,
)
from parley.core.errors import ConfigError

_MINIMAL_YAML = """
name: smoke
seed: 0
pipelines:
  - name: codec+scripted
    speech: { name: codec, params: { vocab_size: 128 } }
    policy: { name: scripted }
"""


def _write(tmp_path: Path, body: str, fname: str = "cfg.yaml") -> Path:
    p = tmp_path / fname
    p.write_text(body, encoding="utf-8")
    return p


def test_minimal_yaml_loads(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, _MINIMAL_YAML))
    assert cfg.name == "smoke"
    assert cfg.pipelines[0].speech.name == "codec"
    assert cfg.pipelines[0].grounding.name == "rule_based"  # default
    assert cfg.metrics  # default populated


def test_missing_file_raises_configerror(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_yaml_syntax_error_raises_configerror(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="YAML parse"):
        load_config(_write(tmp_path, "this is: : not: valid"))


def test_top_level_must_be_mapping(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="mapping"):
        load_config(_write(tmp_path, "- 1\n- 2\n"))


def test_unknown_field_rejected(tmp_path: Path) -> None:
    body = _MINIMAL_YAML + "this_does_not_exist: 1\n"
    with pytest.raises(ConfigError, match=r"(Extra|forbidden|unexpected)"):
        load_config(_write(tmp_path, body))


def test_round_trip() -> None:
    cfg = BenchmarkConfig(
        name="t",
        pipelines=[
            PipelineConfig(
                name="p",
                speech=PluginSpec(name="mock"),
                policy=PluginSpec(name="scripted"),
            )
        ],
    )
    text = dump_config(cfg)
    assert "name: t" in text
    assert "speech:" in text
