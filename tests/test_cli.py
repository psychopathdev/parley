"""CLI smoke tests via Typer's runner."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from parley.cli import app

runner = CliRunner()


def test_cli_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for sub in ("synth", "run", "report", "list", "validate"):
        assert sub in result.output


def test_cli_list_metrics() -> None:
    result = runner.invoke(app, ["list", "--kind", "metric"])
    assert result.exit_code == 0
    assert "wer" in result.output
    assert "success_rate" in result.output


def test_cli_list_unknown_kind_errors() -> None:
    result = runner.invoke(app, ["list", "--kind", "nope"])
    assert result.exit_code == 2


def test_cli_synth_writes_dataset(tmp_path: Path) -> None:
    out = tmp_path / "ds.jsonl"
    result = runner.invoke(app, ["synth", "--out", str(out), "--episodes", "3"])
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert (tmp_path / "ds.audio.npz").exists()


@pytest.mark.integration
def test_cli_synth_run_report_round_trip(tmp_path: Path) -> None:
    """End-to-end: synth -> run -> report --format json."""
    ds = tmp_path / "ds.jsonl"
    runs = tmp_path / "runs"
    runner.invoke(app, ["synth", "--out", str(ds), "--episodes", "3"])

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        f"""
name: t
seed: 0
dataset: {{ source: file, path: {ds}, episodes: 3 }}
env: {{ name: tabletop }}
pipelines:
  - name: scripted
    speech: {{ name: codec }}
    policy: {{ name: scripted }}
metrics: [ wer, success_rate ]
runner: {{ max_steps: 25, workers: 1 }}
output_dir: {runs}
""",
        encoding="utf-8",
    )

    val = runner.invoke(app, ["validate", str(cfg)])
    assert val.exit_code == 0

    run_res = runner.invoke(app, ["run", str(cfg)])
    assert run_res.exit_code == 0, run_res.output
    assert (runs / "report.json").exists()

    rep = runner.invoke(app, ["report", str(runs), "--format", "json"])
    assert rep.exit_code == 0
    assert '"schema_version"' in rep.output
