"""Command-line entry point: ``parley {synth, run, report, list, validate}``.

Implementation kept thin — every subcommand is a few lines of glue over
the library. Heavy logic stays in :mod:`parley.runner`, :mod:`parley.data`
and :mod:`parley.report`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from parley._version import __version__
from parley.core.config import dump_config, load_config
from parley.core.registry import registry
from parley.data import SynthConfig, generate_dataset, load_episodes, save_episodes
from parley.report import (
    aggregate_results,
    dump_report,
    render_csv,
    render_markdown,
)
from parley.runner import BenchmarkEngine

app = typer.Typer(help="Parley — spoken-instruction VLA benchmark toolkit", no_args_is_help=True)
console = Console()


@app.callback()
def _global_options(
    version: Annotated[
        bool, typer.Option("--version", help="Show parley version and exit.")
    ] = False,
) -> None:
    if version:
        typer.echo(__version__)
        raise typer.Exit()


@app.command("synth")
def synth_cmd(
    out: Annotated[
        Path, typer.Option(..., help="Output jsonl path (an .audio.npz sibling is also written).")
    ],
    episodes: Annotated[int, typer.Option(help="Number of episodes to generate.")] = 32,
    seed: Annotated[int, typer.Option(help="RNG seed (reproducible).")] = 0,
    sample_rate: Annotated[int, typer.Option(help="Sample rate of generated audio (Hz).")] = 16_000,
) -> None:
    """Generate a synthetic dataset and write it to disk."""
    cfg = SynthConfig(n_episodes=episodes, seed=seed, sample_rate=sample_rate)
    eps = generate_dataset(cfg)
    save_episodes(eps, out)
    console.print(f"Wrote [bold]{len(eps)}[/bold] episodes to [cyan]{out}[/cyan]")


@app.command("run")
def run_cmd(
    config: Annotated[Path, typer.Argument(help="Path to a benchmark YAML config.")],
    dataset: Annotated[
        Path | None,
        typer.Option(help="Override the dataset path; otherwise the config's dataset is used."),
    ] = None,
    seed: Annotated[int | None, typer.Option(help="Override the config seed.")] = None,
) -> None:
    """Run a benchmark suite and write per-run traces + a JSON report."""
    cfg = load_config(config)
    if seed is not None:
        cfg = cfg.model_copy(update={"seed": seed})

    if dataset is not None:
        eps = load_episodes(dataset)
    elif cfg.dataset.source == "file" and cfg.dataset.path:
        eps = load_episodes(cfg.dataset.path)
    else:
        eps = generate_dataset(SynthConfig(n_episodes=cfg.dataset.episodes, seed=cfg.seed))

    engine = BenchmarkEngine(cfg)
    results = engine.run(eps)
    rows = aggregate_results(results, seed=cfg.seed)
    report_path = Path(cfg.output_dir) / "report.json"
    dump_report(rows, report_path, suite_name=cfg.name)

    # Persist the resolved config alongside the report.
    (Path(cfg.output_dir) / "config.resolved.yaml").write_text(dump_config(cfg), encoding="utf-8")

    console.print(
        f"\n[bold green]{cfg.name}[/bold green] - {len(results)} episodes across "
        f"{len({r.pipeline for r in results})} pipelines x "
        f"{len({r.perturbation for r in results})} perturbation groups\n"
    )
    console.print(render_markdown(rows))
    console.print(f"\nReport written to [cyan]{report_path}[/cyan]")


@app.command("report")
def report_cmd(
    runs_dir: Annotated[
        Path, typer.Argument(help="Run output directory (containing report.json).")
    ],
    fmt: Annotated[str, typer.Option("--format", "-f", help="markdown | csv | json")] = "markdown",
) -> None:
    """Re-render a previously-written report."""
    body = json.loads((runs_dir / "report.json").read_text(encoding="utf-8"))
    if fmt == "json":
        typer.echo(json.dumps(body, indent=2))
        return
    rows = aggregate_results([])  # placeholder, then we'll patch below
    # We have aggregate already in the JSON; render directly.
    if fmt == "csv":
        # Reconstitute minimal rows for CSV
        from parley.metrics.aggregate import Summary
        from parley.report.aggregate import ReportRow

        rows = []
        for r in body["rows"]:
            metrics = {
                name: Summary(
                    n=int(s.get("n", 0)),
                    mean=float(s.get("mean", 0.0)),
                    sem=float(s.get("sem", 0.0)),
                    ci_low=float(s.get("ci_low", 0.0)),
                    ci_high=float(s.get("ci_high", 0.0)),
                    confidence=float(s.get("confidence", 0.95)),
                )
                for name, s in r.get("metrics", {}).items()
            }
            rows.append(
                ReportRow(
                    pipeline=r["pipeline"],
                    perturbation=r["perturbation"],
                    n_episodes=int(r["n_episodes"]),
                    success_rate=float(r["success_rate"]),
                    metrics=metrics,
                )
            )
        typer.echo(render_csv(rows))
        return
    # markdown (default)
    table = Table(title=body.get("suite_name") or "report")
    for col in ("pipeline", "perturbation", "n", "success"):
        table.add_column(col)
    for r in body["rows"]:
        table.add_row(
            r["pipeline"], r["perturbation"], str(r["n_episodes"]), f"{r['success_rate']:.2%}"
        )
    console.print(table)
    if body.get("leaderboard"):
        console.print("\n[bold]Leaderboard[/bold]")
        for entry in body["leaderboard"]:
            console.print(
                f"  #{entry['rank']:<2} {entry['pipeline']:<28} "
                f"clean={entry['clean_success_rate']:.2%}  "
                f"mean Δ={entry['mean_degradation']:+.2%}"
            )


@app.command("list")
def list_cmd(
    kind: Annotated[
        str | None, typer.Option(help="Filter to one kind (speech, perturbation, ...).")
    ] = None,
) -> None:
    """Print all registered plugins by kind."""
    registries = {
        "speech": registry.speech,
        "grounding": registry.grounding,
        "perturbation": registry.perturbation,
        "policy": registry.policy,
        "env": registry.env,
        "metric": registry.metric,
    }
    if kind is not None:
        if kind not in registries:
            typer.echo(f"unknown kind: {kind}; choices: {', '.join(sorted(registries))}")
            raise typer.Exit(code=2)
        registries = {kind: registries[kind]}
    for name, reg in registries.items():
        console.print(f"[bold]{name}[/bold] ({len(reg)})")
        for n in reg.names():
            console.print(f"  - {n}")


@app.command("validate")
def validate_cmd(config: Annotated[Path, typer.Argument(help="YAML config to validate.")]) -> None:
    """Parse a benchmark config and report errors."""
    cfg = load_config(config)
    n_pipelines = len(cfg.pipelines)
    n_perturbs = len(cfg.perturbations) + 1  # plus clean
    console.print(
        f"[green]ok[/green] {cfg.name}: {n_pipelines} pipelines, "
        f"{n_perturbs} perturbation groups, {cfg.dataset.episodes} episodes"
    )


def main() -> None:  # pragma: no cover - thin wrapper
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
