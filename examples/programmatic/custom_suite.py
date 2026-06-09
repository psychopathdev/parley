"""End-to-end programmatic example: build a custom suite, run it, print analysis.

Mirrors what ``parley run`` does under the hood, but in code so users can
see the moving parts and adapt them. Run with::

    python examples/programmatic/custom_suite.py
"""

from __future__ import annotations

from pathlib import Path

from parley.core.config import (
    BenchmarkConfig,
    DatasetConfig,
    EnvConfig,
    PipelineConfig,
    PluginSpec,
    RunnerConfig,
)
from parley.data import SynthConfig, generate_dataset
from parley.perturb.suites import codec_sweep, snr_sweep
from parley.report import (
    aggregate_results,
    render_markdown,
    sensitivity_index,
    worst_group_report,
)
from parley.runner import BenchmarkEngine


def main() -> None:
    # 1. Build a small dataset.
    episodes = generate_dataset(SynthConfig(n_episodes=20, seed=42))

    # 2. Build a config programmatically using the sweep helpers.
    cfg = BenchmarkConfig(
        name="programmatic-demo",
        seed=42,
        dataset=DatasetConfig(source="synth", episodes=len(episodes)),
        env=EnvConfig(name="tabletop"),
        pipelines=[
            PipelineConfig(
                name="codec+scripted",
                speech=PluginSpec(name="codec"),
                policy=PluginSpec(name="scripted"),
            ),
            PipelineConfig(
                name="codec+random",
                speech=PluginSpec(name="codec"),
                policy=PluginSpec(name="random"),
            ),
        ],
        perturbations=[*snr_sweep(snr_dbs=(10.0, 0.0, -10.0)), *codec_sweep()],
        metrics=["wer", "grounding_f1", "success_rate", "latency"],
        runner=RunnerConfig(max_steps=30, workers=1, cache_dir=None),
        output_dir=str(Path("runs/programmatic-demo")),
    )

    # 3. Run.
    results = BenchmarkEngine(cfg).run(episodes)
    rows = aggregate_results(results, seed=cfg.seed)

    # 4. Headline table.
    print("\n=== Per-(pipeline, perturbation) summary ===\n")
    print(render_markdown(rows))

    # 5. Sensitivity index (ΔTask / ΔWER).
    print("\n=== Sensitivity index (success_rate vs WER) ===\n")
    print(f"{'pipeline':<20s} {'pert':<22s} {'ΔWER':>8s} {'ΔTask':>8s} {'ratio':>10s}")
    for s in sensitivity_index(rows):
        ratio = "inf" if s.ratio == float("inf") else f"{s.ratio:+.3f}"
        print(
            f"{s.pipeline:<20s} {s.perturbation:<22s} "
            f"{s.delta_input:+8.3f} {s.delta_task:+8.3f} {ratio:>10s}"
        )

    # 6. Worst-group success.
    print("\n=== Worst-group success rate ===\n")
    for w in worst_group_report(rows, metric="success_rate"):
        print(f"  {w.pipeline:<20s} worst on {w.worst_group!r}: {w.worst_value:.2%}")


if __name__ == "__main__":
    main()
