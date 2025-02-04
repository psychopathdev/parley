# Parley

[![ci](https://github.com/mrvellang/parley/actions/workflows/ci.yml/badge.svg)](https://github.com/mrvellang/parley/actions/workflows/ci.yml)
[![codeql](https://github.com/mrvellang/parley/actions/workflows/codeql.yml/badge.svg)](https://github.com/mrvellang/parley/actions/workflows/codeql.yml)
[![python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue)](https://github.com/mrvellang/parley)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![ruff](https://img.shields.io/badge/ruff-checked-261230)](https://docs.astral.sh/ruff/)
[![mypy](https://img.shields.io/badge/mypy-strict-blue)](http://mypy-lang.org/)

A benchmark toolkit for **spoken-instruction Vision-Language-Action**
(VLA) pipelines. Parley measures how a speech frontend feeding a VLA
policy degrades end-to-end under realistic audio and linguistic
perturbations — noise, codec, accent, disfluency — and attributes the
loss to the right stage.

The toolkit ships a self-contained synthetic environment and reference
policies so the **full pipeline runs in CI without GPUs or model
downloads**. Real frontends (Whisper, Qwen-Audio) and real policies
(OpenVLA, Octo, π0) plug in via small `Protocol`s.

> Status: alpha. APIs may change before 1.0.

## Why

The mainstream robot benchmarks (LIBERO, CALVIN, RLBench, ManiSkill,
SimplerEnv, VLABench) condition policies on *ground-truth text* and
don't natively expose audio perturbations. The mainstream speech
benchmarks (AIR-Bench, Dynamic-SUPERB, SLURP) measure transcript or
intent quality but never reach for a robot-task success metric.

Parley is the missing piece: an evaluation harness that runs
`audio → ASR → grounding → VLA policy → env → success` end-to-end and
attributes task-success degradation to speech-side perturbations.

See [`docs/design-notes.md`](docs/design-notes.md) for the literature
survey behind the design.

## Install

```bash
pip install parley-bench            # core deps only
pip install 'parley-bench[dev]'     # + ruff / mypy / pytest
pip install 'parley-bench[whisper]' # + the optional Whisper adapter
```

Python ≥3.10. Core dependencies: numpy, pydantic, pyyaml, typer, rich.

## 60-second tour

```bash
# 1. Generate a synthetic dataset.
parley synth --out runs/demo/dataset.jsonl --episodes 32

# 2. Run a benchmark suite.
parley run examples/configs/robustness_panel.yaml

# 3. Re-render the report.
parley report runs/robustness --format markdown
```

…or programmatically:

```python
from parley.data import SynthConfig, generate_dataset
from parley.perturb.suites import snr_sweep, codec_sweep
from parley.runner import BenchmarkEngine
from parley.report import aggregate_results, sensitivity_index, render_markdown
from parley.core.config import BenchmarkConfig, EnvConfig, PipelineConfig, PluginSpec, RunnerConfig

cfg = BenchmarkConfig(
    name="demo", seed=0,
    env=EnvConfig(name="tabletop"),
    pipelines=[PipelineConfig(name="codec+scripted",
                              speech=PluginSpec(name="codec"),
                              policy=PluginSpec(name="scripted"))],
    perturbations=[*snr_sweep(snr_dbs=(10, 0, -10)), *codec_sweep()],
    metrics=["wer", "grounding_f1", "success_rate", "latency"],
    runner=RunnerConfig(max_steps=40, workers=1),
    output_dir="runs/demo",
)
episodes = generate_dataset(SynthConfig(n_episodes=20, seed=0))
results = BenchmarkEngine(cfg).run(episodes)
rows = aggregate_results(results, seed=cfg.seed)

print(render_markdown(rows))
for s in sensitivity_index(rows):
    print(s)
```

## What you get

- **Speech frontends**: `mock` (perfect-ASR baseline), `codec` (the
  self-contained reference), `whisper` (optional extra). Add your own
  with a 10-line `@registry.speech.register("…")`.
- **Perturbations**: SNR-calibrated noise, gain, clipping, mu-law, μ-law
  band-limit, packet loss, spectral decimation, reverb, time/pitch
  shift, disfluency, fillers, accent substitution — all seedable, all
  composable.
- **Grounding + policies**: rule-based grounder, scripted oracle policy
  (success-rate ceiling), random floor, Gaussian-noise wrapper.
- **Synthetic env**: 2-D tabletop with pick/place/push verbs and a
  predicate-based success function.
- **Metrics**: WER/CER/keyword-recall (with sub/ins/del), slot F1, exact
  match, success rate, action MSE/MAE/DTW, latency p50/p95/p99 + RTF,
  robustness deltas, **sensitivity index** (ΔTask/ΔWER), **worst-group**
  reports. Aggregation with percentile bootstrap CIs + paired
  significance tests.
- **Report**: markdown / CSV / versioned JSON, plus a leaderboard ranked
  by clean success-rate with mean-degradation tie-breaks.
- **Runner**: content-addressed cache, optional thread-pool parallelism,
  deterministic across re-runs (same `seed` → same numbers).
- **CLI**: `parley {synth, run, report, list, validate}`.

## Docs

- [`docs/architecture.md`](docs/architecture.md) — layered design,
  Protocol surfaces, determinism + caching.
- [`docs/usage.md`](docs/usage.md) — install, configs, output layout,
  programmatic API.
- [`docs/metrics.md`](docs/metrics.md) — every metric, scale, when to
  use it.
- [`docs/api-reference.md`](docs/api-reference.md) — curated public
  surface.
- [`docs/design-notes.md`](docs/design-notes.md) — literature survey,
  citations, scope decisions.
- [`examples/README.md`](examples/README.md) — the runnable example index.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). The four CI gates are
`ruff check`, `ruff format --check`, `mypy`, `pytest` — all green
locally before you push.

## License

[MIT](LICENSE).
