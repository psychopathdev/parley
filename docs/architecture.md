# Parley architecture

Parley is a layered toolkit. Each layer talks to its neighbours through
small, frozen `Protocol`s defined in `parley.core.types`. Internals are
free to change; the protocols are what stays stable.

```
                                ┌──────────────────────────┐
                                │     parley.cli (typer)   │
                                └─────────────┬────────────┘
                                              │
                                              ▼
        ┌─────────────────────────────────────────────────────────────┐
        │                       parley.runner                         │
        │   suite.expand_suite → engine.BenchmarkEngine.run()         │
        │   pipeline.run_episode (perturb → speech → ground → act)    │
        └─────────────┬───────────────────────────────────────────────┘
                      │
   ┌──────────────────┼──────────────────┬──────────────────┐
   ▼                  ▼                  ▼                  ▼
┌────────┐      ┌───────────┐      ┌──────────┐      ┌────────────┐
│ speech │      │ grounding │      │  policy  │      │    env     │
│        │      │           │      │          │      │            │
│ codec  │      │ rule_based│      │ scripted │      │  tabletop  │
│ mock   │      │           │      │ random   │      │            │
│whisper*│      │           │      │ noisy    │      │            │
└────────┘      └───────────┘      └──────────┘      └────────────┘
                                                   * = optional extra

                  ┌─────────────────────────────────┐
                  │       parley.perturb            │
                  │  audio / linguistic / channel   │
                  │  + Compose + suites helpers     │
                  └─────────────────────────────────┘

                  ┌─────────────────────────────────┐
                  │       parley.metrics            │
                  │  WER CER F1 SuccessRate DTW     │
                  │  latency robustness aggregate   │
                  └─────────────────────────────────┘

                  ┌─────────────────────────────────┐
                  │       parley.report             │
                  │  aggregate → table / leaderbd   │
                  │  sensitivity_index              │
                  │  worst_group_report             │
                  │  versioned dump_report (JSON)   │
                  └─────────────────────────────────┘
```

## Wire format: `parley.core.types`

Every protocol speaks the same dataclasses:

| Type | What | Origin → Sink |
|---|---|---|
| `Audio` | float32 mono PCM + sample rate | Dataset → Perturbation → SpeechFrontend |
| `Instruction` | natural-language text + optional reference | Dataset → Perturbation → SpeechFrontend |
| `Transcript` | recognized text + tokens + confidence | SpeechFrontend → Grounder |
| `Grounding` | verb/target/destination slots | Grounder → Policy (via `Observation`) |
| `Frame` | env state vector + scene + optional image/proprio | Env → Policy |
| `Action` | continuous vec + space tag + discrete label | Policy → Env |
| `Trace` | everything a single episode produced | Engine → Metric → Report |
| `EpisodeResult` | one row of the headline table | Engine → Report |

All types are frozen dataclasses with `numpy.ndarray` for the heavy
fields. They are deliberately small — adding a field forces a thoughtful
choice rather than letting downstream code rely on something that wasn't
meant to be part of the contract.

## Plugin registry

`parley.core.registry.registry` holds seven typed registries: `speech`,
`grounding`, `perturbation`, `policy`, `env`, `metric`, `task`. Plugins
register themselves with a decorator:

```python
@registry.speech.register("my_frontend")
class MyFrontend:
    name = "my_frontend"
    def transcribe(self, audio: Audio, *, reference=None) -> Transcript: ...
```

Configs reference them by string (`speech: { name: my_frontend }`), so a
new backend ships as a third-party `pip install` + an import without
touching the core toolkit.

## Determinism + reproducibility

Two layers of seeding:

1. `RngManager(seed=…)` is constructed per-run from the config's `seed`.
2. Each consumer asks for a *named* sub-stream:
   `rng_mgr.stream("perturb.ep-0001")`. Sub-streams are derived from the
   parent seed via BLAKE2b over the name, so the same `(seed, name)`
   yields the same generator across processes, Python versions, and
   import orders.

Concretely: re-running the same suite with the same `seed` produces the
same audio waveforms after perturbation, the same transcripts, the same
action sequences, and the same metrics. The on-disk
`config.resolved.yaml` snapshot next to each report makes that
reproducibility checkable.

## Caching

`ContentCache` is a file-backed `(key → JSON)` store keyed by
`blake2b(pipeline||perturbation||episode_id||seed)`. The engine writes
`EpisodeResult` dicts in; a second `parley run` of the same suite skips
the expensive parts (audio perturbation, ASR, env rollout) and returns
the cached result. `cache_dir: null` in the YAML disables it; that's the
default for CI where every invocation is fresh.

## Parallelism

`RunnerConfig.workers > 1` switches the engine to a
`ThreadPoolExecutor`. The synthetic env + codec ASR are pure-numpy so
they get only modest speedups under threading (the GIL still bites), but
the surface is in place for adapters that drop the GIL (most C / CUDA
extensions). A `multiprocessing` worker pool would be a drop-in
replacement.

## Where the abstractions came from

The structure is informed by `lm-evaluation-harness`,
[HELM](https://github.com/stanford-crfm/helm), and
[Inspect](https://github.com/UKGovernmentBEIS/inspect_ai). Specifically:

- **Registry + name-strings in configs** mirrors `lm-eval`'s task
  registry — it lets a third-party plug in without core edits.
- **Versioned, JSON-dumpable result schema** comes from HELM's per-run
  artifact, which the project explicitly does not break across versions.
- **Per-instance trace persistence** plus an aggregator on top is the
  pattern Inspect uses; it lets reports be re-rendered without
  re-running.

For the rationale behind the *robustness*-flavored metrics specifically,
see [`design-notes.md`](design-notes.md).
