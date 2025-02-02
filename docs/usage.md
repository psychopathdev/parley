# Usage

## Install

```bash
pip install parley-bench           # core deps only
pip install 'parley-bench[dev]'    # + pytest / ruff / mypy
pip install 'parley-bench[whisper]'  # + openai-whisper for the optional adapter
```

Python 3.10+. The default install pulls only numpy, pydantic, pyyaml,
typer, rich — no heavy ML deps.

## The five subcommands

```bash
parley synth --out path.jsonl --episodes 32   # write a synthetic dataset
parley run config.yaml                        # run a benchmark suite
parley report runs/<name>/ --format markdown  # re-render
parley list --kind metric                     # enumerate plugins
parley validate config.yaml                   # parse-check a config
```

`parley --help` shows the full surface; `parley <cmd> --help` for any
subcommand.

## A minimal config

```yaml
name: my-suite
seed: 0
dataset:
  source: synth          # or 'file' with `path:` set
  episodes: 24
env:
  name: tabletop
pipelines:
  - name: codec+scripted
    speech:    { name: codec }       # see `parley list --kind speech`
    grounding: { name: rule_based }
    policy:    { name: scripted }
perturbations:
  - name: noise5
    steps:
      - { name: additive_noise, params: { snr_db: 5 } }
metrics: [wer, grounding_f1, success_rate, latency]
runner:
  max_steps: 32
  workers: 1
output_dir: runs/my-suite
```

The full example set lives under [`examples/`](../examples/) — start
with [`quickstart.yaml`](../examples/configs/quickstart.yaml).

## What `parley run` writes

Given `output_dir: runs/my-suite`:

```
runs/my-suite/
├── report.json              # versioned JSON, the canonical machine output
├── config.resolved.yaml     # snapshot of the parsed config for reproducibility
└── traces/                  # one JSON per (pipeline, perturbation, episode)
    ├── codec+scripted--clean--ep-00000.json
    ├── codec+scripted--noise5--ep-00000.json
    └── …
```

`report.json` carries the full set of per-cell summaries plus a
leaderboard. Re-rendering happens without re-running:

```bash
parley report runs/my-suite/                     # markdown table + leaderboard
parley report runs/my-suite/ --format csv > x.csv
parley report runs/my-suite/ --format json | jq  # raw schema
```

## Programmatic usage

Everything the CLI does is a thin wrapper. See
[`examples/programmatic/custom_suite.py`](../examples/programmatic/custom_suite.py)
for an end-to-end example that builds the config in code, uses the
`snr_sweep()` / `codec_sweep()` helpers, runs the engine, and prints the
sensitivity index + worst-group report on top of the standard table.

## Plugging in a real frontend or policy

1. Implement the relevant `Protocol`
   (`parley.speech.base.SpeechFrontend`, `parley.policy.base.VLAPolicy`,
   etc.). It just has to expose the right attributes — Parley uses
   `runtime_checkable` Protocols.
2. Register it: `@registry.speech.register("my_thing")` at import time.
3. Reference it from the YAML by name. Constructor kwargs go under
   `params:`.

That's it — no entry-point boilerplate, no factory classes. See
`parley/speech/whisper_adapter.py` for the smallest interesting example
(it's an import-lazy wrapper around `openai-whisper`).

## Reproducing a run

The `seed` field on the config drives every stochastic component
(perturbation noise, env spawn jitter, random policy actions). Two runs
of the same YAML with the same seed produce byte-identical
`report.json`. CI runs with caching disabled to keep this property
honest; once you have a `report.json` you trust, the
`config.resolved.yaml` next to it is enough to reproduce.
