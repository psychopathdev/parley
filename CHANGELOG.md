# Changelog

All notable changes to Parley are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] — 2026-06-09

Initial public release.

### Added
- Core contracts: `parley.core.types`, `parley.core.errors`,
  `parley.core.rng`, `parley.core.registry`, `parley.core.config`.
- Speech frontends: `MockSpeechFrontend`, `CodecSpeechFrontend`, and
  an optional `WhisperSpeechFrontend` behind the `whisper` extra.
- Self-contained spectral codec (`parley.speech._codec`) that
  round-trips text ↔ audio so WER is measurable in CI.
- Perturbations:
  - Audio — `AdditiveNoise`, `Gain`, `Clip`, `MuLawCodec`, `Reverb`,
    `TimeStretch`, `PitchShift`.
  - Channel — `BandLimit`, `PacketLoss`, `SpectralDecimate`.
  - Linguistic — `Disfluency`, `FillerInsertion`, `AccentSubstitution`.
  - `Compose` adapter and `snr_sweep`/`codec_sweep`/`linguistic_sweep`
    helpers.
- Grounding: `RuleBasedGrounder` over the tabletop grammar.
- Env + policies: 2-D `TabletopEnv`, `ScriptedPolicy`, `RandomPolicy`,
  `NoisyPolicy`.
- Synthetic dataset (`parley.data.synth`) with jsonl + npz round-trip.
- Metrics — `WER`, `CER`, `KeywordRecall`, `GroundingExactMatch`,
  `GroundingSlotF1`, `SuccessRate`, `ActionMSE`, `DTWDistance`,
  `LatencyPercentiles`, `RobustnessDelta`, plus `summarize` /
  `bootstrap_ci` / `paired_bootstrap_pvalue` aggregation primitives.
- Runner — `BenchmarkEngine`, `Pipeline`, `ContentCache`,
  `expand_suite`, `run_episode`. Optional `ThreadPoolExecutor` workers.
- Report — `aggregate_results`, `render_markdown` / `render_csv`,
  `build_leaderboard`, `sensitivity_index`, `worst_group_report`,
  versioned JSON `dump_report` / `load_report`.
- CLI — `parley {synth, run, report, list, validate}` via Typer.
- CI matrix across Python 3.10–3.13 on Ubuntu and macOS, plus a
  `smoke-cli` end-to-end job; CodeQL weekly; tag-triggered PyPI release
  via trusted publishing.
