# Changelog

All notable changes to Parley are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- Linguistic perturbations (disfluency / filler / accent) are no longer
  silent: when the codec frontend is in use, the runner re-encodes the
  perturbed instruction text so the change actually reaches the ASR
  stage and moves WER.
- The result cache key now folds in a fingerprint of the resolved
  pipeline + perturbation params, so two suites that share group *names*
  but differ in params (e.g. `additive_noise` snr_db 0 vs -20) no longer
  collide.
- The engine builds a private pipeline per call under `workers > 1`,
  removing a data race on the stateful policy when running multi-threaded.
- `_resample_linear` no longer raises on empty input.

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
