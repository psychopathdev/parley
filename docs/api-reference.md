# API reference

This is the curated public surface. Anything not listed here is internal
and may move between releases. Run `parley list` to see registered
plugin names.

## `parley.core.types` — wire format

Frozen dataclasses passed between subsystems. See
[`architecture.md`](architecture.md) for the data-flow diagram.

- `Audio(samples: ndarray, sample_rate: int)` — mono float32 PCM.
- `Transcript(text, tokens, confidence, metadata)`
- `Instruction(text, reference, language, metadata)`
- `Grounding(verb, target, modifier, destination, slots, confidence)`
- `Frame(state, scene, image, proprio, step)`
- `Action(vec, space: "xy_delta"|"xy_pick_place"|"discrete", label)`
- `Observation(frame, transcript, grounding)`
- `Trace`, `StepRecord`, `EpisodeResult` — produced by the engine.

## `parley.core.config` — pydantic v2 config

- `BenchmarkConfig` — top-level. Fields: `name, seed, dataset, env,
  pipelines, perturbations, metrics, runner, output_dir`.
- `PipelineConfig`, `PluginSpec(name, params)`, `PerturbationGroup(name,
  steps)`, `EnvConfig`, `RunnerConfig`, `DatasetConfig`.
- `load_config(path)`, `dump_config(cfg)`.

All extra YAML fields are rejected (`extra='forbid'`).

## `parley.core.registry`

- `registry.speech`, `.grounding`, `.perturbation`, `.policy`, `.env`,
  `.metric`, `.task` — `Registry[T]` instances.
- `Registry.register(name, replace=False)` — decorator form.
- `Registry.register_value(name, value, replace=False)` — programmatic.
- `Registry.get(name) -> T`, `.names() -> list[str]`.

## `parley.core.rng`

- `derive_seed(parent_seed, name) -> int` — BLAKE2b-derived, stable.
- `RngManager(seed)` — `.stream(name)`, `.fresh(name)`.

## Plugins (the registered names)

Run `parley list` for the canonical list. As of v0.1:

| Kind | Name | Class |
|---|---|---|
| speech | `mock` | `MockSpeechFrontend` |
| speech | `codec` | `CodecSpeechFrontend` |
| speech | `whisper` (optional) | `WhisperSpeechFrontend` |
| grounding | `rule_based` | `RuleBasedGrounder` |
| perturbation (audio) | `additive_noise`, `gain`, `clip`, `mu_law`, `reverb`, `time_stretch`, `pitch_shift` | see `parley.perturb.audio` |
| perturbation (channel) | `band_limit`, `packet_loss`, `spectral_decimate` | see `parley.perturb.channel` |
| perturbation (linguistic) | `disfluency`, `filler`, `accent_subst` | see `parley.perturb.linguistic` |
| policy | `scripted`, `random`, `noisy` | see `parley.policy` |
| env | `tabletop` | `TabletopEnv` |
| metric | `wer`, `cer`, `keyword_recall`, `grounding_exact_match`, `grounding_f1`, `success_rate`, `action_mse`, `dtw`, `latency` | see `parley.metrics` |

## `parley.data`

- `SynthConfig(n_episodes, sample_rate, seed, objects_per_scene,
  include_directions)`
- `generate_dataset(cfg) -> list[Episode]`
- `vocab_for(cfg) -> tuple[str, ...]` — closed lexicon shared by encoder
  and codec ASR. Pass through when constructing a `CodecSpeechFrontend`
  manually.
- `save_episodes(episodes, path)` / `load_episodes(path) -> list[Episode]`
  — paired jsonl + npz format.

## `parley.runner`

- `BenchmarkEngine(cfg, engine_cfg=None)` — `.run(episodes) ->
  list[EpisodeResult]`.
- `build_pipeline(name, speech, grounding, policy, extra_kwargs=None)
  -> Pipeline`
- `run_episode(pipeline, env, episode, perturbation, rng_mgr, max_steps,
  perturbation_name)` — single-episode runner used by the engine but
  callable directly for testing.
- `expand_suite(cfg, episodes)` — cartesian product as `list[RunSpec]`.
- `ContentCache(cache_dir)` — `.get / .put / .clear / .enabled`.

## `parley.report`

- `aggregate_results(results, bootstraps=1000, seed=0, confidence=0.95)
  -> list[ReportRow]`
- `render_markdown(rows, columns=DEFAULT_COLUMNS) -> str`
- `render_csv(rows, columns=DEFAULT_COLUMNS) -> str`
- `dump_report(rows, path, suite_name=None) -> Path`
- `load_report(path) -> dict` — validates schema version.
- `build_leaderboard(rows) -> list[LeaderboardEntry]`
- `sensitivity_index(rows, input_metric="wer", task_metric="success_rate",
  baseline="clean") -> list[SensitivityRow]`
- `worst_group_report(rows, metric="success_rate",
  group_by="perturbation") -> list[WorstGroupResult]`

## `parley.metrics`

- `WER`, `CER`, `KeywordRecall` — ASR.
- `GroundingExactMatch`, `GroundingSlotF1` — intent.
- `SuccessRate`, `ActionMSE`, `DTWDistance` — action.
- `LatencyPercentiles` — efficiency.
- `RobustnessDelta` — post-hoc aggregator over per-perturbation means.
- `summarize(values, confidence=0.95, bootstraps=1000, seed=0) -> Summary`
- `bootstrap_ci(values, confidence=0.95, bootstraps=1000, seed=0)`
- `paired_bootstrap_pvalue(a, b, bootstraps=1000, seed=0)`

## `parley.perturb`

- Audio: `Gain`, `Clip`, `AdditiveNoise`, `MuLawCodec`, `Reverb`,
  `TimeStretch`, `PitchShift`.
- Channel: `BandLimit`, `PacketLoss`, `SpectralDecimate`.
- Linguistic: `Disfluency`, `FillerInsertion`, `AccentSubstitution`.
- `Compose(steps, name)` — sequential application.
- `parley.perturb.suites.snr_sweep(...)`, `codec_sweep()`,
  `linguistic_sweep()` — programmatic group builders.
