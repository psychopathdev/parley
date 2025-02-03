# Design notes

This document explains *why* Parley looks the way it does, by walking
through the systems it positions itself against. It is grounded in a
literature survey (VLA policies, speech LLMs, robot benchmarks, ASR
robustness, evaluation toolkit architecture). Every link below is a real
system the design was checked against.

## Why a separate toolkit?

The closest things to "spoken-instruction VLA benchmarks" today don't
exist as a single piece of software. The mainstream robot benchmarks
condition policies on *ground-truth text strings* — none of them
natively expose a spoken-instruction channel or audio perturbations:

- [LIBERO](https://libero-project.github.io/) — Spatial/Object/Goal/Long
  splits via BDDL predicates. Now the de facto VLA leaderboard.
- [CALVIN](http://calvin.cs.uni-freiburg.de/) — five-subtask chains
  across env splits A/B/C/D.
- [RLBench](https://github.com/stepjam/RLBench),
  [Meta-World](https://github.com/Farama-Foundation/Metaworld),
  [ManiSkill2/3](https://github.com/haosulab/ManiSkill),
  [robomimic](https://robomimic.github.io/),
  [SimplerEnv](https://simpler-env.github.io/),
  [VLABench](https://arxiv.org/abs/2412.18194),
  [VIMA-Bench](https://vimalabs.github.io/).

Meanwhile the speech-side benchmarks
([AIR-Bench](https://arxiv.org/abs/2402.07729),
[Dynamic-SUPERB](https://dynamic-superb.github.io/),
[SLURP](https://github.com/pswietojanski/slurp)) measure transcript /
intent quality but never reach for an end-to-end robot success metric.

Parley is the missing piece: an evaluation harness that runs the
**full chain** — `audio → ASR → grounding → VLA policy → env → success`
— and *attributes* task-success degradation to speech-side
perturbations. The toolkit is small and synthetic on purpose: the
self-contained env + codec ASR keep CI honest while the protocol layer
lets adapters for real systems (Whisper / Qwen-Audio / OpenVLA / Octo /
π0 / LIBERO / ManiSkill) plug in cleanly.

## Policy interface — what real VLAs look like

The 2024–2026 VLA landscape converges on a pretrained VLM coupled to a
"action expert" that emits continuous action chunks via flow-matching or
diffusion:

| System | Action representation | Reference |
|---|---|---|
| RT-1, RT-2 | 256-bin discrete tokens | [RT-1](https://arxiv.org/abs/2212.06817), [RT-2](https://arxiv.org/abs/2307.15818) |
| Octo | T5 + diffusion head | [octo-models/octo](https://github.com/octo-models/octo) |
| OpenVLA | Llama-2 + DINOv2/SigLIP, autoregressive | [openvla/openvla](https://github.com/openvla/openvla) |
| OpenVLA-OFT | parallel decoding, L1 regression, ~26× throughput | [openvla-oft](https://openvla-oft.github.io/) |
| π₀ / π₀-FAST / π₀.5 | PaliGemma + 300M expert, 50-step chunks, flow-matching | [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) |
| GR00T N1 | Eagle-2 System 2 + DiT System 1, K=16 | [arXiv:2503.14734](https://arxiv.org/abs/2503.14734) |
| SmolVLA | ~450M, asynchronous predict/execute | [HF blog](https://huggingface.co/blog/smolvla) |
| RDT-1B | diffusion transformer | [RoboticsDiffusionTransformer](https://github.com/thu-ml/RoboticsDiffusionTransformer) |

The axes that vary across systems are **chunk length K**, **action
representation** (discrete / FAST / continuous / denoised), **camera
count**, and **proprioception**. Parley's `VLAPolicy` Protocol is shaped
to abstract over all of them: a policy sees an `Observation` (frame +
transcript + grounding), returns an `Action` with an explicit
`action_space` tag, and the env decodes it. Chunking is implemented by
the policy itself caching its output across `act()` calls — the engine
doesn't impose a single-step assumption.

## Speech frontend interface — what real frontends look like

Three tiers of frontend exist today; Parley's `SpeechFrontend` Protocol
covers all of them with the same `(Audio) → Transcript` shape:

1. **Pure ASR**:
   [Whisper large-v3](https://huggingface.co/openai/whisper-large-v3) /
   [Whisper turbo](https://huggingface.co/openai/whisper-large-v3-turbo),
   [Distil-Whisper](https://github.com/huggingface/distil-whisper),
   [SeamlessM4T v2](https://github.com/facebookresearch/seamless_communication).
2. **Audio-LLMs that follow spoken instructions**:
   [Qwen2-Audio](https://arxiv.org/abs/2407.10759),
   [SALMONN](https://arxiv.org/abs/2310.13289),
   [SpeechGPT](https://arxiv.org/abs/2305.11000).
3. **Full-duplex S2S**:
   [Moshi](https://github.com/kyutai-labs/moshi) (Mimi 12.5 Hz codec,
   ~200 ms latency).

Spoken instruction-following is benchmarked on
[AIR-Bench](https://arxiv.org/abs/2402.07729),
[Dynamic-SUPERB Phase-2](https://dynamic-superb.github.io/), and SLU
sets like [SLURP](https://github.com/pswietojanski/slurp).

The bundled `CodecSpeechFrontend` is *not* meant to compete with these
— it's a deterministic round-trippable substitute that makes WER
*measurable in CI without GPUs or model downloads*. The on-disk synth
audio is the codec encoding of the instruction text; the codec frontend
decodes it back. Clean audio round-trips perfectly; perturbed audio
produces realistic word substitutions and insertions. Real Whisper is
provided as an opt-in extra (`pip install 'parley-bench[whisper]'`) for
when measurement-of-Whisper is the point.

## Metrics — the union of what's worth reporting

The literature agrees on roughly these axes; Parley implements one
representative metric per axis (with room to add more):

### ASR / speech stage
- **WER**, **CER** on normalized text. Standard practice computes them
  via [jiwer](https://github.com/jitsi/jiwer) with the
  [Whisper normalizer](https://github.com/openai/whisper/tree/main/whisper/normalizers);
  Parley uses a direct DP edit-distance to keep zero deps but exposes
  sub/ins/del counts in the same shape.
- **Keyword recall** for "content words" (verbs, colors, shapes,
  directions) — closer to what downstream grounding cares about than
  raw WER.

### Grounding / intent stage
- **Slot F1** + **exact match** over `(verb, target, destination)`.
  Mirrors SLURP-style SLU scoring.

### Action / task stage
- **Success rate** (LIBERO/CALVIN-style binary predicate, averaged).
- **ActionMSE**, **DTW distance** against an oracle reference action
  sequence when one exists. Useful for offline comparisons à la
  robomimic.

### Efficiency
- **Latency p50/p95/p99**, **RTF**. The total ÷ audio-duration ratio
  matches the convention in ASR papers and toolkits.

### Robustness (the headline)
- **Δ vs clean** per perturbation, mean and max degradation across the
  perturbation panel.
- **Sensitivity index** ΔTask/ΔWER — interpretable as: a 1-point
  increase in WER costs N points of task success.
- **Worst-group** report — the lowest cell of a target metric across the
  grouping axis (currently perturbation; the shape supports future
  accent / L1 strata).

### Statistical bookkeeping
- **Percentile bootstrap CI** for each cell (lm-eval-harness style).
- **Paired-bootstrap p-value** for pipeline-vs-pipeline significance.

## Perturbations worth implementing

Drawn from CHiME / MUSAN / DEMAND noise-robustness practice plus
spoken-language-variation literature. Parley ships:

### Audio
| Perturbation | Why it matters |
|---|---|
| `additive_noise` (SNR-calibrated) | canonical robustness ladder; CHiME-style |
| `gain`, `clip` | cheap mic / quiet-talker proxies |
| `mu_law` | G.711 telephony codec round-trip |
| `band_limit` | ITU-T G.712 narrowband passband |
| `spectral_decimate` | poor-man's lossy-codec degradation |
| `reverb` | smears adjacent codec symbols — the failure mode rooms cause |
| `packet_loss` | VoIP-style dropped chunks |
| `time_stretch`, `pitch_shift` | speaking-rate / pitch variance |

### Linguistic
| Perturbation | Why it matters |
|---|---|
| `disfluency` | self-repetitions ("the the red cube") |
| `filler` | "uhm", "uh" insertions — speech-natural and rarely benched |
| `accent_subst` | configurable lexical-style substitution |

What's deliberately **not** implemented (yet):

- Real RIRs (e.g. [OpenSLR SLR28](https://www.openslr.org/28/)) — would
  add a download dep and aren't necessary for the synthetic env.
- Real noise corpora ([MUSAN](https://www.openslr.org/17/),
  [DEMAND](https://zenodo.org/record/1227121)). Again, an extra.
- Per-speaker / accent strata (e.g. [Common Voice](https://commonvoice.mozilla.org/),
  [L2-ARCTIC](https://psi.engr.tamu.edu/l2-arctic-corpus/),
  [EdAcc](https://groups.inf.ed.ac.uk/edacc/)). The `worst_group_report`
  surface accommodates these once a real dataset is plugged in.

## Architecture patterns adopted

- **Scenario / Adapter / Metric / RunSpec** from
  [HELM](https://github.com/stanford-crfm/helm) — separation between
  task definition, prompting strategy, and scoring.
- **Decorator + name-string registry** following
  [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
  and [Inspect](https://github.com/UKGovernmentBEIS/inspect_ai). Plugins
  register themselves; configs reference them by name.
- **Versioned result schema** — MTEB does this with `.v2` task suffixes;
  Parley does it with `REPORT_SCHEMA_VERSION` on the JSON dump.
- **Per-instance JSONL/JSON persistence** then aggregate. Mirrors HELM's
  helm-server and Inspect's eval-log viewer.
- **Bootstrap percentile CI** following lm-eval's recipe: resample with
  replacement, deterministic per `(metric, seed)`.

## What this design is *not* good at

Worth being honest about:

- The synthetic env is not a physics simulator. It's a controllable
  testbed. Numbers it produces are calibrated against itself, not
  against a real robot.
- The codec ASR is robust to surprisingly low SNRs (~-20 dB before WER
  spikes) because its log-spaced tone vocabulary has high bin SNR even
  under heavy noise. Real ASR breaks earlier; Whisper adapter measures
  that earlier breakage.
- Action chunking, off-policy IL evaluation, and real RIR convolution
  are deliberately out of scope for v0. The protocol surface
  accommodates them; the implementations don't yet exist.
