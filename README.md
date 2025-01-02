# Parley

A benchmark toolkit for spoken-instruction Vision-Language-Action (VLA) pipelines.

Parley measures how a speech frontend (ASR / audio-LLM) feeding a VLA policy degrades
end-to-end under realistic audio and linguistic perturbations: noise, codec, accent,
disfluency. It ships a self-contained synthetic environment and reference policies so
the full pipeline runs in CI without GPUs or model downloads, and exposes pluggable
protocols for real frontends (Whisper, Qwen-Audio) and real policies (OpenVLA, Octo).

> Status: early. APIs may change.

## Install

```bash
pip install -e ".[dev]"
```

## Quick start

```bash
parley synth --out runs/demo/dataset.jsonl --episodes 32
parley run examples/configs/quickstart.yaml
parley report runs/demo --format markdown
```

See `docs/usage.md` and `examples/` for more.
