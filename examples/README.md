# Examples

Runnable examples for `parley`. The YAML configs under `configs/` work
with `parley run`; `programmatic/` shows the equivalent in Python.

| Example | Type | What it demonstrates |
|---|---|---|
| [`configs/quickstart.yaml`](configs/quickstart.yaml) | CLI | One pipeline, mild noise — the smallest interesting run. |
| [`configs/robustness_panel.yaml`](configs/robustness_panel.yaml) | CLI | Wide perturbation panel comparing scripted vs random. |
| [`configs/snr_sweep.yaml`](configs/snr_sweep.yaml) | CLI | Five-rung SNR ladder, the canonical degradation curve. |
| [`programmatic/custom_suite.py`](programmatic/custom_suite.py) | Python | Build a config in code, run, print sensitivity + worst-group. |

## Reproducing

The CLI examples all generate their dataset on the fly from a fixed
seed, so two invocations of the same config produce identical numbers.

```bash
parley run examples/configs/quickstart.yaml
parley run examples/configs/robustness_panel.yaml
parley run examples/configs/snr_sweep.yaml
python examples/programmatic/custom_suite.py
```

Outputs land in `runs/<name>/` next to the working directory and include
per-episode traces under `runs/<name>/traces/`.
