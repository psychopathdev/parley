"""Helpers that programmatically build perturbation suites.

The runner consumes :class:`PerturbationGroup` objects directly from
YAML, but for ad-hoc sweeps (notebooks, hyperparameter studies) it's
useful to materialize a family of groups in code. These helpers do that.
"""

from __future__ import annotations

from collections.abc import Sequence

from parley.core.config import PerturbationGroup, PluginSpec


def snr_sweep(
    snr_dbs: Sequence[float] = (20.0, 10.0, 5.0, 0.0, -5.0, -10.0, -20.0),
    *,
    prefix: str = "snr",
) -> list[PerturbationGroup]:
    """Build an additive-noise sweep across the supplied SNR ladder.

    The ladder defaults to the points emphasized by CHiME/MUSAN-style
    robustness studies. Names are emitted as ``<prefix>_<+/-><db>db`` so
    reports sort numerically when the SNR ladder is the X axis.
    """

    groups: list[PerturbationGroup] = []
    for db in snr_dbs:
        sign = "p" if db >= 0 else "n"
        name = f"{prefix}_{sign}{abs(round(db))}db"
        groups.append(
            PerturbationGroup(
                name=name,
                steps=[PluginSpec(name="additive_noise", params={"snr_db": float(db)})],
            )
        )
    return groups


def codec_sweep() -> list[PerturbationGroup]:
    """A common channel-degradation panel: mu-law, narrowband, spectral hole."""
    return [
        PerturbationGroup(name="mu_law", steps=[PluginSpec(name="mu_law")]),
        PerturbationGroup(
            name="telephone",
            steps=[PluginSpec(name="band_limit", params={"low_hz": 300.0, "high_hz": 3400.0})],
        ),
        PerturbationGroup(
            name="spectral_decimate_40",
            steps=[PluginSpec(name="spectral_decimate", params={"drop_fraction": 0.4})],
        ),
        PerturbationGroup(
            name="packet_loss_10",
            steps=[PluginSpec(name="packet_loss", params={"loss_rate": 0.1})],
        ),
    ]


def linguistic_sweep() -> list[PerturbationGroup]:
    """Three linguistic axes commonly tested independently in NLU benchmarks."""
    return [
        PerturbationGroup(
            name="disfluency",
            steps=[PluginSpec(name="disfluency", params={"rate": 0.3})],
        ),
        PerturbationGroup(
            name="filler",
            steps=[PluginSpec(name="filler", params={"rate": 0.2})],
        ),
        PerturbationGroup(
            name="accent_subst",
            steps=[PluginSpec(name="accent_subst", params={"rate": 0.5})],
        ),
    ]
