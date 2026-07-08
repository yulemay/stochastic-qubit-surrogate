from __future__ import annotations

import numpy as np
from scipy.io import savemat

from qubit_surrogate.data import (
    TARGET_NAMES,
    build_dataset,
    denormalise_sequence,
    normalise_pulse_parameters,
)


def make_pulse_parameters(scale: float = 1.0) -> np.ndarray:
    total_time = 0.25e-6
    n_gaussians = 10
    spacing = total_time / (n_gaussians + 1)
    width = 0.15 * spacing
    amplitude_bound = np.pi / (np.sqrt(2*np.pi) * width)

    pulses = np.zeros((2, n_gaussians, 3), dtype=float)
    centres = np.arange(1, n_gaussians + 1) * spacing

    for control in range(2):
        pulses[control, :, 0] = scale * np.linspace(
            -0.8*amplitude_bound,
            0.8*amplitude_bound,
            n_gaussians,
        )
        pulses[control, :, 1] = centres
        pulses[control, :, 2] = width

    return pulses


def test_normalisation_round_trip() -> None:
    pulses = make_pulse_parameters()
    sequence = normalise_pulse_parameters(pulses)
    recovered = denormalise_sequence(sequence)

    assert sequence.shape == (10, 6)
    assert np.allclose(recovered, pulses, rtol=2e-6, atol=1e-12)


def test_build_dataset(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    out_file = tmp_path / "processed" / "dataset.npz"

    for idx, scale in enumerate((0.95, 0.90, 0.85), start=1):
        response = np.linspace(-0.8, 0.8, len(TARGET_NAMES)) + idx*1e-3
        savemat(
            raw_dir / f"dataset_{idx:04d}.mat",
            {
                "pulse_parameters": make_pulse_parameters(scale),
                "data": response,
            },
        )

    X, Y = build_dataset(raw_dir, out_file)

    assert X.shape == (3, 10, 6)
    assert Y.shape == (3, 18)
    assert out_file.exists()
