"""MATLAB-to-NumPy preprocessing for stochastic qubit response data."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
from scipy.io import loadmat

DEFAULT_TOTAL_TIME = 0.25e-6
DEFAULT_N_GAUSSIANS = 10

FEATURE_NAMES = np.asarray(
    ["A_x", "tau_x", "sigma_x", "A_y", "tau_y", "sigma_y"]
)

_INITIAL_STATES = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
_OBSERVABLES = ["<X>", "<Y>", "<Z>"]
TARGET_NAMES = np.asarray(
    [f"{state}/{obs}" for state in _INITIAL_STATES for obs in _OBSERVABLES]
)


class DatasetValidationError(ValueError):
    """Raised when a raw simulation file does not match the public schema."""


def _numeric_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)(?=\.mat$)", path.name)
    return (int(match.group(1)) if match else -1, path.name)


def discover_mat_files(mat_folder: str | Path) -> list[Path]:
    """Return MATLAB dataset files in numeric filename order."""

    folder = Path(mat_folder)
    files = sorted(folder.glob("dataset_*.mat"), key=_numeric_sort_key)

    if not files:
        raise FileNotFoundError(
            f"No dataset_*.mat files found in {folder.resolve()}"
        )

    return files


def pulse_bounds(
    total_time: float = DEFAULT_TOTAL_TIME,
    n_gaussians: int = DEFAULT_N_GAUSSIANS,
) -> tuple[float, float, np.ndarray]:
    """Return public Gaussian-width bounds and fixed normalised centres."""

    spacing = total_time / (n_gaussians + 1)
    sigma_min = 0.04 * spacing
    sigma_max = 0.30 * spacing
    centres_norm = np.arange(1, n_gaussians + 1) / (n_gaussians + 1)
    return sigma_min, sigma_max, centres_norm


def validate_raw_example(
    pulse_parameters: np.ndarray,
    response: np.ndarray,
    total_time: float = DEFAULT_TOTAL_TIME,
) -> None:
    """Validate one raw MATLAB simulation example."""

    pulse_parameters = np.asarray(pulse_parameters)
    response = np.asarray(response).reshape(-1)

    if pulse_parameters.ndim != 3:
        raise DatasetValidationError(
            "pulse_parameters must be rank 3: (controls, gaussians, parameters)."
        )

    n_controls, n_gaussians, n_parameters = pulse_parameters.shape

    if (n_controls, n_gaussians, n_parameters) != (2, DEFAULT_N_GAUSSIANS, 3):
        raise DatasetValidationError(
            "Expected pulse_parameters shape (2, 10, 3); "
            f"received {pulse_parameters.shape}."
        )

    if response.size != len(TARGET_NAMES):
        raise DatasetValidationError(
            f"Expected {len(TARGET_NAMES)} response values; received {response.size}."
        )

    if not np.all(np.isfinite(pulse_parameters)):
        raise DatasetValidationError("pulse_parameters contains NaN or inf.")

    if not np.all(np.isfinite(response)):
        raise DatasetValidationError("response contains NaN or inf.")

    if np.any(np.abs(response) > 1.0001):
        raise DatasetValidationError(
            "Pauli expectation outside [-1, 1] numerical tolerance."
        )

    sigma_min, sigma_max, centres_norm = pulse_bounds(total_time, n_gaussians)
    centres = pulse_parameters[:, :, 1] / total_time
    widths = pulse_parameters[:, :, 2]

    if not np.allclose(centres[0], centres_norm, rtol=0.0, atol=1e-6):
        raise DatasetValidationError("x-channel centres do not match the fixed grid.")

    if not np.allclose(centres[1], centres_norm, rtol=0.0, atol=1e-6):
        raise DatasetValidationError("y-channel centres do not match the fixed grid.")

    if not np.allclose(widths, widths[0, 0], rtol=1e-6, atol=0.0):
        raise DatasetValidationError(
            "Expected one Gaussian width shared across all components and channels."
        )

    if not sigma_min <= float(widths[0, 0]) <= sigma_max:
        raise DatasetValidationError("Gaussian width is outside the configured bounds.")


def normalise_pulse_parameters(
    pulse_parameters: np.ndarray,
    total_time: float = DEFAULT_TOTAL_TIME,
) -> np.ndarray:
    """Convert one MATLAB pulse tensor to a GRU-ready sequence.

    Returns an array shaped ``(10, 6)`` with feature order
    ``A_x, tau_x, sigma_x, A_y, tau_y, sigma_y``.
    """

    pulse_parameters = np.asarray(pulse_parameters, dtype=np.float64)
    dummy_response = np.zeros(len(TARGET_NAMES), dtype=np.float64)
    validate_raw_example(pulse_parameters, dummy_response, total_time)

    n_controls, n_gaussians, _ = pulse_parameters.shape
    amplitudes = pulse_parameters[:, :, 0]
    centres = pulse_parameters[:, :, 1]
    widths = pulse_parameters[:, :, 2]

    sigma_min, sigma_max, _ = pulse_bounds(total_time, n_gaussians)

    amplitude_bound = np.pi / (np.sqrt(2.0 * np.pi) * widths)
    amplitude_norm = 0.5 * (amplitudes / amplitude_bound + 1.0)
    centre_norm = centres / total_time
    width_norm = (widths - sigma_min) / (sigma_max - sigma_min)

    channel_features = np.stack(
        [amplitude_norm, centre_norm, width_norm],
        axis=-1,
    )

    sequence = (
        channel_features
        .transpose(1, 0, 2)
        .reshape(n_gaussians, n_controls * 3)
    )

    if np.any(sequence < -1e-6) or np.any(sequence > 1.0 + 1e-6):
        raise DatasetValidationError("Normalised pulse features fall outside [0, 1].")

    return sequence.astype(np.float32)


def denormalise_sequence(
    sequence: np.ndarray,
    total_time: float = DEFAULT_TOTAL_TIME,
) -> np.ndarray:
    """Convert one normalised ``(10, 6)`` sequence to MATLAB pulse parameters."""

    sequence = np.asarray(sequence, dtype=np.float64)

    if sequence.shape != (DEFAULT_N_GAUSSIANS, 6):
        raise DatasetValidationError(
            f"Expected normalised sequence shape (10, 6); received {sequence.shape}."
        )

    if np.any(sequence < -1e-6) or np.any(sequence > 1.0 + 1e-6):
        raise DatasetValidationError("Normalised sequence falls outside [0, 1].")

    if not np.allclose(sequence[:, 1], sequence[:, 4], atol=1e-7):
        raise DatasetValidationError("x/y centre columns must match.")

    if not np.allclose(sequence[:, 2], sequence[:, 5], atol=1e-7):
        raise DatasetValidationError("x/y width columns must match.")

    if not np.allclose(sequence[:, 2], sequence[0, 2], atol=1e-7):
        raise DatasetValidationError("Controller width must be shared across components.")

    sigma_min, sigma_max, centres_norm = pulse_bounds(total_time)

    if not np.allclose(sequence[:, 1], centres_norm, atol=1e-6):
        raise DatasetValidationError("Centre columns do not match the fixed grid.")

    sigma = sigma_min + sequence[0, 2] * (sigma_max - sigma_min)
    amplitude_bound = np.pi / (np.sqrt(2.0 * np.pi) * sigma)

    amplitudes_x = (2.0 * sequence[:, 0] - 1.0) * amplitude_bound
    amplitudes_y = (2.0 * sequence[:, 3] - 1.0) * amplitude_bound
    centres = centres_norm * total_time

    pulse_parameters = np.zeros((2, DEFAULT_N_GAUSSIANS, 3), dtype=np.float64)
    pulse_parameters[0, :, 0] = amplitudes_x
    pulse_parameters[1, :, 0] = amplitudes_y
    pulse_parameters[:, :, 1] = centres
    pulse_parameters[:, :, 2] = sigma

    return pulse_parameters


def load_raw_example(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate one MATLAB simulation file."""

    path = Path(path)
    mat = loadmat(
        path,
        squeeze_me=True,
        variable_names=["pulse_parameters", "data"],
        verify_compressed_data_integrity=True,
    )

    missing = {"pulse_parameters", "data"} - set(mat)
    if missing:
        raise DatasetValidationError(
            f"{path.name} is missing fields: {sorted(missing)}"
        )

    pulse_parameters = np.asarray(mat["pulse_parameters"], dtype=np.float64)
    response = np.asarray(mat["data"], dtype=np.float64).reshape(-1)

    validate_raw_example(pulse_parameters, response)
    return pulse_parameters, response


def build_dataset(
    mat_folder: str | Path,
    out_file: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Build and save the GRU-ready pulse-response dataset."""

    files = discover_mat_files(mat_folder)
    X: list[np.ndarray] = []
    Y: list[np.ndarray] = []

    for path in files:
        pulse_parameters, response = load_raw_example(path)
        X.append(normalise_pulse_parameters(pulse_parameters))
        Y.append(response.astype(np.float32))

    data_input = np.stack(X, axis=0)
    data_target = np.stack(Y, axis=0)

    unique_inputs = np.unique(data_input.reshape(len(data_input), -1), axis=0)
    if unique_inputs.shape[0] != len(data_input):
        raise DatasetValidationError("Duplicate input examples detected.")

    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        data_input=data_input,
        data_target=data_target,
        source_files=np.asarray([p.name for p in files]),
        feature_names=FEATURE_NAMES,
        target_names=TARGET_NAMES,
    )

    return data_input, data_target
