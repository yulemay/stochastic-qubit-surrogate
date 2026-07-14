"""Train the stochastic single-qubit GRU response surrogate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from qubit_surrogate.ml_model import create_compiled_model


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    """Load the JSON training configuration."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def split_indices(
    n_samples: int,
    train_fraction: float,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create reproducible train, validation and test indices."""

    if n_samples < 3:
        raise ValueError("At least three samples are required.")

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie between 0 and 1.")

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie between 0 and 1.")

    if train_fraction + validation_fraction >= 1.0:
        raise ValueError(
            "train_fraction + validation_fraction must be less than 1."
        )

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)

    n_train = int(train_fraction * n_samples)
    n_validation = int(validation_fraction * n_samples)

    train_idx = indices[:n_train]
    validation_idx = indices[n_train : n_train + n_validation]
    test_idx = indices[n_train + n_validation :]

    return train_idx, validation_idx, test_idx


def main() -> None:
    """Train and save the recurrent response surrogate."""

    args = parse_args()
    config = load_config(args.config)

    np.random.seed(config["seed"])
    tf.random.set_seed(config["seed"])

    dataset = np.load(args.dataset)

    data_input = np.asarray(dataset["data_input"], dtype=np.float32)
    data_target = np.asarray(dataset["data_target"], dtype=np.float32)

    if data_input.ndim != 3 or data_input.shape[1:] != (10, 6):
        raise ValueError(
            f"Expected data_input shape (N, 10, 6); received {data_input.shape}."
        )

    if data_target.ndim != 2 or data_target.shape[1] != 18:
        raise ValueError(
            f"Expected data_target shape (N, 18); received {data_target.shape}."
        )

    if data_input.shape[0] != data_target.shape[0]:
        raise ValueError("Input and target sample counts do not match.")

    train_idx, validation_idx, test_idx = split_indices(
        n_samples=len(data_input),
        train_fraction=config["train_fraction"],
        validation_fraction=config["validation_fraction"],
        seed=config["seed"],
    )

    model = create_compiled_model(
        learning_rate=config["learning_rate"],
    )

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "best_model.keras"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config["early_stopping_patience"],
            restore_best_weights=True,
        ),
    ]

    history = model.fit(
        data_input[train_idx],
        data_target[train_idx],
        validation_data=(
            data_input[validation_idx],
            data_target[validation_idx],
        ),
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        callbacks=callbacks,
        verbose=2,
    )

    np.savez_compressed(
        output_dir / "training_history.npz",
        **{
            key: np.asarray(value)
            for key, value in history.history.items()
        },
    )

    np.savez_compressed(
        output_dir / "split_indices.npz",
        train_indices=train_idx,
        validation_indices=validation_idx,
        test_indices=test_idx,
    )

    run_config = {
        **config,
        "dataset": str(Path(args.dataset)),
        "input_shape": list(data_input.shape),
        "target_shape": list(data_target.shape),
        "train_samples": int(len(train_idx)),
        "validation_samples": int(len(validation_idx)),
        "test_samples": int(len(test_idx)),
    }

    with (output_dir / "run_config.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(run_config, handle, indent=2)

    print(f"Saved model to {model_path}")
    print(f"Training samples: {len(train_idx)}")
    print(f"Validation samples: {len(validation_idx)}")
    print(f"Test samples: {len(test_idx)}")


if __name__ == "__main__":
    main()
