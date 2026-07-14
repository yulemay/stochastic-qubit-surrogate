"""Tests for the recurrent stochastic-qubit response surrogate."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from qubit_surrogate.ml_model import (
    build_response_model,
    compile_response_model,
    create_compiled_model,
)


def test_model_input_output_shapes() -> None:
    """The surrogate must map (10, 6) pulse sequences to 18 responses."""

    model = build_response_model()

    assert model.input_shape == (None, 10, 6)
    assert model.output_shape == (None, 18)


def test_model_uses_three_gru_layers() -> None:
    """The response surrogate must preserve the stacked GRU architecture."""

    model = build_response_model()

    gru_layers = [
        layer
        for layer in model.layers
        if isinstance(layer, tf.keras.layers.GRU)
    ]

    assert len(gru_layers) == 3
    assert gru_layers[0].units == 60
    assert gru_layers[1].units == 60
    assert gru_layers[2].units == 18


def test_hidden_grus_return_sequences() -> None:
    """The hidden recurrent layers must retain temporal sequences."""

    model = build_response_model()

    gru_layers = [
        layer
        for layer in model.layers
        if isinstance(layer, tf.keras.layers.GRU)
    ]

    assert gru_layers[0].return_sequences is True
    assert gru_layers[1].return_sequences is True
    assert gru_layers[2].return_sequences is False


def test_response_layer_uses_tanh_activation() -> None:
    """The final recurrent layer must constrain Pauli predictions."""

    model = build_response_model()
    response_layer = model.get_layer("response_gru")

    assert response_layer.activation == tf.keras.activations.tanh


def test_forward_pass_shape_and_bounds() -> None:
    """A forward pass must return bounded 18-component responses."""

    model = build_response_model()

    pulse_sequences = np.random.default_rng(7).uniform(
        low=0.0,
        high=1.0,
        size=(4, 10, 6),
    ).astype(np.float32)

    predictions = model(pulse_sequences, training=False).numpy()

    assert predictions.shape == (4, 18)
    assert np.all(np.isfinite(predictions))
    assert np.all(predictions >= -1.0)
    assert np.all(predictions <= 1.0)


def test_model_compilation() -> None:
    """The compiled surrogate must use Adam and MSE training."""

    model = build_response_model()
    compiled_model = compile_response_model(model)

    assert isinstance(
        compiled_model.optimizer,
        tf.keras.optimizers.Adam,
    )

    assert np.isclose(
        float(compiled_model.optimizer.learning_rate.numpy()),
        0.012,
    )

    assert compiled_model.loss == "mse"


def test_create_compiled_model() -> None:
    """The convenience constructor must return a compiled model."""

    model = create_compiled_model()

    assert model.input_shape == (None, 10, 6)
    assert model.output_shape == (None, 18)
    assert model.optimizer is not None
