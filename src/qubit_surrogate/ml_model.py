"""TensorFlow GRU response surrogate for stochastic single-qubit data."""

from __future__ import annotations

import tensorflow as tf


DEFAULT_SEQUENCE_LENGTH = 10
DEFAULT_FEATURE_COUNT = 6
DEFAULT_RESPONSE_SIZE = 18
DEFAULT_HIDDEN_UNITS = 60
DEFAULT_LEARNING_RATE = 0.012


def build_response_model(
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    feature_count: int = DEFAULT_FEATURE_COUNT,
    response_size: int = DEFAULT_RESPONSE_SIZE,
    hidden_units: int = DEFAULT_HIDDEN_UNITS,
) -> tf.keras.Model:
    """Build the stacked GRU pulse-response surrogate."""

    inputs = tf.keras.Input(
        shape=(sequence_length, feature_count),
        name="pulse_sequence",
    )

    x = tf.keras.layers.GRU(
        hidden_units,
        return_sequences=True,
        name="gru_1",
    )(inputs)

    x = tf.keras.layers.GRU(
        hidden_units,
        return_sequences=True,
        name="gru_2",
    )(x)

    outputs = tf.keras.layers.GRU(
        response_size,
        activation="tanh",
        return_sequences=False,
        name="response_gru",
    )(x)

    return tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="stochastic_qubit_surrogate",
    )


def compile_response_model(
    model: tf.keras.Model,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> tf.keras.Model:
    """Compile the response surrogate for mean-squared-error training."""

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
    )

    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
        ],
    )

    return model


def create_compiled_model(
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    feature_count: int = DEFAULT_FEATURE_COUNT,
    response_size: int = DEFAULT_RESPONSE_SIZE,
    hidden_units: int = DEFAULT_HIDDEN_UNITS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> tf.keras.Model:
    """Build and compile the recurrent response surrogate."""

    model = build_response_model(
        sequence_length=sequence_length,
        feature_count=feature_count,
        response_size=response_size,
        hidden_units=hidden_units,
    )

    return compile_response_model(
        model,
        learning_rate=learning_rate,
    )
