"""Minimal fix for the FordA Transformer (appendix): give attention and LayerNorm real channels.

A Conv1D(d_model, kernel_size=1) input projection plus a learned positional embedding feed the
unchanged `transformer_encoder` blocks, so each LayerNormalization normalizes over d_model
channels instead of one and the attention/feed-forward branches reach the output. Pooling is
channels_last (as on master); every other hyperparameter, the data pipeline, the training
loop and the outputs are those of transformer_official.py.

Usage:
    python experiments/transformer_fixed.py --run-name T3_fixed
    python experiments/transformer_fixed.py --run-name smoke --epochs 1 --max-train-samples 256
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ["KERAS_BACKEND"] = "tensorflow"  # before keras is imported (see transformer_official.py)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
from keras import layers

from transformer_official import base_parser, prepare, train_and_report, transformer_encoder


class PositionalEmbedding(layers.Layer):
    """Learned (sequence_length, d_model) table added to the projected input."""

    def build(self, input_shape):
        self.table = self.add_weight(
            shape=tuple(input_shape[1:]), initializer=keras.initializers.RandomNormal(stddev=0.02), name="table"
        )

    def call(self, x):
        return x + self.table


def build_fixed_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
    n_classes=2,
    d_model=64,
):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv1D(filters=d_model, kernel_size=1)(inputs)
    x = PositionalEmbedding()(x)
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


def main() -> None:
    p = base_parser(__doc__.splitlines()[0], "T3_fixed")
    p.add_argument("--d-model", type=int, default=64, help="channels after the input projection")
    args = p.parse_args()
    data = prepare(args)
    model = build_fixed_model(
        data[0].shape[1:],
        head_size=256,
        num_heads=4,
        ff_dim=4,
        num_transformer_blocks=args.blocks,
        mlp_units=[128],
        mlp_dropout=0.4,
        dropout=0.25,
        n_classes=data[4],
        d_model=args.d_model,
    )
    train_and_report(model, data, args, model_kind="fixed")


if __name__ == "__main__":
    main()
