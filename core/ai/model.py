"""The perspective-transformer-v3 position evaluator architecture.

Importing this module registers SquarePositionEmbedding, which .keras
deserialization of a saved evaluator needs.
"""

import tensorflow as tf

from board_features import (
	BOARD_SIZE,
	PERSPECTIVE_V3_BOARD_FEATURE_SIZE,
	PERSPECTIVE_V3_FEATURE_SIZE,
)


@tf.keras.utils.register_keras_serializable(package="ChessAI")
class SquarePositionEmbedding(tf.keras.layers.Layer):
	"""Add a learned absolute embedding for each of the 64 board squares."""

	def __init__(self, square_count=BOARD_SIZE * BOARD_SIZE, **kwargs):
		super().__init__(**kwargs)
		self.square_count = square_count

	def build(self, input_shape):
		self.position_embeddings = self.add_weight(
			name="position_embeddings",
			shape=(self.square_count, int(input_shape[-1])),
			initializer="random_normal",
			trainable=True,
		)

	def call(self, inputs):
		return inputs + self.position_embeddings[tf.newaxis, :, :]

	def get_config(self):
		config = super().get_config()
		config.update({"square_count": self.square_count})
		return config


def transformer_encoder_block(x, d_model, heads, ff_dim, dropout, block_index):
	attention_input = tf.keras.layers.LayerNormalization(
		epsilon=1e-6, name=f"transformer_{block_index}_attention_norm"
	)(x)
	attention_output = tf.keras.layers.MultiHeadAttention(
		num_heads=heads,
		key_dim=d_model // heads,
		dropout=dropout,
		name=f"transformer_{block_index}_attention",
	)(attention_input, attention_input)
	attention_output = tf.keras.layers.Dropout(
		dropout, name=f"transformer_{block_index}_attention_dropout"
	)(attention_output)
	x = tf.keras.layers.Add(name=f"transformer_{block_index}_attention_residual")(
		[x, attention_output]
	)

	ffn_input = tf.keras.layers.LayerNormalization(
		epsilon=1e-6, name=f"transformer_{block_index}_ffn_norm"
	)(x)
	ffn_output = tf.keras.layers.Dense(
		ff_dim, activation="gelu", name=f"transformer_{block_index}_ffn_expand"
	)(ffn_input)
	ffn_output = tf.keras.layers.Dropout(
		dropout, name=f"transformer_{block_index}_ffn_dropout_1"
	)(ffn_output)
	ffn_output = tf.keras.layers.Dense(
		d_model, name=f"transformer_{block_index}_ffn_project"
	)(ffn_output)
	ffn_output = tf.keras.layers.Dropout(
		dropout, name=f"transformer_{block_index}_ffn_dropout_2"
	)(ffn_output)
	return tf.keras.layers.Add(name=f"transformer_{block_index}_ffn_residual")(
		[x, ffn_output]
	)


def build_perspective_transformer_v3_model(
	d_model=384,
	heads=8,
	layers=6,
	ff_dim=1536,
	dropout=0.05,
):
	board_feature_size = PERSPECTIVE_V3_BOARD_FEATURE_SIZE
	inputs = tf.keras.Input(
		shape=(PERSPECTIVE_V3_FEATURE_SIZE,), name="position_features"
	)
	board_inputs = inputs[:, :board_feature_size]
	x = tf.keras.layers.Reshape(
		(64, board_feature_size // 64), name="square_feature_planes"
	)(board_inputs)
	x = tf.keras.layers.Dense(d_model, name="square_projection")(x)
	x = SquarePositionEmbedding(name="square_position_embedding")(x)
	x = tf.keras.layers.Dropout(dropout, name="input_dropout")(x)

	for block_index in range(1, layers + 1):
		x = transformer_encoder_block(
			x=x,
			d_model=d_model,
			heads=heads,
			ff_dim=ff_dim,
			dropout=dropout,
			block_index=block_index,
		)

	x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="final_norm")(x)
	mean_pool = tf.keras.layers.GlobalAveragePooling1D(name="square_mean_pool")(x)
	max_pool = tf.keras.layers.GlobalMaxPooling1D(name="square_max_pool")(x)

	metadata_inputs = inputs[:, board_feature_size:]
	metadata = tf.keras.layers.LayerNormalization(
		epsilon=1e-6, name="metadata_norm"
	)(metadata_inputs)
	metadata = tf.keras.layers.Dense(
		max(32, d_model // 2),
		activation="gelu",
		name="metadata_projection",
	)(metadata)

	x = tf.keras.layers.Concatenate(name="pooled_features")(
		[mean_pool, max_pool, metadata]
	)
	x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="head_norm")(x)
	x = tf.keras.layers.Dense(
		ff_dim, activation="gelu", name="evaluation_head_hidden_1"
	)(x)
	x = tf.keras.layers.Dropout(dropout, name="evaluation_head_dropout_1")(x)
	x = tf.keras.layers.Dense(
		max(64, ff_dim // 2),
		activation="gelu",
		name="evaluation_head_hidden_2",
	)(x)
	x = tf.keras.layers.Dropout(dropout, name="evaluation_head_dropout_2")(x)
	outputs = tf.keras.layers.Dense(
		1,
		dtype="float32",
		name="side_to_move_pawn_score",
	)(x)
	return tf.keras.Model(
		inputs=inputs, outputs=outputs, name="perspective_transformer_v3"
	)


def build_model_from_config(config):
	return build_perspective_transformer_v3_model(
		d_model=config.transformer_d_model,
		heads=config.transformer_heads,
		layers=config.transformer_layers,
		ff_dim=config.transformer_ff_dim,
		dropout=config.transformer_dropout,
	)
