"""TensorFlow-facing position encoding module.

The pure-Python encoding lives in board_features (kept TensorFlow-free so
dataset pre-encoding workers can import it cheaply); everything is re-exported
here for backward compatibility.
"""

import tensorflow as tf

from board_features import (  # noqa: F401
	ATTACK_BOARD_FEATURE_SIZE,
	ATTACK_FEATURE_SIZE,
	ATTACK_PLANE_COUNT,
	BOARD_FEATURE_SIZE,
	BOARD_SIZE,
	FEATURE_SIZE,
	MAX_SIDE_MATERIAL,
	MAX_TOTAL_PHASE,
	METADATA_SIZE,
	PAPER_BITMAP_FEATURE_SIZE,
	PAPER_MAX_ABS_CENTIPAWNS,
	PAPER_PAWN_TARGET_LIMIT,
	PERSPECTIVE_FEATURE_SIZE,
	PERSPECTIVE_METADATA_SIZE,
	PERSPECTIVE_V2_FEATURE_SIZE,
	PERSPECTIVE_V2_METADATA_SIZE,
	PHASE_WEIGHTS,
	PIECE_PLANE_COUNT,
	PIECE_PLANES,
	PIECE_VALUES,
	RELATIVE_PIECE_PLANES,
	encode_board,
	encode_board_paper_bitmap,
	encode_board_perspective,
	encode_board_perspective_v2,
	encode_fen,
	encode_fen_paper_bitmap,
	encode_fen_perspective,
	encode_fen_perspective_v2,
	evaluation_to_paper_pawns,
	evaluation_to_paper_target,
	evaluation_to_pawns,
	hash_fraction,
	paper_target_to_pawns,
	position_key,
	selected_for_sample,
	split_for_fen,
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
