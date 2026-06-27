import copy
from pathlib import Path

import numpy as np
import tensorflow as tf

from position_encoding import (
	ATTACK_FEATURE_SIZE,
	FEATURE_SIZE,
	PAPER_BITMAP_FEATURE_SIZE,
	PERSPECTIVE_FEATURE_SIZE,
	encode_board,
	encode_board_paper_bitmap,
	encode_board_perspective,
	paper_target_to_pawns,
)


MATE_SCORE = 100000.0
DEFAULT_SEARCH_DEPTH = 1


class Engine:
	def __init__(self, board, model_path=None, depth=DEFAULT_SEARCH_DEPTH):
		self.board = board
		self.depth = depth
		self.model_path = self._resolve_model_path(model_path)
		self.nn, self.model_kind = self._load_model(self.model_path)
		self.model_input_size = self._model_input_size(self.nn, self.model_kind)
		self.paper_output_is_normalized = self._paper_output_is_normalized()
		self.encoder = None

		if self.model_input_size == 128:
			self.encoder = tf.saved_model.load(str(Path(__file__).resolve().parent / "encoder_model"))
		elif self.model_input_size not in (
			None,
			FEATURE_SIZE,
			ATTACK_FEATURE_SIZE,
			PAPER_BITMAP_FEATURE_SIZE,
			PERSPECTIVE_FEATURE_SIZE,
		):
			raise ValueError(
				f"unsupported evaluator input size {self.model_input_size}; "
				f"expected {FEATURE_SIZE}, {ATTACK_FEATURE_SIZE}, "
				f"{PAPER_BITMAP_FEATURE_SIZE}, {PERSPECTIVE_FEATURE_SIZE}, "
				"or legacy 128"
			)

	def _resolve_model_path(self, model_path):
		if model_path is not None:
			path = Path(model_path)
			if not path.exists():
				raise FileNotFoundError(path)
			return path

		model_dir = Path(__file__).resolve().parent
		for name in (
			"position_evaluator_transformer_mae.keras",
			"position_evaluator_paper_mlp_mae.keras",
			"position_evaluator_paper_mlp.keras",
			"position_evaluator_cnn_v2.keras",
			"position_evaluator_cnn_attacks.keras",
			"position_evaluator_cnn.keras",
			"position_evaluator.keras",
			"position_evaluator",
			"my_model_v2",
		):
			path = model_dir / name
			if path.exists():
				return path
		raise FileNotFoundError("no evaluator model found in core/ai")

	def _load_model(self, path):
		try:
			return tf.keras.models.load_model(str(path), compile=False), "keras"
		except Exception:
			return tf.saved_model.load(str(path)), "saved_model"

	def _model_input_size(self, model, model_kind):
		if model_kind == "keras":
			input_shape = model.input_shape
			if isinstance(input_shape, list):
				input_shape = input_shape[0]
			if input_shape is not None and input_shape[-1] is not None:
				return int(input_shape[-1])

		signature = getattr(model, "signatures", {}).get("serving_default")
		if signature is not None:
			_, kwargs = signature.structured_input_signature
			for spec in kwargs.values():
				if spec.shape[-1] is not None:
					return int(spec.shape[-1])

		first_layer = getattr(model, "layer-0", None)
		if first_layer is not None:
			for variable in first_layer.variables:
				if "kernel" in variable.name and len(variable.shape) >= 2:
					return int(variable.shape[0])

		if self.model_path.name == "my_model_v2":
			return 128
		return FEATURE_SIZE

	def _run_layers(self, model, inp, stop_layer):
		out = tf.convert_to_tensor(inp, dtype=tf.float32)
		for idx in range(stop_layer + 1):
			out = getattr(model, f"layer-{idx}")(out)
		return out

	def _predict_model(self, inp):
		inp = tf.convert_to_tensor(inp, dtype=tf.float32)
		if self.model_kind == "keras":
			return self.nn(inp, training=False)

		signature = getattr(self.nn, "signatures", {}).get("serving_default")
		if signature is not None:
			_, kwargs = signature.structured_input_signature
			if kwargs:
				input_name = next(iter(kwargs))
				outputs = signature(**{input_name: inp})
			else:
				outputs = signature(inp)
			return next(iter(outputs.values()))

		return self._run_layers(self.nn, inp, 3)

	def nn_input(self):
		if self.model_input_size == PAPER_BITMAP_FEATURE_SIZE:
			return encode_board_paper_bitmap(self.board).reshape(
				1, PAPER_BITMAP_FEATURE_SIZE
			)

		if self.model_input_size == PERSPECTIVE_FEATURE_SIZE:
			return encode_board_perspective(self.board).reshape(
				1, PERSPECTIVE_FEATURE_SIZE
			)

		include_attack_maps = self.model_input_size == ATTACK_FEATURE_SIZE
		feature_size = ATTACK_FEATURE_SIZE if include_attack_maps else FEATURE_SIZE
		features = encode_board(
			self.board, include_attack_maps=include_attack_maps
		).reshape(1, feature_size)
		if self.model_input_size == 128:
			return self._run_layers(self.encoder, features, 2)
		return features

	def eval_position(self):
		eval_tensor = self._predict_model(self.nn_input())
		score = float(np.asarray(eval_tensor)[0][0])
		if self.paper_output_is_normalized:
			return paper_target_to_pawns(score)
		return score

	def _paper_output_is_normalized(self):
		if self.model_input_size != PAPER_BITMAP_FEATURE_SIZE:
			return False
		if self.model_kind == "keras":
			output_names = list(getattr(self.nn, "output_names", []))
			layer_name = getattr(self.nn.layers[-1], "name", "")
			names = set(output_names + [layer_name])
			if "side_to_move_pawn_score" in names:
				return False
			if "normalized_evaluation" in names:
				return True
		return "mae" not in self.model_path.stem

	def eval_position_for(self, root_white):
		side_to_move_is_white = self.board.turn == 0
		score = self.eval_position()
		if side_to_move_is_white == root_white:
			return score
		return -score

	def snapshot_board(self):
		return copy.deepcopy(self.board.pieces), self.board.turn

	def restore_board(self, snapshot):
		self.board.pieces, self.board.turn = snapshot

	def terminal_eval_for(self, root_white):
		side_to_move_is_white = self.board.turn == 0
		if self.board.checked(side_to_move_is_white):
			if side_to_move_is_white == root_white:
				return -MATE_SCORE
			return MATE_SCORE
		return 0.0

	def minimax(self, depth, root_white, alpha, beta):
		side_to_move_is_white = self.board.turn == 0
		moves = self.board.allMoves(side_to_move_is_white)
		if len(moves) == 0:
			return self.terminal_eval_for(root_white), None
		if depth == 0:
			return self.eval_position_for(root_white), None

		if side_to_move_is_white == root_white:
			best = -MATE_SCORE
			best_move = None
			for move in moves:
				y, x, newY, newX = move
				snapshot = self.snapshot_board()
				self.board.makeMove(x, y, newX, newY)
				value, _ = self.minimax(depth - 1, root_white, alpha, beta)
				self.restore_board(snapshot)

				if value > best:
					best = value
					best_move = move
				alpha = max(alpha, best)
				if alpha >= beta:
					break
			return best, best_move

		best = MATE_SCORE
		best_move = None
		for move in moves:
			y, x, newY, newX = move
			snapshot = self.snapshot_board()
			self.board.makeMove(x, y, newX, newY)
			value, _ = self.minimax(depth - 1, root_white, alpha, beta)
			self.restore_board(snapshot)

			if value < best:
				best = value
				best_move = move
			beta = min(beta, best)
			if alpha >= beta:
				break
		return best, best_move

	def selectMove(self, white=None):
		if white is not None and white != (self.board.turn == 0):
			raise ValueError("selectMove side does not match board.turn")
		root_white = self.board.turn == 0
		best_eval, best_move = self.minimax(self.depth, root_white, -MATE_SCORE, MATE_SCORE)
		return best_move
