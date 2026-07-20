import copy
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from board_features import (
	PERSPECTIVE_V3_FEATURE_SIZE,
	PIECE_VALUES,
	board_position_key,
	encode_board_perspective_v3,
)
# Also registers SquarePositionEmbedding, which .keras deserialization needs.
from model import build_perspective_transformer_v3_model


MATE_SCORE = 100000.0
DEFAULT_SEARCH_DEPTH = 1
DEFAULT_QUIESCENCE_DEPTH = 6
DELTA_MARGIN = 2.0
PROMOTION_GAIN = 8.0
EVAL_CACHE_MAX_ENTRIES = 200000
TT_MAX_ENTRIES = 500000
TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2
PERSPECTIVE_V3_WEIGHTS_NAME = "position_evaluator_perspective_transformer_v3_mae.weights.h5"


class SearchTimeout(Exception):
	"""Raised inside the search when the selectMove time budget expires."""


class Engine:
	def __init__(
		self,
		board,
		model_path=None,
		depth=DEFAULT_SEARCH_DEPTH,
		quiescence=True,
		quiescence_depth=DEFAULT_QUIESCENCE_DEPTH,
	):
		self.board = board
		self.depth = depth
		self.quiescence = quiescence
		self.quiescence_depth = quiescence_depth
		self.qsearch_checks = True
		self.use_tt = True
		self._deadline = None
		self._eval_cache = {}
		self._tt = {}
		self.game_history = {}
		self.model_path = self._resolve_model_path(model_path)
		self.nn = self._load_model(self.model_path)

	def _resolve_model_path(self, model_path):
		if model_path is not None:
			path = Path(model_path)
			if not path.exists():
				raise FileNotFoundError(path)
			return path

		model_dir = Path(__file__).resolve().parent
		for name in (
			PERSPECTIVE_V3_WEIGHTS_NAME,
			"position_evaluator_perspective_transformer_v3_mae.keras",
		):
			path = model_dir / name
			if path.exists():
				return path
		raise FileNotFoundError("no evaluator model found in core/ai")

	def _load_model(self, path):
		if path.name.endswith(".weights.h5"):
			if path.name != PERSPECTIVE_V3_WEIGHTS_NAME:
				raise ValueError(
					f"cannot infer evaluator architecture from weights file "
					f"{path}; pass a full .keras model instead"
				)
			model = build_perspective_transformer_v3_model()
			model.load_weights(str(path))
		else:
			model = tf.keras.models.load_model(str(path), compile=False)
		input_shape = model.input_shape
		if isinstance(input_shape, list):
			input_shape = input_shape[0]
		if int(input_shape[-1]) != PERSPECTIVE_V3_FEATURE_SIZE:
			raise ValueError(
				f"evaluator input size {input_shape[-1]} does not match the "
				f"perspective-v3 feature size {PERSPECTIVE_V3_FEATURE_SIZE}"
			)
		return model

	def nn_input(self):
		return encode_board_perspective_v3(self.board).reshape(
			1, PERSPECTIVE_V3_FEATURE_SIZE
		)

	def _predict_model(self, inp):
		inp = tf.convert_to_tensor(inp, dtype=tf.float32)
		return self.nn(inp, training=False)

	def _position_key(self):
		return board_position_key(self.board)

	def set_game_history(self, history):
		"""Set position-key -> occurrence counts for the game in progress.

		Any searched position already in the history scores as a draw, so
		TT entries computed under an older history are stale; the eval
		cache is history-independent and survives.
		"""
		self.game_history = history
		self._tt.clear()

	def _store_eval(self, key, score):
		if len(self._eval_cache) >= EVAL_CACHE_MAX_ENTRIES:
			self._eval_cache.clear()
		self._eval_cache[key] = score

	def eval_position(self):
		key = self._position_key()
		cached = self._eval_cache.get(key)
		if cached is not None:
			return cached
		eval_tensor = self._predict_model(self.nn_input())
		score = float(np.asarray(eval_tensor)[0][0])
		self._store_eval(key, score)
		return score

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

	def _capture_gain(self, move):
		y, x, newY, newX = move
		victim = self.board.pieces[newY][newX]
		if victim is not None:
			return PIECE_VALUES[victim.symbol.lower()]
		if self.board.pieces[y][x].symbol.lower() == "p" and newY in (0, 7):
			return PROMOTION_GAIN
		return 0.0

	def _order_noisy_moves(self, moves):
		def order_key(move):
			y, x, newY, newX = move
			attacker = self.board.pieces[y][x]
			return (
				-self._capture_gain(move),
				PIECE_VALUES[attacker.symbol.lower()],
			)

		return sorted(moves, key=order_key)

	def _order_moves(self, moves, hint_move=None):
		hint = None if hint_move is None else list(hint_move)

		def order_key(move):
			y, x, newY, newX = move
			victim = self.board.pieces[newY][newX]
			gain = 0.0 if victim is None else PIECE_VALUES[victim.symbol.lower()]
			return (
				0 if list(move) == hint else 1,
				-gain,
				PIECE_VALUES[self.board.pieces[y][x].symbol.lower()],
			)

		return sorted(moves, key=order_key)

	def _batch_scores(self, features):
		outputs = np.asarray(self._predict_model(np.vstack(features)))
		return [float(value) for value in outputs.reshape(-1)]

	def _child_stand_pats(self, moves, root_white):
		"""Batched stand-pat scores for the positions after each move:
		one GPU call for the cache misses instead of one call per child."""
		scores = [None] * len(moves)
		features = []
		miss_keys = []
		miss_indices = []
		for index, move in enumerate(moves):
			y, x, newY, newX = move
			snapshot = self.snapshot_board()
			self.board.makeMove(x, y, newX, newY)
			key = self._position_key()
			cached = self._eval_cache.get(key)
			if cached is None:
				features.append(self.nn_input())
				miss_keys.append(key)
				miss_indices.append(index)
			else:
				scores[index] = cached
			self.restore_board(snapshot)
		if features:
			for key, index, score in zip(
				miss_keys, miss_indices, self._batch_scores(features)
			):
				self._store_eval(key, score)
				scores[index] = score
		# Every child has the same side to move: the parent's opponent.
		child_side_is_white = self.board.turn != 0
		if child_side_is_white == root_white:
			return scores
		return [-score for score in scores]

	def _check_deadline(self):
		if self._deadline is not None and time.perf_counter() > self._deadline:
			raise SearchTimeout()

	def qsearch(self, qdepth, root_white, alpha, beta, stand_pat=None):
		self._check_deadline()
		# Game-history repetitions score as draws (see minimax); qsearch is
		# never entered at the root, so no ply guard is needed. This also
		# catches perpetual-check loops inside the check extension.
		if self.game_history and self.game_history.get(self._position_key()):
			return 0.0
		side_to_move_is_white = self.board.turn == 0
		is_max = side_to_move_is_white == root_white

		if self.board.checked(side_to_move_is_white):
			# Standing pat while in check is unsound: the position may be a
			# mate the evaluator cannot see. Search every evasion instead.
			moves = self.board.allMoves(side_to_move_is_white)
			if len(moves) == 0:
				return self.terminal_eval_for(root_white)
			if qdepth <= 0:
				return self.eval_position_for(root_white)
			stand_pats = None
			if len(moves) > 1:
				stand_pats = self._child_stand_pats(moves, root_white)
			best = -MATE_SCORE if is_max else MATE_SCORE
			for index, move in enumerate(moves):
				y, x, newY, newX = move
				snapshot = self.snapshot_board()
				self.board.makeMove(x, y, newX, newY)
				value = self.qsearch(
					qdepth - 1, root_white, alpha, beta,
					stand_pat=None if stand_pats is None else stand_pats[index],
				)
				self.restore_board(snapshot)
				if is_max:
					best = max(best, value)
					alpha = max(alpha, best)
				else:
					best = min(best, value)
					beta = min(beta, best)
				if alpha >= beta:
					break
			return best

		if stand_pat is None:
			stand_pat = self.eval_position_for(root_white)
		if qdepth <= 0:
			return stand_pat
		if is_max:
			if stand_pat >= beta:
				return stand_pat
			alpha = max(alpha, stand_pat)
		else:
			if stand_pat <= alpha:
				return stand_pat
			beta = min(beta, stand_pat)

		# Quiet checking moves are generated only in the first plies of
		# quiescence: horizon zwischenzugs live there, while checks deeper
		# inside a capture chain explode the tree for little value.
		include_checks = (
			self.qsearch_checks and qdepth >= self.quiescence_depth - 1
		)
		# A position with no noisy moves stands pat here; a stalemate is
		# scored by the evaluator rather than as a draw, which avoids a
		# full legal-move scan per node.
		best = stand_pat
		noisy_moves = self._order_noisy_moves(
			self.board.noisyMoves(side_to_move_is_white, include_checks)
		)
		searchable = []
		for move in noisy_moves:
			gain = self._capture_gain(move)
			# Delta pruning: skip when even winning this material plus a
			# noise margin cannot move the window. Quiet checks win no
			# material and are searched for mate/fork threats instead.
			if gain > 0.0:
				if is_max and stand_pat + gain + DELTA_MARGIN <= alpha:
					continue
				if not is_max and stand_pat - gain - DELTA_MARGIN >= beta:
					continue
			searchable.append(move)
		stand_pats = None
		if len(searchable) > 1:
			stand_pats = self._child_stand_pats(searchable, root_white)
		for index, move in enumerate(searchable):
			y, x, newY, newX = move
			snapshot = self.snapshot_board()
			self.board.makeMove(x, y, newX, newY)
			value = self.qsearch(
				qdepth - 1, root_white, alpha, beta,
				stand_pat=None if stand_pats is None else stand_pats[index],
			)
			self.restore_board(snapshot)
			if is_max:
				best = max(best, value)
				alpha = max(alpha, best)
			else:
				best = min(best, value)
				beta = min(beta, best)
			if alpha >= beta:
				break
		return best

	def minimax(self, depth, root_white, alpha, beta, hint_move=None, stand_pat=None, ply=0):
		side_to_move_is_white = self.board.turn == 0

		# A position the game has already visited scores as a draw: reaching
		# it again heads toward threefold repetition (lichess auto-adjudicates
		# the third occurrence). The ply guard exempts the root, whose own
		# position is always in the history. Checked before the TT so no
		# history-dependent draw score is ever probed or stored.
		if ply and self.game_history and self.game_history.get(self._position_key()):
			return 0.0, None

		if depth == 0:
			if self.quiescence:
				return self.qsearch(
					self.quiescence_depth, root_white, alpha, beta, stand_pat
				), None
			moves = self.board.allMoves(side_to_move_is_white)
			if len(moves) == 0:
				return self.terminal_eval_for(root_white), None
			if stand_pat is not None:
				return stand_pat, None
			return self.eval_position_for(root_white), None

		# Search options are part of the TT key because they change what a
		# stored value means.
		tt_key = None
		if self.use_tt:
			tt_key = (
				self._position_key(),
				root_white,
				self.quiescence,
				self.qsearch_checks,
			)
			entry = self._tt.get(tt_key)
			if entry is not None:
				entry_depth, flag, value, tt_move = entry
				if entry_depth >= depth:
					if flag == TT_EXACT:
						return value, tt_move
					if flag == TT_LOWER:
						alpha = max(alpha, value)
					else:
						beta = min(beta, value)
					if alpha >= beta:
						return value, tt_move
				if hint_move is None:
					hint_move = tt_move
		alpha_in = alpha
		beta_in = beta

		moves = self.board.allMoves(side_to_move_is_white)
		if len(moves) == 0:
			return self.terminal_eval_for(root_white), None
		moves = self._order_moves(moves, hint_move)

		# One batched model call for all leaf children instead of one call
		# per child inside qsearch.
		stand_pats = None
		if depth == 1:
			stand_pats = self._child_stand_pats(moves, root_white)

		is_max = side_to_move_is_white == root_white
		best = -MATE_SCORE if is_max else MATE_SCORE
		best_move = None
		for index, move in enumerate(moves):
			self._check_deadline()
			y, x, newY, newX = move
			snapshot = self.snapshot_board()
			self.board.makeMove(x, y, newX, newY)
			value, _ = self.minimax(
				depth - 1,
				root_white,
				alpha,
				beta,
				stand_pat=None if stand_pats is None else stand_pats[index],
				ply=ply + 1,
			)
			self.restore_board(snapshot)

			if is_max:
				if value > best or best_move is None:
					best = value
					best_move = move
				alpha = max(alpha, best)
			else:
				if value < best or best_move is None:
					best = value
					best_move = move
				beta = min(beta, best)
			if alpha >= beta:
				break

		if tt_key is not None and best_move is not None:
			if best <= alpha_in:
				flag = TT_UPPER
			elif best >= beta_in:
				flag = TT_LOWER
			else:
				flag = TT_EXACT
			if len(self._tt) >= TT_MAX_ENTRIES:
				self._tt.clear()
			self._tt[tt_key] = (depth, flag, best, best_move)
		return best, best_move

	def selectMove(self, white=None, time_budget=None, max_depth=None):
		if white is not None and white != (self.board.turn == 0):
			raise ValueError("selectMove side does not match board.turn")
		root_white = self.board.turn == 0
		if max_depth is None:
			max_depth = self.depth
		start = time.perf_counter()
		self._deadline = None if time_budget is None else start + time_budget

		# Seed with an ordered legal move. A search that runs out of time
		# before completing even the first iteration must still return
		# something playable; None is reserved for "no legal moves", which
		# is how callers distinguish checkmate and stalemate.
		root_moves = self.board.allMoves(root_white)
		if not root_moves:
			return None
		best_move = self._order_moves(root_moves)[0]

		try:
			for depth in range(1, max_depth + 1):
				if depth > 1 and time_budget is not None:
					# A deeper iteration costs a multiple of everything so
					# far; don't start one that is unlikely to finish.
					if time.perf_counter() - start > 0.25 * time_budget:
						break
				snapshot = self.snapshot_board()
				try:
					_, move = self.minimax(
						depth, root_white, -MATE_SCORE, MATE_SCORE,
						hint_move=best_move,
					)
				except SearchTimeout:
					self.restore_board(snapshot)
					break
				if move is not None:
					best_move = move
		finally:
			self._deadline = None
		return best_move
