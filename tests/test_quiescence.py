"""Tests for Board.captureMoves and Engine.qsearch (quiescence search).

Runnable directly (python tests/test_quiescence.py) or under pytest.
Engine tests are skipped when no evaluator model exists in core/ai.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "core" / "game"))
sys.path.append(str(ROOT / "core" / "ai"))

from board import Board
from king import King
from pawn import Pawn
from queen import Queen
from rook import Rook


# Board coordinates: pieces[y][x], y=0 is black's back rank (rank 8),
# y=7 is white's back rank (rank 1). Moves are [y, x, newY, newX].


def empty_board():
	board = Board()
	board.pieces = [[None for _ in range(8)] for _ in range(8)]
	board.turn = 0
	return board


def put(board, piece_cls, x, y, white, filename, symbol):
	board.pieces[y][x] = piece_cls(x, y, white, filename, symbol)


def moves_set(moves):
	return set(tuple(move) for move in moves)


def noisy_from_all_moves(board, w):
	"""Captures + promotion pushes filtered from the slow full generator."""
	res = set()
	for move in board.allMoves(w):
		y, x, newY, newX = move
		if board.pieces[newY][newX] is not None:
			res.add(tuple(move))
		elif type(board.pieces[y][x]) is Pawn and newY in (0, 7):
			res.add(tuple(move))
	return res


def queen_vs_pawns_board():
	"""White Qd4 forks black pawns on c5 and e5; kings on h1/h8."""
	board = empty_board()
	put(board, King, 7, 7, True, 'whiteking.png', 'K')
	put(board, King, 7, 0, False, 'blackking.png', 'k')
	put(board, Queen, 3, 4, True, 'whitequeen.png', 'Q')
	put(board, Pawn, 2, 3, False, 'blackpawn.png', 'p')
	put(board, Pawn, 4, 3, False, 'blackpawn.png', 'p')
	board.update()
	return board


def test_initial_position_has_no_captures():
	board = Board()
	assert board.captureMoves(True) == []
	assert board.captureMoves(False) == []


def test_known_capture_set():
	board = queen_vs_pawns_board()
	assert moves_set(board.captureMoves(True)) == {(4, 3, 3, 2), (4, 3, 3, 4)}
	assert moves_set(board.captureMoves(False)) == {(3, 2, 4, 3), (3, 4, 4, 3)}


def test_pinned_capture_excluded():
	# White Re4 is pinned to Ke1 by black Re8; black pawn on a4 is bait.
	board = empty_board()
	put(board, King, 4, 7, True, 'whiteking.png', 'K')
	put(board, King, 7, 0, False, 'blackking.png', 'k')
	put(board, Rook, 4, 4, True, 'whiterook.png', 'R')
	put(board, Rook, 4, 0, False, 'blackrook.png', 'r')
	put(board, Pawn, 0, 4, False, 'blackpawn.png', 'p')
	board.update()
	white = moves_set(board.captureMoves(True))
	assert (4, 4, 0, 4) in white
	assert (4, 4, 4, 0) not in white
	assert white == {(4, 4, 0, 4)}


def test_promotion_pushes():
	board = empty_board()
	put(board, King, 7, 7, True, 'whiteking.png', 'K')
	put(board, King, 7, 0, False, 'blackking.png', 'k')
	put(board, Pawn, 0, 1, True, 'whitepawn.png', 'P')
	board.update()
	assert moves_set(board.captureMoves(True)) == {(1, 0, 0, 0)}

	# Occupied promotion square: pawns cannot capture straight ahead.
	put(board, Rook, 0, 0, False, 'blackrook.png', 'r')
	board.update()
	assert board.captureMoves(True) == []

	# Promotion capture and promotion push coexist, no duplicates.
	board.pieces[0][0] = None
	put(board, Rook, 1, 0, False, 'blackrook.png', 'r')
	board.update()
	result = board.captureMoves(True)
	assert moves_set(result) == {(1, 0, 0, 0), (1, 0, 0, 1)}
	assert len(result) == len(moves_set(result))


def test_capture_moves_match_all_moves():
	board = Board()
	board.makeMove(4, 6, 4, 4)  # e2-e4
	board.makeMove(3, 1, 3, 3)  # d7-d5
	for w in (True, False):
		assert moves_set(board.captureMoves(w)) == noisy_from_all_moves(board, w)

	board = queen_vs_pawns_board()
	for w in (True, False):
		assert moves_set(board.captureMoves(w)) == noisy_from_all_moves(board, w)


_ENGINE = None
_ENGINE_MISSING = False


def get_engine():
	"""One shared Engine (model load is slow); None when no model exists."""
	global _ENGINE, _ENGINE_MISSING
	if _ENGINE is None and not _ENGINE_MISSING:
		from engine import Engine
		try:
			_ENGINE = Engine(Board(), depth=1)
		except FileNotFoundError:
			_ENGINE_MISSING = True
	return _ENGINE


def test_qsearch_stands_pat_on_quiet_position():
	engine = get_engine()
	if engine is None:
		print("  SKIP: no evaluator model in core/ai")
		return
	from engine import MATE_SCORE
	board = empty_board()
	put(board, King, 7, 7, True, 'whiteking.png', 'K')
	put(board, King, 7, 0, False, 'blackking.png', 'k')
	put(board, Pawn, 0, 6, True, 'whitepawn.png', 'P')
	put(board, Pawn, 0, 1, False, 'blackpawn.png', 'p')
	board.update()
	engine.board = board
	stand_pat = engine.eval_position_for(True)
	value = engine.qsearch(6, True, -MATE_SCORE, MATE_SCORE)
	assert abs(value - stand_pat) < 1e-5, (value, stand_pat)


def test_engine_declines_poisoned_pawn():
	engine = get_engine()
	if engine is None:
		print("  SKIP: no evaluator model in core/ai")
		return
	# White Qd3 can take the d6 pawn, but e7 recaptures: a -8 line that
	# depth-1 search without quiescence cannot see.
	board = empty_board()
	put(board, King, 7, 7, True, 'whiteking.png', 'K')
	put(board, King, 7, 0, False, 'blackking.png', 'k')
	put(board, Queen, 3, 5, True, 'whitequeen.png', 'Q')
	put(board, Pawn, 3, 2, False, 'blackpawn.png', 'p')
	put(board, Pawn, 4, 1, False, 'blackpawn.png', 'p')
	board.update()
	engine.board = board
	move = engine.selectMove()
	assert move is not None
	assert tuple(move) != (5, 3, 2, 3), "engine grabbed the poisoned pawn"


def test_quiescence_off_matches_legacy_path():
	engine = get_engine()
	if engine is None:
		print("  SKIP: no evaluator model in core/ai")
		return
	engine.board = Board()
	engine.quiescence = False
	try:
		move = engine.selectMove()
	finally:
		engine.quiescence = True
	assert move is not None


def test_select_move_timing():
	engine = get_engine()
	if engine is None:
		print("  SKIP: no evaluator model in core/ai")
		return
	engine.board = Board()
	engine.eval_position()  # TF warmup
	start = time.perf_counter()
	move = engine.selectMove()
	elapsed = time.perf_counter() - start
	print(f"  selectMove from initial position: {elapsed:.1f}s")
	assert move is not None
	assert elapsed < 20.0, f"selectMove took {elapsed:.1f}s"


if __name__ == "__main__":
	tests = [
		test_initial_position_has_no_captures,
		test_known_capture_set,
		test_pinned_capture_excluded,
		test_promotion_pushes,
		test_capture_moves_match_all_moves,
		test_qsearch_stands_pat_on_quiet_position,
		test_engine_declines_poisoned_pawn,
		test_quiescence_off_matches_legacy_path,
		test_select_move_timing,
	]
	failures = 0
	for test in tests:
		try:
			test()
		except AssertionError as error:
			failures += 1
			print(f"FAIL {test.__name__}: {error}")
		else:
			print(f"PASS {test.__name__}")
	if failures:
		sys.exit(1)
	print("All tests passed.")
