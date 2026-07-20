"""Perspective-v3 position encoding and dataset split helpers.

Must stay TensorFlow-free: dataset encoder worker processes import it.
"""

import hashlib

import numpy as np


BOARD_SIZE = 8
PIECE_PLANES = {
	"P": 0,
	"R": 1,
	"N": 2,
	"B": 3,
	"Q": 4,
	"K": 5,
	"p": 6,
	"r": 7,
	"n": 8,
	"b": 9,
	"q": 10,
	"k": 11,
}
PIECE_PLANE_COUNT = len(PIECE_PLANES)
ATTACK_PLANE_COUNT = len(PIECE_PLANES)
TACTICAL_PLANE_COUNT = 4
PERSPECTIVE_V3_PLANE_COUNT = (
	PIECE_PLANE_COUNT + ATTACK_PLANE_COUNT + TACTICAL_PLANE_COUNT
)
PERSPECTIVE_V3_BOARD_FEATURE_SIZE = (
	BOARD_SIZE * BOARD_SIZE * PERSPECTIVE_V3_PLANE_COUNT
)
PERSPECTIVE_METADATA_SIZE = 6
PERSPECTIVE_V2_METADATA_SIZE = 24
PERSPECTIVE_V3_METADATA_SIZE = PERSPECTIVE_V2_METADATA_SIZE + 8
PERSPECTIVE_V3_FEATURE_SIZE = (
	PERSPECTIVE_V3_BOARD_FEATURE_SIZE + PERSPECTIVE_V3_METADATA_SIZE
)
RELATIVE_PIECE_PLANES = {
	"p": 0,
	"r": 1,
	"n": 2,
	"b": 3,
	"q": 4,
	"k": 5,
}
PIECE_VALUES = {
	"p": 1.0,
	"n": 3.0,
	"b": 3.0,
	"r": 5.0,
	"q": 9.0,
	"k": 0.0,
}
PHASE_WEIGHTS = {
	"p": 0.0,
	"n": 1.0,
	"b": 1.0,
	"r": 2.0,
	"q": 4.0,
	"k": 0.0,
}
MAX_SIDE_MATERIAL = 39.0
MAX_TOTAL_PHASE = 24.0


def evaluation_to_pawns(evaluation, max_abs_pawns=10.0):
	value = str(evaluation).strip()
	if not value:
		raise ValueError("empty evaluation")

	if value.startswith("#"):
		if len(value) < 2 or value[1] not in "+-":
			raise ValueError(f"invalid mate evaluation: {value}")
		score = max_abs_pawns if value[1] == "+" else -max_abs_pawns
	else:
		score = int(value) / 100.0

	return float(np.clip(score, -max_abs_pawns, max_abs_pawns))


def position_key(fen):
	fields = str(fen).strip().split()
	if len(fields) < 4:
		raise ValueError(f"invalid FEN: {fen}")
	return " ".join(fields[:4])


def board_position_key(board):
	"""Hashable identity of a Board position for caching and repetition.

	Placement symbols + side to move + castling rights fully determine
	both the evaluator features and the legal moves: pawn hasMoved is
	derivable from the rank and there is no en passant.
	"""
	chars = []
	for row in board.pieces:
		for piece in row:
			chars.append("." if piece is None else piece.symbol)
	return (
		"".join(chars),
		board.turn,
		board.has_kingside_castling_rights(1),
		board.has_queenside_castling_rights(1),
		board.has_kingside_castling_rights(0),
		board.has_queenside_castling_rights(0),
	)


def hash_fraction(seed, value):
	digest = hashlib.blake2b(
		f"{seed}\0{value}".encode("utf-8"), digest_size=8
	).digest()
	return int.from_bytes(digest, "big") / 2**64


def split_for_fen(fen, split_seed):
	value = hash_fraction(split_seed, position_key(fen))
	if value < 0.8:
		return "train"
	if value < 0.9:
		return "validation"
	return "test"


def selected_for_sample(fen, sample_fraction, sample_seed):
	if sample_fraction >= 1.0:
		return True
	return hash_fraction(sample_seed, position_key(fen)) < sample_fraction


def sample_fraction_for(raw_rows, sample_size):
	if sample_size == 0 or sample_size >= raw_rows:
		return 1.0
	return sample_size / raw_rows


def encode_fen_perspective_v3(fen, mirror_files=False):
	fields = str(fen).strip().split()
	if len(fields) < 4:
		raise ValueError(f"invalid FEN: {fen}")

	board_part, turn, castling = fields[0], fields[1], fields[2]
	if turn not in ("w", "b"):
		raise ValueError(f"invalid FEN turn field: {turn}")

	return _encode_perspective_symbols(
		board_symbols=_symbols_from_fen_board(board_part),
		white_to_move=(turn == "w"),
		castling_rights={
			"K": "K" in castling,
			"Q": "Q" in castling,
			"k": "k" in castling,
			"q": "q" in castling,
		},
		mirror_files=mirror_files,
	)


def encode_board_perspective_v3(board):
	board_symbols = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
	for row in board.pieces:
		for piece in row:
			if piece is not None:
				board_symbols[piece.y][piece.x] = piece.symbol

	return _encode_perspective_symbols(
		board_symbols=board_symbols,
		white_to_move=_board_white_to_move(board),
		castling_rights=_board_castling_rights(board),
		mirror_files=False,
	)


def _symbols_from_fen_board(board_part):
	rows = board_part.split("/")
	if len(rows) != BOARD_SIZE:
		raise ValueError(f"invalid FEN board rows: {board_part}")

	board_symbols = []
	for row in rows:
		symbols = []
		for char in row:
			if char.isdigit():
				symbols.extend([None] * int(char))
			elif char in PIECE_PLANES:
				symbols.append(char)
			else:
				raise ValueError(f"invalid FEN piece symbol: {char}")

		if len(symbols) != BOARD_SIZE:
			raise ValueError(f"invalid FEN row width: {row}")
		board_symbols.append(symbols)

	return board_symbols


def _encode_perspective_symbols(
	board_symbols,
	white_to_move,
	castling_rights,
	mirror_files,
):
	board_features = np.zeros(
		(BOARD_SIZE, BOARD_SIZE, PERSPECTIVE_V3_PLANE_COUNT), dtype=np.float32
	)
	# Attacker counts accumulate in feature coordinates so the derived
	# tactical planes mirror exactly like every other plane.
	own_attack_counts = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int32)
	opponent_attack_counts = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int32)
	own_squares = []
	opponent_squares = []

	for y, row in enumerate(board_symbols):
		for x, symbol in enumerate(row):
			if symbol is None:
				continue

			piece_is_white = symbol.isupper()
			piece_is_own = piece_is_white == white_to_move
			player_offset = 0 if piece_is_own else 6
			piece_plane = player_offset + RELATIVE_PIECE_PLANES[symbol.lower()]
			feature_y = y if white_to_move else BOARD_SIZE - 1 - y
			feature_x = BOARD_SIZE - 1 - x if mirror_files else x
			board_features[feature_y, feature_x, piece_plane] = 1.0
			squares = own_squares if piece_is_own else opponent_squares
			squares.append((feature_y, feature_x, symbol.lower()))
			attack_counts = (
				own_attack_counts if piece_is_own else opponent_attack_counts
			)

			attack_plane = PIECE_PLANE_COUNT + piece_plane
			for attack_y, attack_x in _piece_attack_squares(
				board_symbols, y, x, symbol
			):
				feature_attack_y = (
					attack_y if white_to_move else BOARD_SIZE - 1 - attack_y
				)
				feature_attack_x = (
					BOARD_SIZE - 1 - attack_x if mirror_files else attack_x
				)
				board_features[
					feature_attack_y, feature_attack_x, attack_plane
				] = 1.0
				attack_counts[feature_attack_y, feature_attack_x] += 1

	tactical_metadata = _apply_tactical_features(
		board_features=board_features,
		own_attack_counts=own_attack_counts,
		opponent_attack_counts=opponent_attack_counts,
		own_squares=own_squares,
		opponent_squares=opponent_squares,
	)

	features = np.zeros(PERSPECTIVE_V3_FEATURE_SIZE, dtype=np.float32)
	features[:PERSPECTIVE_V3_BOARD_FEATURE_SIZE] = board_features.reshape(-1)
	metadata = features[PERSPECTIVE_V3_BOARD_FEATURE_SIZE:]
	metadata[0] = 1.0 if _king_in_check(board_symbols, white_to_move) else 0.0
	metadata[1] = 1.0 if _king_in_check(board_symbols, not white_to_move) else 0.0

	castling_keys = _perspective_castling_keys(white_to_move, mirror_files)
	for index, key in enumerate(castling_keys, start=2):
		metadata[index] = 1.0 if castling_rights.get(key, False) else 0.0

	metadata[PERSPECTIVE_METADATA_SIZE:PERSPECTIVE_V2_METADATA_SIZE] = (
		_perspective_extra_metadata(
			board_symbols=board_symbols,
			white_to_move=white_to_move,
		)
	)
	metadata[PERSPECTIVE_V2_METADATA_SIZE:] = tactical_metadata
	return features


def _apply_tactical_features(
	board_features,
	own_attack_counts,
	opponent_attack_counts,
	own_squares,
	opponent_squares,
):
	"""Fill the four tactical planes; return the eight tactical scalars.

	Hanging = a non-king piece attacked at least once and defended zero
	times: a static-exchange approximation that ignores pins and attacker
	values, cheap enough for the dataset encoder.
	"""
	base = PIECE_PLANE_COUNT + ATTACK_PLANE_COUNT
	board_features[:, :, base] = own_attack_counts >= 2
	board_features[:, :, base + 1] = opponent_attack_counts >= 2

	own_king_square = None
	opponent_king_square = None
	own_hanging_count = 0
	own_hanging_material = 0.0
	own_attacked_count = 0
	opponent_hanging_count = 0
	opponent_hanging_material = 0.0
	opponent_attacked_count = 0

	for feature_y, feature_x, piece_type in own_squares:
		if piece_type == "k":
			own_king_square = (feature_y, feature_x)
			continue
		if opponent_attack_counts[feature_y, feature_x] > 0:
			own_attacked_count += 1
			if own_attack_counts[feature_y, feature_x] == 0:
				own_hanging_count += 1
				own_hanging_material += PIECE_VALUES[piece_type]
				board_features[feature_y, feature_x, base + 2] = 1.0

	for feature_y, feature_x, piece_type in opponent_squares:
		if piece_type == "k":
			opponent_king_square = (feature_y, feature_x)
			continue
		if own_attack_counts[feature_y, feature_x] > 0:
			opponent_attacked_count += 1
			if opponent_attack_counts[feature_y, feature_x] == 0:
				opponent_hanging_count += 1
				opponent_hanging_material += PIECE_VALUES[piece_type]
				board_features[feature_y, feature_x, base + 3] = 1.0

	return np.array(
		[
			own_hanging_count / 8.0,
			opponent_hanging_count / 8.0,
			own_hanging_material / MAX_SIDE_MATERIAL,
			opponent_hanging_material / MAX_SIDE_MATERIAL,
			_king_zone_attack_count(opponent_attack_counts, own_king_square) / 16.0,
			_king_zone_attack_count(own_attack_counts, opponent_king_square) / 16.0,
			own_attacked_count / 16.0,
			opponent_attacked_count / 16.0,
		],
		dtype=np.float32,
	)


def _king_zone_attack_count(attack_counts, king_square):
	if king_square is None:
		return 0
	y, x = king_square
	y_start, y_stop = max(0, y - 1), min(BOARD_SIZE, y + 2)
	x_start, x_stop = max(0, x - 1), min(BOARD_SIZE, x + 2)
	return int(attack_counts[y_start:y_stop, x_start:x_stop].sum())


def _perspective_castling_keys(white_to_move, mirror_files):
	if white_to_move:
		own_keys = ("K", "Q")
		opponent_keys = ("k", "q")
	else:
		own_keys = ("k", "q")
		opponent_keys = ("K", "Q")

	# Mirroring swaps kingside and queenside.
	if mirror_files:
		own_keys = (own_keys[1], own_keys[0])
		opponent_keys = (opponent_keys[1], opponent_keys[0])
	return own_keys + opponent_keys


def _perspective_extra_metadata(board_symbols, white_to_move):
	own_counts = _empty_piece_counts()
	opponent_counts = _empty_piece_counts()
	own_attacks = set()
	opponent_attacks = set()

	for y, row in enumerate(board_symbols):
		for x, symbol in enumerate(row):
			if symbol is None:
				continue

			counts = own_counts
			attacks = own_attacks
			if symbol.isupper() != white_to_move:
				counts = opponent_counts
				attacks = opponent_attacks
			piece_type = symbol.lower()
			counts[piece_type] += 1
			attacks.update(_piece_attack_squares(board_symbols, y, x, symbol))

	own_material = _material_count(own_counts)
	opponent_material = _material_count(opponent_counts)
	total_phase = _phase_count(own_counts) + _phase_count(opponent_counts)

	return np.array(
		[
			(own_material - opponent_material) / MAX_SIDE_MATERIAL,
			own_material / MAX_SIDE_MATERIAL,
			opponent_material / MAX_SIDE_MATERIAL,
			total_phase / MAX_TOTAL_PHASE,
			_piece_count(own_counts) / 16.0,
			_piece_count(opponent_counts) / 16.0,
			own_counts["p"] / 8.0,
			opponent_counts["p"] / 8.0,
			(own_counts["n"] + own_counts["b"]) / 4.0,
			(opponent_counts["n"] + opponent_counts["b"]) / 4.0,
			own_counts["r"] / 2.0,
			opponent_counts["r"] / 2.0,
			own_counts["q"],
			opponent_counts["q"],
			1.0 if own_counts["b"] >= 2 else 0.0,
			1.0 if opponent_counts["b"] >= 2 else 0.0,
			len(own_attacks) / 64.0,
			len(opponent_attacks) / 64.0,
		],
		dtype=np.float32,
	)


def _empty_piece_counts():
	return {piece_type: 0 for piece_type in RELATIVE_PIECE_PLANES}


def _material_count(counts):
	return sum(PIECE_VALUES[piece_type] * count for piece_type, count in counts.items())


def _phase_count(counts):
	return sum(PHASE_WEIGHTS[piece_type] * count for piece_type, count in counts.items())


def _piece_count(counts):
	return sum(counts.values())


def _piece_attack_squares(board_symbols, y, x, symbol):
	piece_type = symbol.lower()

	if piece_type == "p":
		dy = -1 if symbol.isupper() else 1
		return [
			(attack_y, attack_x)
			for attack_y, attack_x in ((y + dy, x - 1), (y + dy, x + 1))
			if _on_board(attack_y, attack_x)
		]

	if piece_type == "n":
		return [
			(y + dy, x + dx)
			for dy, dx in (
				(-2, -1),
				(-2, 1),
				(-1, -2),
				(-1, 2),
				(1, -2),
				(1, 2),
				(2, -1),
				(2, 1),
			)
			if _on_board(y + dy, x + dx)
		]

	if piece_type == "k":
		return [
			(y + dy, x + dx)
			for dy in (-1, 0, 1)
			for dx in (-1, 0, 1)
			if (dy != 0 or dx != 0) and _on_board(y + dy, x + dx)
		]

	directions = []
	if piece_type in ("r", "q"):
		directions.extend(((-1, 0), (1, 0), (0, -1), (0, 1)))
	if piece_type in ("b", "q"):
		directions.extend(((-1, -1), (-1, 1), (1, -1), (1, 1)))

	attacks = []
	for dy, dx in directions:
		attacks.extend(_ray_attack_squares(board_symbols, y, x, dy, dx))
	return attacks


def _ray_attack_squares(board_symbols, y, x, dy, dx):
	attacks = []
	y += dy
	x += dx
	while _on_board(y, x):
		attacks.append((y, x))
		if board_symbols[y][x] is not None:
			break
		y += dy
		x += dx
	return attacks


def _on_board(y, x):
	return 0 <= y < BOARD_SIZE and 0 <= x < BOARD_SIZE


def _king_in_check(board_symbols, white):
	king = "K" if white else "k"
	king_square = None
	for y, row in enumerate(board_symbols):
		for x, symbol in enumerate(row):
			if symbol == king:
				king_square = (y, x)
				break
		if king_square is not None:
			break

	if king_square is None:
		return False

	y, x = king_square
	return (
		_attacked_by_pawn(board_symbols, y, x, white)
		or _attacked_by_knight(board_symbols, y, x, white)
		or _attacked_by_slider(board_symbols, y, x, white)
		or _attacked_by_king(board_symbols, y, x, white)
	)


def _attacked_by_pawn(board_symbols, y, x, white_king):
	enemy_pawn = "p" if white_king else "P"
	pawn_y = y - 1 if white_king else y + 1
	for pawn_x in (x - 1, x + 1):
		if _symbol_at(board_symbols, pawn_y, pawn_x) == enemy_pawn:
			return True
	return False


def _attacked_by_knight(board_symbols, y, x, white_king):
	enemy_knight = "n" if white_king else "N"
	for dy, dx in (
		(-2, -1),
		(-2, 1),
		(-1, -2),
		(-1, 2),
		(1, -2),
		(1, 2),
		(2, -1),
		(2, 1),
	):
		if _symbol_at(board_symbols, y + dy, x + dx) == enemy_knight:
			return True
	return False


def _attacked_by_slider(board_symbols, y, x, white_king):
	enemy_rook = "r" if white_king else "R"
	enemy_bishop = "b" if white_king else "B"
	enemy_queen = "q" if white_king else "Q"

	for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
		if _ray_contains(board_symbols, y, x, dy, dx, {enemy_rook, enemy_queen}):
			return True

	for dy, dx in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
		if _ray_contains(board_symbols, y, x, dy, dx, {enemy_bishop, enemy_queen}):
			return True

	return False


def _attacked_by_king(board_symbols, y, x, white_king):
	enemy_king = "k" if white_king else "K"
	for dy in (-1, 0, 1):
		for dx in (-1, 0, 1):
			if dy == 0 and dx == 0:
				continue
			if _symbol_at(board_symbols, y + dy, x + dx) == enemy_king:
				return True
	return False


def _ray_contains(board_symbols, y, x, dy, dx, attackers):
	y += dy
	x += dx
	while 0 <= y < BOARD_SIZE and 0 <= x < BOARD_SIZE:
		symbol = board_symbols[y][x]
		if symbol is not None:
			return symbol in attackers
		y += dy
		x += dx
	return False


def _symbol_at(board_symbols, y, x):
	if y < 0 or y >= BOARD_SIZE or x < 0 or x >= BOARD_SIZE:
		return None
	return board_symbols[y][x]


def _board_white_to_move(board):
	# Board.turn is 0 when white is to move, 1 when black is to move.
	return board.turn == 0


def _board_castling_rights(board):
	return {
		"K": _board_has_rook_castling_right(board, white=True, kingside=True),
		"Q": _board_has_rook_castling_right(board, white=True, kingside=False),
		"k": _board_has_rook_castling_right(board, white=False, kingside=True),
		"q": _board_has_rook_castling_right(board, white=False, kingside=False),
	}


def _board_has_rook_castling_right(board, white, kingside):
	row = 7 if white else 0
	rook_col = 7 if kingside else 0
	king = board.pieces[row][4]
	rook = board.pieces[row][rook_col]
	if king is None or rook is None:
		return False
	return (
		king.symbol == ("K" if white else "k")
		and rook.symbol == ("R" if white else "r")
		and king.white == white
		and rook.white == white
		and not king.hasMoved
		and not rook.hasMoved
	)
