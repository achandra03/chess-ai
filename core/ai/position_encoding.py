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
METADATA_SIZE = 7
PIECE_PLANE_COUNT = len(PIECE_PLANES)
ATTACK_PLANE_COUNT = len(PIECE_PLANES)
BOARD_FEATURE_SIZE = BOARD_SIZE * BOARD_SIZE * PIECE_PLANE_COUNT
PAPER_BITMAP_FEATURE_SIZE = BOARD_FEATURE_SIZE
PAPER_MAX_ABS_CENTIPAWNS = 5000
ATTACK_BOARD_FEATURE_SIZE = BOARD_SIZE * BOARD_SIZE * (
	PIECE_PLANE_COUNT + ATTACK_PLANE_COUNT
)
FEATURE_SIZE = BOARD_FEATURE_SIZE + METADATA_SIZE
ATTACK_FEATURE_SIZE = ATTACK_BOARD_FEATURE_SIZE + METADATA_SIZE
PERSPECTIVE_METADATA_SIZE = 6
PERSPECTIVE_FEATURE_SIZE = ATTACK_BOARD_FEATURE_SIZE + PERSPECTIVE_METADATA_SIZE
RELATIVE_PIECE_PLANES = {
	"p": 0,
	"r": 1,
	"n": 2,
	"b": 3,
	"q": 4,
	"k": 5,
}


def evaluation_to_pawns(evaluation, max_abs_pawns=10.0):
	"""Convert repo evaluation strings to a clipped side-to-move pawn score."""
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


def evaluation_to_paper_target(
	evaluation, max_abs_centipawns=PAPER_MAX_ABS_CENTIPAWNS
):
	"""Map a side-to-move Stockfish score to the paper's [0, 1] target range."""
	value = str(evaluation).strip()
	if not value:
		raise ValueError("empty evaluation")

	if value.startswith("#"):
		if len(value) < 2 or value[1] not in "+-":
			raise ValueError(f"invalid mate evaluation: {value}")
		centipawns = max_abs_centipawns if value[1] == "+" else -max_abs_centipawns
	else:
		centipawns = int(value)

	centipawns = np.clip(
		centipawns, -max_abs_centipawns, max_abs_centipawns
	)
	return float(
		(centipawns + max_abs_centipawns) / (2.0 * max_abs_centipawns)
	)


def paper_target_to_pawns(
	target, max_abs_centipawns=PAPER_MAX_ABS_CENTIPAWNS
):
	"""Convert the paper MLP's [0, 1] output back to side-to-move pawns."""
	centipawns = (
		float(target) * (2.0 * max_abs_centipawns) - max_abs_centipawns
	)
	return centipawns / 100.0


def encode_fen_paper_bitmap(fen):
	"""Encode a FEN as the paper's 768-value, 12-plane bitmap input."""
	fields = str(fen).strip().split()
	if len(fields) < 1:
		raise ValueError(f"invalid FEN: {fen}")

	board_symbols = _symbols_from_fen_board(fields[0])
	return _paper_bitmap_from_symbols(board_symbols)


def encode_board_paper_bitmap(board):
	"""Encode a live board with the same bitmap schema used for paper training."""
	board_symbols = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
	for row in board.pieces:
		for piece in row:
			if piece is not None:
				board_symbols[piece.y][piece.x] = piece.symbol
	return _paper_bitmap_from_symbols(board_symbols)


def encode_fen(fen, include_attack_maps=False):
	"""Encode a FEN string for either the base or attack-map evaluator."""
	fields = str(fen).strip().split()
	if len(fields) < 4:
		raise ValueError(f"invalid FEN: {fen}")

	board_part, turn, castling = fields[0], fields[1], fields[2]
	if turn not in ("w", "b"):
		raise ValueError(f"invalid FEN turn field: {turn}")

	board_symbols = _symbols_from_fen_board(board_part)
	return _encode_symbols(
		board_symbols=board_symbols,
		white_to_move=(turn == "w"),
		castling_rights={
			"K": "K" in castling,
			"Q": "Q" in castling,
			"k": "k" in castling,
			"q": "q" in castling,
		},
		include_attack_maps=include_attack_maps,
	)


def encode_board(board, include_attack_maps=False):
	"""Encode the project's live Board object with the same schema as encode_fen."""
	board_symbols = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
	for row in board.pieces:
		for piece in row:
			if piece is None:
				continue
			board_symbols[piece.y][piece.x] = piece.symbol

	return _encode_symbols(
		board_symbols=board_symbols,
		white_to_move=_board_white_to_move(board),
		castling_rights=_board_castling_rights(board),
		include_attack_maps=include_attack_maps,
	)


def encode_fen_perspective(fen, mirror_files=False):
	"""Encode a FEN with the side to move represented as the first player."""
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


def encode_board_perspective(board):
	"""Encode a live board from the side-to-move player's perspective."""
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


def _paper_bitmap_from_symbols(board_symbols):
	planes = np.zeros(
		(BOARD_SIZE, BOARD_SIZE, PIECE_PLANE_COUNT), dtype=np.float32
	)
	for fen_y, row in enumerate(board_symbols):
		paper_y = BOARD_SIZE - 1 - fen_y
		for x, symbol in enumerate(row):
			if symbol is not None:
				planes[paper_y, x, PIECE_PLANES[symbol]] = 1.0
	return planes.reshape(PAPER_BITMAP_FEATURE_SIZE)


def _encode_symbols(board_symbols, white_to_move, castling_rights, include_attack_maps):
	plane_count = PIECE_PLANE_COUNT
	feature_size = FEATURE_SIZE
	if include_attack_maps:
		plane_count += ATTACK_PLANE_COUNT
		feature_size = ATTACK_FEATURE_SIZE

	features = np.zeros(feature_size, dtype=np.float32)
	board_features = np.zeros((BOARD_SIZE, BOARD_SIZE, plane_count), dtype=np.float32)

	for y, row in enumerate(board_symbols):
		for x, symbol in enumerate(row):
			if symbol is None:
				continue
			board_features[y, x, PIECE_PLANES[symbol]] = 1.0
			if include_attack_maps:
				attack_plane = PIECE_PLANE_COUNT + PIECE_PLANES[symbol]
				for attack_y, attack_x in _piece_attack_squares(
					board_symbols, y, x, symbol
				):
					board_features[attack_y, attack_x, attack_plane] = 1.0

	board_feature_size = BOARD_SIZE * BOARD_SIZE * plane_count
	features[:board_feature_size] = board_features.reshape(-1)
	features[board_feature_size] = 1.0 if white_to_move else 0.0
	features[board_feature_size + 1] = (
		1.0 if white_to_move and _king_in_check(board_symbols, True) else 0.0
	)
	features[board_feature_size + 2] = (
		1.0 if (not white_to_move) and _king_in_check(board_symbols, False) else 0.0
	)
	features[board_feature_size + 3] = 1.0 if castling_rights.get("K", False) else 0.0
	features[board_feature_size + 4] = 1.0 if castling_rights.get("Q", False) else 0.0
	features[board_feature_size + 5] = 1.0 if castling_rights.get("k", False) else 0.0
	features[board_feature_size + 6] = 1.0 if castling_rights.get("q", False) else 0.0
	return features


def _encode_perspective_symbols(
	board_symbols, white_to_move, castling_rights, mirror_files
):
	board_features = np.zeros(
		(BOARD_SIZE, BOARD_SIZE, PIECE_PLANE_COUNT + ATTACK_PLANE_COUNT),
		dtype=np.float32,
	)

	for y, row in enumerate(board_symbols):
		for x, symbol in enumerate(row):
			if symbol is None:
				continue

			piece_is_white = symbol.isupper()
			player_offset = 0 if piece_is_white == white_to_move else 6
			piece_plane = player_offset + RELATIVE_PIECE_PLANES[symbol.lower()]
			feature_y = y if white_to_move else BOARD_SIZE - 1 - y
			feature_x = BOARD_SIZE - 1 - x if mirror_files else x
			board_features[feature_y, feature_x, piece_plane] = 1.0

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

	features = np.zeros(PERSPECTIVE_FEATURE_SIZE, dtype=np.float32)
	features[:ATTACK_BOARD_FEATURE_SIZE] = board_features.reshape(-1)
	metadata = features[ATTACK_BOARD_FEATURE_SIZE:]
	metadata[0] = 1.0 if _king_in_check(board_symbols, white_to_move) else 0.0
	metadata[1] = 1.0 if _king_in_check(board_symbols, not white_to_move) else 0.0

	if white_to_move:
		castling_keys = ("K", "Q", "k", "q")
	else:
		castling_keys = ("k", "q", "K", "Q")
	for index, key in enumerate(castling_keys, start=2):
		metadata[index] = 1.0 if castling_rights.get(key, False) else 0.0
	return features


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
	# Board.turn is initialized to 0 before white's first move and toggled after
	# every legal move, so 0 means white to move and 1 means black to move.
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
