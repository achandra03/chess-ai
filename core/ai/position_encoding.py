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
FEATURE_SIZE = BOARD_SIZE * BOARD_SIZE * len(PIECE_PLANES) + 7


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


def encode_fen(fen):
	"""Encode a FEN string into the 775-feature input used by the evaluator."""
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
	)


def encode_board(board):
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


def _encode_symbols(board_symbols, white_to_move, castling_rights):
	features = np.zeros(FEATURE_SIZE, dtype=np.float32)
	piece_features = np.zeros((BOARD_SIZE, BOARD_SIZE, len(PIECE_PLANES)), dtype=np.float32)

	for y, row in enumerate(board_symbols):
		for x, symbol in enumerate(row):
			if symbol is None:
				continue
			piece_features[y, x, PIECE_PLANES[symbol]] = 1.0

	features[:768] = piece_features.reshape(-1)
	features[768] = 1.0 if white_to_move else 0.0
	features[769] = 1.0 if white_to_move and _king_in_check(board_symbols, True) else 0.0
	features[770] = 1.0 if (not white_to_move) and _king_in_check(board_symbols, False) else 0.0
	features[771] = 1.0 if castling_rights.get("K", False) else 0.0
	features[772] = 1.0 if castling_rights.get("Q", False) else 0.0
	features[773] = 1.0 if castling_rights.get("k", False) else 0.0
	features[774] = 1.0 if castling_rights.get("q", False) else 0.0
	return features


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
