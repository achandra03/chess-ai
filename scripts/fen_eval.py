import chess
import chess.pgn
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers
import numpy as np


WHITE_PAWN = 0
WHITE_ROOK = 1
WHITE_KNIGHT = 2
WHITE_BISHOP = 3
WHITE_QUEEN = 4
WHITE_KING = 5

BLACK_PAWN = 6
BLACK_ROOK = 7
BLACK_KNIGHT = 8
BLACK_BISHOP = 9
BLACK_QUEEN = 10
BLACK_KING = 11


PIECE_TO_POSITION = {'K': WHITE_KING, 'Q': WHITE_QUEEN, 'R': WHITE_ROOK, 'B': WHITE_BISHOP, 'N': WHITE_KNIGHT, 'P': WHITE_PAWN, 'k': BLACK_KING, 'q': BLACK_QUEEN, 'r': BLACK_ROOK, 'b': BLACK_BISHOP, 'n': BLACK_KNIGHT, 'p': BLACK_PAWN}

VALUE_MAP = {'K': 15, 'Q': 9, 'R': 5, 'B': 3, 'N': 3, 'P': 1, 'k': -15, 'q': -9, 'r': -5, 'b': -3, 'n': -3, 'p': -1}

def process(board):
	imboard = [[0 for _ in range(8)] for _ in range(8)]
	squares = board.piece_map()
	for square in squares:
		x = chess.parse_square(chess.square_name(square)) % 8
		y = chess.parse_square(chess.square_name(square)) // 8
		
		piece = squares[square]
		val = VALUE_MAP[piece.symbol()]
		imboard[y][x] = val

	imboard = np.array(imboard, dtype='object')
	imboard = imboard.flatten()
	turn = (1 if board.turn == chess.WHITE else -1)
	imboard = np.append(imboard, [turn for _ in range(1)])
	return imboard


fen_string = "r1b3k1/pp1nb1pp/1q2p3/3pP3/3n4/P2B1P2/1PQBN2P/R3K2R w KQ - 0 16"
board = chess.Board()
board.set_fen(fen=fen_string)

tmp = process(board)

q = []
for x in tmp:
	q.append(int(x))

q = np.array(q, dtype='object')
q = q.reshape((1, 1, 65))
inp = np.array(q).astype('float32')
nn = keras.models.load_model("model")
res = nn(inp)
eva = tf.get_static_value(res)[0][0][0]

ma = 10
mi = -10
print(eva)
print(eva * (ma - mi) + mi)
