import chess
import chess.pgn
import sys
import numpy as np
import pandas as pd
from stockfish import Stockfish

class GameParser:

	def __init__(self):

		self.WHITE_PAWN = 0
		self.WHITE_ROOK = 1
		self.WHITE_KNIGHT = 2
		self.WHITE_BISHOP = 3
		self.WHITE_QUEEN = 4
		self.WHITE_KING = 5

		self.BLACK_PAWN = 6
		self.BLACK_ROOK = 7
		self.BLACK_KNIGHT = 8
		self.BLACK_BISHOP = 9
		self.BLACK_QUEEN = 10
		self.BLACK_KING = 11

		self.PIECE_TO_POSITION = {'K': self.WHITE_KING, 'Q': self.WHITE_QUEEN, 'R': self.WHITE_ROOK, 'B': self.WHITE_BISHOP, 'N': self.WHITE_KNIGHT, 'P': self.WHITE_PAWN, 'k': self.BLACK_KING, 'q': self.BLACK_QUEEN, 'r': self.BLACK_ROOK, 'b': self.BLACK_BISHOP, 'n': self.BLACK_KNIGHT, 'p': self.BLACK_PAWN}



		self.imboards = []
		self.evals = []
		self.seen = set()



	def parse_file(self, filename):
		reset_position = 100000
		positions = 0
		outname = 1
		df = pd.read_csv(filename)
		fens = df['FEN']
		evals = df['Evaluation']
		max_positions = 16000000
		for (position, evaluation) in zip(fens, evals):
			b = chess.Board()
			b.set_fen(position)
			self.process(b)
			self.evals.append(self.process_evaluation(evaluation))
			positions += 1
			max_positions -= 1
			print(max_positions)
			if(max_positions == 0):
				break

			
			if(positions >= reset_position):
				d = {'Positions': self.imboards, 'Evaluations': self.evals}
				df = pd.DataFrame(data=d)
				df.to_csv(str(outname) + '.csv', index=False)
				outname += 1
				self.imboards = []
				self.evals = []
				positions = 0
			
		
		d = {'Positions': self.imboards, 'Evaluations': self.evals}
		df = pd.DataFrame(data=d)
		df.to_csv('out.csv', index=False)


	def process(self, board):
		imboard = []
		for i in range(8):
			curr = []
			for j in range(8):
				c = [0 for i in range(12)]
				curr.append(c)
			imboard.append(curr)

				

		squares = board.piece_map()
		for square in squares:
			x = chess.parse_square(chess.square_name(square)) % 8
			y = chess.parse_square(chess.square_name(square)) // 8

			piece = squares[square]

			imboard[y][x][self.PIECE_TO_POSITION[piece.symbol()]] = 1


		imboard = np.array(imboard, dtype='object')
		imboard = imboard.flatten()
		imboard = np.append(imboard, [0 for i in range(7)])


		imboard[768] = board.turn
		if(board.is_check()):
			if(board.turn):
				imboard[769] = 1
			else:
				imboard[770] = 1
		imboard[771] = int(board.has_kingside_castling_rights(1))
		imboard[772] = int(board.has_queenside_castling_rights(1))
		imboard[773] = int(board.has_kingside_castling_rights(0))
		imboard[774] = int(board.has_queenside_castling_rights(0))

		self.imboards.append(imboard)


	def process_evaluation(self, evaluation):
		x = 0
		ma = 5000
		mi = -5000
		if(evaluation[0] == '#'):
			if(evaluation[1] == '+'):
				x = ma
			else:
				x = mi
		elif(evaluation[0] == '+'):
			x = int(evaluation[1:])
		elif(evaluation[0] == '-'):
			x = -1 * int(evaluation[1:])

		x = max(x, mi)
		x = min(x, ma)

		y = (2 * (x - mi) / (ma - mi)) - 1
		return y

g = GameParser()
g.parse_file('/Users/arnavchandra/Desktop/chess/data/chessData.csv')
