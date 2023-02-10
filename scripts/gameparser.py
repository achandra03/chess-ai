import chess
import chess.pgn
import chess.engine
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

		self.BLACK_PAWN = 0
		self.BLACK_ROOK = 1
		self.BLACK_KNIGHT = 2
		self.BLACK_BISHOP = 3
		self.BLACK_QUEEN = 4
		self.BLACK_KING = 5

		self.PIECE_TO_POSITION = {'K': self.WHITE_KING, 'Q': self.WHITE_QUEEN, 'R': self.WHITE_ROOK, 'B': self.WHITE_BISHOP, 'N': self.WHITE_KNIGHT, 'P': self.WHITE_PAWN, 'k': self.BLACK_KING, 'q': self.BLACK_QUEEN, 'r': self.BLACK_ROOK, 'b': self.BLACK_BISHOP, 'n': self.BLACK_KNIGHT, 'p': self.BLACK_PAWN}

		self.VALUE_MAP = {'K': 15, 'Q': 9, 'R': 5, 'B': 3, 'N': 3, 'P': 1, 'k': -15, 'q': -9, 'r': -5, 'b': -3, 'n': -3, 'p': -1}




		self.imboards = []
		self.evals = []
		self.seen = set()
		self.stockfish = Stockfish(path="stockfish/stockfish-exe", depth=1)



	def parse_file(self, filename):
		positions = 0
		outname = 1
		df = pd.read_csv(filename)
		fens = df['FEN']
		evals = df['Evaluation']
		reset_position = 100000
		max_positions = 10000000
		for (position, evaluation) in zip(fens, evals):
			b = chess.Board()
			b.set_fen(position)
			self.process2(b)
			#self.evals.append(self.process_evaluation(evaluation))
			self.stockfish.set_fen_position(position)
			self.evals.append(self.process_stockfish())
			positions += 1
			max_positions -= 1
			print(max_positions)
			break
			if(max_positions == 0):
				break
			
			'''	
			if(positions >= reset_position):
				d = {'Positions': self.imboards, 'Evaluations': self.evals}
				df = pd.DataFrame(data=d)
				df.to_csv(str(outname) + '.csv', index=False)
				outname += 1
				self.imboards = []
				self.evals = []
				positions = 0
			'''	
		
		d = {'Positions': self.imboards, 'Evaluations': self.evals}
		df = pd.DataFrame(data=d)
		df.to_csv('out.csv', index=False)

	def process(self, board):
		imboard = [[0 for _ in range(8)] for _ in range(8)]
		squares = board.piece_map()
		for square in squares:
			x = chess.parse_square(chess.square_name(square)) % 8
			y = chess.parse_square(chess.square_name(square)) // 8
			
			piece = squares[square]
			val = self.VALUE_MAP[piece.symbol()]
			imboard[y][x] = val

		imboard = np.array(imboard, dtype='object')
		imboard = imboard.flatten()
		turn = (1 if board.turn == chess.WHITE else -1)
		imboard = np.append(imboard, [turn for _ in range(1)])
		self.imboards.append(imboard)


	def process2(self, board):
		imboard = []
		for i in range(8):
			curr = []
			for j in range(8):
				c = [0 for i in range(6)]
				curr.append(c)
			imboard.append(curr)

				

		squares = board.piece_map()
		for square in squares:
			x = chess.parse_square(chess.square_name(square)) % 8
			y = chess.parse_square(chess.square_name(square)) // 8

			piece = squares[square]
			
			res = 1
			if(piece.color == chess.BLACK):
				res = -1
			imboard[y][x][self.PIECE_TO_POSITION[piece.symbol()]] = res


			
		
		imboard = np.append(imboard, [board.turn for i in range(1)])
		'''
		imboard[768] = int(board.turn)
		if(board.is_check()):
			if(board.turn):
				imboard[769] = 1
			else:
				imboard[770] = 1
		imboard[771] = int(board.has_kingside_castling_rights(1))
		imboard[772] = int(board.has_queenside_castling_rights(1))
		imboard[773] = int(board.has_kingside_castling_rights(0))
		imboard[774] = int(board.has_queenside_castling_rights(0))
		'''
		self.imboards.append(imboard)


	def process_evaluation(self, evaluation):
		x = 0
		ma = 10
		mi = -10
		if(evaluation[0] == '#'):
			if(evaluation[1] == '+'):
				x = ma
			else:
				x = mi
		elif(evaluation[0] == '+'):
			x = int(evaluation[1:])
			x = x / 100
			x = min(x, ma)
		elif(evaluation[0] == '-'):
			x = -1 * int(evaluation[1:])
			x = x / 100
			x = max(x, mi)
		x = (x - mi) / (ma - mi)
		return x

	def process_stockfish(self):
		mi = -10
		ma = 10
		x = 0

		evaluation = self.stockfish.get_evaluation()
		typ = evaluation['type']
		val = evaluation['value']
		if(typ == 'mate'):
			if(val < 0):
				x = mi
			else:
				x = ma
		else:
			x = val / 100
			x = max(x, mi)
			x = min(x, ma)

		x = (x - mi) / (ma - mi)
		return x



g = GameParser()
g.parse_file('/Users/arnavchandra/Desktop/chess/data/chessData.csv')
