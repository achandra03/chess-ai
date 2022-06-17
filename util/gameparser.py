import chess
import chess.pgn
import sys
import numpy as np
import pandas as pd
from stockfish import Stockfish

class GameParser:

	def __init__(self):

		self.WHITE_KING = 0
		self.WHITE_QUEEN = 1
		self.WHITE_ROOK = 2
		self.WHITE_BISHOP = 3
		self.WHITE_KNIGHT = 4
		self.WHITE_PAWN = 5

		self.BLACK_KING = 6
		self.BLACK_QUEEN = 7
		self.BLACK_ROOK = 8
		self.BLACK_BISHOP = 9
		self.BLACK_KNIGHT = 10
		self.BLACK_PAWN = 11

		self.PIECE_TO_POSITION = {'K': self.WHITE_KING, 'Q': self.WHITE_QUEEN, 'R': self.WHITE_ROOK, 'B': self.WHITE_BISHOP, 'N': self.WHITE_KNIGHT, 'P': self.WHITE_PAWN, 'k': self.BLACK_KING, 'q': self.BLACK_QUEEN, 'r': self.BLACK_ROOK, 'b': self.BLACK_BISHOP, 'n': self.BLACK_KNIGHT, 'p': self.BLACK_PAWN}


		self.stockfish = Stockfish(path='/Users/arnavchandra/Desktop/chess/stockfish/stockfish-exe')

		self.imboards = []
		self.evals = []


	def parse_file(self, filename):
		NUMBER_OF_POSITIONS = 10000
		with open(filename) as f:
			game = chess.pgn.read_game(f)
			while(game is not None):
				self.process(game)
				print(len(self.imboards))
				if(len(self.imboards) >= NUMBER_OF_POSITIONS):
					break
				game = chess.pgn.read_game(f)

		
		d = {'Positions': self.imboards, 'Evaluations': self.evals}
		df = pd.DataFrame(data=d)
		df.to_csv('out.csv', index=False)


	def process(self, game):
		white_move = True
		while(game is not None):
			board = game.board()
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

				if((piece.color is chess.WHITE and white_move) or (piece.color is chess.BLACK and not white_move)):
					imboard[y][x][self.PIECE_TO_POSITION[piece.symbol()]] = 1
				else:
					imboard[y][x][self.PIECE_TO_POSITION[piece.symbol()]] = -1


			imboard = np.array(imboard)
			imboard = imboard.flatten()
			self.imboards.append(imboard)

			fen_rep = board.fen()
			self.stockfish.set_fen_position(fen_rep)
			eva = self.stockfish.get_evaluation()
			if(eva['type'] == 'cp'):
				self.evals.append(self.sigmoid(eva['value'] / 100))
			else:
				if(eva['value'] < 0):
					self.evals.append(0)
				else:
					self.evals.append(1)
			
			game = game.next()
			white_move = not white_move


	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))


g = GameParser()
g.parse_file('/Users/arnavchandra/Desktop/chess/data/ficsgamesdb_2021_chess_nomovetimes_252123.pgn')
