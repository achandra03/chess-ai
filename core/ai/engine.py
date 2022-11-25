import sys
import os
sys.path.append(os.path.abspath('../game'))
from board import Board
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers
import numpy as np
import copy

class Engine:
	

	def __init__(self, board):
		self.encoder = keras.models.load_model('encoder_model')
		self.func = K.function([self.encoder.get_layer(index=0).input], self.encoder.get_layer(index=2).output)
		self.board = board
		self.nn = tf.keras.models.load_model("my_model_v2")
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

	def inv_sigmoid(self, x):
		return np.log(x / (1 - x))

	def nn_input(self, white):
		imboard = []
		for i in range(8):
			curr = []
			for j in range(8):
				c = [0 for i in range(12)]
				curr.append(c)
			imboard.append(curr)
				

		for row in self.board.pieces:
			for piece in row:
				if(piece is None):
					continue

				if(piece.white != white):
					imboard[piece.y][piece.x][self.PIECE_TO_POSITION[piece.symbol]] = -1
				else:
					imboard[piece.y][piece.x][self.PIECE_TO_POSITION[piece.symbol]] = 1


		imboard = np.array(imboard)
		imboard = imboard.flatten()
		imboard = np.append(imboard, [0 for i in range(7)])

		imboard[768] = int(self.board.turn)
		if(self.board.checked(True) and self.board.turn == 1):
			imboard[769] = 1
		elif(self.board.checked(False) and self.board.turn == 0):
				imboard[770] = 1
		imboard[771] = int(self.board.has_kingside_castling_rights(1))
		imboard[772] = int(self.board.has_queenside_castling_rights(1))
		imboard[773] = int(self.board.has_kingside_castling_rights(0))
		imboard[774] = int(self.board.has_queenside_castling_rights(0))


		imboard = imboard.reshape((-1, 775))
		return self.func([imboard])


	def eval_position(self, white):
		inp = self.nn_input(white)
		eva = self.nn(inp)
		return eva


	def minimax(self, depth, maximize, alpha, beta):
		if(depth == 0):
			return self.eval_position(maximize), None

		moves = self.board.allMoves(maximize)
		if(maximize):
			best = -100000000
			ret = None
			for move in moves:
				y = move[0]
				x = move[1]
				newY = move[2]
				newX = move[3]
				snapshot = copy.deepcopy(self.board.pieces)
				self.board.makeMove(x, y, newX, newY)
				val, _ = self.minimax(depth - 1, False, alpha, beta)
				self.board.pieces = snapshot
				if(val > best):
					best = val
					ret = move
				alpha = max(alpha, best)
				if(beta <= alpha):
					break

			return best, ret

		else:
			best = 100000000
			ret = None
			for move in moves:
				y = move[0]
				x = move[1]
				newY = move[2]
				newX = move[3]
				snapshot = copy.deepcopy(self.board.pieces)
				self.board.makeMove(x, y, newX, newY)
				val, _ = self.minimax(depth - 1, True, alpha, beta)
				self.board.pieces = snapshot
				if(val < best):
					best = val
					ret = move
				beta = min(beta, best)
				if(beta <= alpha):
					break
			return best, ret

	def selectMove(self):
		moves = self.board.allMoves(False)
		bestEval = 1000
		bestMove = None
		for move in moves:
			y = move[0]
			x = move[1]
			newY = move[2]
			newX = move[3]
			snapshot = copy.deepcopy(self.board.pieces)
			self.board.makeMove(x, y, newX, newY)
			newEval = self.eval_position(False)
			if(newEval < bestEval):
				bestEval = newEval
				bestMove = move
			self.board.pieces = snapshot
		print(bestEval)
		return bestMove

