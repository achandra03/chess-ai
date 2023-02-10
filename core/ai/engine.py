import sys
import os
sys.path.append(os.path.abspath('../game'))
from board import Board
import numpy as np
import copy
from xgboost import XGBRegressor



class Engine:
	

	def __init__(self, board):

		self.board = board
		
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

		self.VALUE_MAP = {'K': 15, 'Q': 9, 'R': 5, 'B': 3, 'N': 3, 'P': 1, 'k': -15, 'q': -9, 'r': -5, 'b': -3, 'n': -3, 'p': -1}

		self.bst = XGBRegressor(n_estimators=10000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8) 
		self.bst.load_model('boost.json')



	
	def nn_input(self):
		imboard = [[0 for _ in range(8)] for _ in range(8)]
		
		for row in self.board.pieces:
			for piece in row:
				if(piece is None):
					continue
				
				imboard[piece.y][piece.x] = self.VALUE_MAP[piece.symbol]
			
		imboard = np.array(imboard, dtype='object')
		imboard = imboard.flatten()
		imboard = np.append(imboard, [self.board.turn for _ in range(1)])
		return imboard

	def eval_position(self):
		inp = self.nn_input()
		inp = np.array(inp).astype('float32')
		inp = inp.reshape((1, 65))
		eva = self.bst.predict(inp)[0] * 10
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
			newEval = self.eval_position()
			if(newEval < bestEval):
				bestEval = newEval
				bestMove = move
			self.board.pieces = snapshot
		print(bestEval)
		return bestMove

