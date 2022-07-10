from abc import ABC, abstractmethod
class Piece(ABC):

	def __init__(self, x, y, white, filename, symbol):
		self.x = x
		self.y = y
		self.white = white
		self.hasMoved = False
		self.filename = filename
		self.symbol = symbol


	@abstractmethod
	def makeMove(self, newX, newY, board):
		pass

