from piece import Piece
from rook import Rook
class King(Piece):

	def makeMove(self, newX, newY, board):
		# Kingside castle: path clear, king and corner rook unmoved.
		# Check constraints are enforced by Board.makeMove.
		if(newY == self.y and newX == self.x + 2):
			if(board[self.y][self.x].hasMoved):
				return False
			if(board[self.y][self.x + 1] is not None or board[self.y][self.x + 2] is not None):
				return False
			if(type(board[self.y][7]) is not Rook or board[self.y][7].white != board[self.y][self.x].white):
				return False
			if(board[self.y][7].hasMoved):
				return False

			return True

		if(newY == self.y and newX == self.x - 2):
			if(board[self.y][self.x].hasMoved):
				return False
			if(board[self.y][self.x - 1] is not None or board[self.y][self.x - 2] is not None or board[self.y][self.x - 3] is not None):
				return False
			if(type(board[self.y][0]) is not Rook or board[self.y][0].white != board[self.y][self.x].white):
				return False
			if(board[self.y][0].hasMoved):
				return False

			return True

		if(abs(newY - self.y) <= 1 and abs(newX - self.x) <= 1):
			return True

		return False
