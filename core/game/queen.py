from piece import Piece
class Queen(Piece):

	def makeMove(self, newX, newY, board):
		if(abs(newX - self.x) == abs(newY - self.y)):
			return True

		if(newX == self.x or newY == self.y):
			return True

		return False
