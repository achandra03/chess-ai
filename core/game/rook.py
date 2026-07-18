from piece import Piece
class Rook(Piece):

	def makeMove(self, newX, newY, board):
		if(newX == self.x or newY == self.y):
			return True
		return False
