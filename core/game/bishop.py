from piece import Piece
class Bishop(Piece):

	def makeMove(self, newX, newY, board):
		if(abs(newX - self.x) == abs(newY - self.y)):
			return True
		return False
