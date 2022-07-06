from piece import Piece
class Queen(Piece):

	def makeMove(self, newX, newY, board):
		#Bishop moves
		if(abs(newX - self.x) == abs(newY - self.y)):
			return True

		#Rook moves
		if(newX == self.x or newY == self.y):
			return True

		return False

