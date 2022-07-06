from piece import Piece
class Bishop(Piece):

	def makeMove(self, newX, newY, board):
		#Diagonal move - magnitude of changes in x and y direction should equal each other
		if(abs(newX - self.x) == abs(newY - self.y)):
			return True
		return False

