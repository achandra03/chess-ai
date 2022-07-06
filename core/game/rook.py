from piece import Piece
class Rook(Piece):

	def makeMove(self, newX, newY, board):
		#Can only move in 1 direction - if both x and y are different move is illegal
		if(newX == self.x or newY == self.y):
			return True
		return False
