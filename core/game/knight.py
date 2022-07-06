from piece import Piece
class Knight(Piece):


	def makeMove(self, newX, newY, board):
		#8 possible moves
		if(newX == self.x + 2 and newY == self.y + 2):
			return True
		if(newX == self.x + 1 and newY == self.y + 2):
			return True
		if(newX == self.x - 1 and newY == self.y + 2):
			return True
		if(newX == self.x - 2 and newY == self.y + 1):
			return True
		if(newX == self.x - 2 and newY == self.y - 1):
			return True
		if(newX == self.x - 1 and newY == self.y - 2):
			return True
		if(newX == self.x + 1 and newY == self.y - 2):
			return True
		if(newX == self.x + 2 and newY == self.y - 1):
			return True
		
		return False




