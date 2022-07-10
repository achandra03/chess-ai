from piece import Piece
class Pawn(Piece):

	def makeMove(self, newX, newY, board):


		#Pawns can never move backwards
		if(self.white and newY > self.y):
			return False

		if(not self.white and newY < self.y):
			return False

		#2 moves ahead only valid if pawn hasn't moved already and no pieces on square
		if(newX == self.x and abs(newY - self.y) == 2):
			if(self.hasMoved):
				return False
			if(board[newY][newX] is not None):
				return False
			return True
		
		#1 move ahead valid if no pieces on square
		if(newX == self.x and abs(newY - self.y) == 1):
			if(board[newY][newX] is not None):
				return False
			return True

		#Diagonal valid if capturing
		if(abs(self.x - newX) == 1 and abs(self.y - newY) == 1):
			if(board[newY][newX] is None or board[newY][newX].white == board[self.y][self.x].white):
				return False
			return True

		return False

		#TODO: En passant
