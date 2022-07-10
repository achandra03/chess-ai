from piece import Piece
from rook import Rook
class King(Piece):


	def makeMove(self, newX, newY, board):
		#Check for right castle
		if(newY == self.y and newX == self.x + 2):
			#King has already moved
			if(board[self.y][self.x].hasMoved):
				return False
			#Castling path is obstructed
			if(board[self.y][self.x + 1] is not None or board[self.y][self.x + 2] is not None):
				return False
			#Corner piece isn't rook or is rook of different color
			if(type(board[self.y][7]) is not Rook or board[self.y][7].white != board[self.y][self.x].white):
				return False
			#Ensure that rook hasn't moved already
			if(board[self.y][7].hasMoved):
				return False

			return True

		#Check for left castle
		if(newY == self.y and newX == self.x - 2):
			#King has already moved
			if(board[self.y][self.x].hasMoved):
				return False
			#Castling path is obstructed
			if(board[self.y][self.x - 1] is not None or board[self.y][self.x - 2] is not None or board[self.y][self.x - 3] is not None):
				return False
			#Corner piece isn't rook or is rook of different color
			if(type(board[self.y][0]) is not Rook or board[self.y][0].white != board[self.y][self.x].white):
				return False
			#Ensure that rook hasn't moved already
			if(board[self.y][0].hasMoved):
				return False
		
			return True

		#Regular king move otherwise
		if(abs(newY - self.y) <= 1 and abs(newX - self.x) <= 1):
			return True

		return False

