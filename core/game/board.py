from piece import Piece
from king import King
from queen import Queen
from bishop import Bishop
from knight import Knight
from rook import Rook
from pawn import Pawn
import copy

class Board:

	def __init__(self):
		self.pieces = []
		for i in range(0, 8):
			curr = [None for j in range(0, 8)]
			self.pieces.append(curr)

		self.pieces[7][0] = Rook(0, 7, True, 'whiterook.png', 'R')
		self.pieces[7][1] = Knight(1, 7, True, 'whiteknight.png', 'N')
		self.pieces[7][2] = Bishop(2, 7, True, 'whitebishop.png', 'B')
		self.pieces[7][3] = Queen(3, 7, True, 'whitequeen.png', 'Q')
		self.pieces[7][4] = King(4, 7, True, 'whiteking.png', 'K')
		self.pieces[7][5] = Bishop(5, 7, True, 'whitebishop.png', 'B')
		self.pieces[7][6] = Knight(6, 7, True, 'whiteknight.png', 'N')
		self.pieces[7][7] = Rook(7, 7, True, 'whiterook.png', 'R')

		for i in range(0, 8):
			self.pieces[6][i] = Pawn(i, 6, True, 'whitepawn.png', 'P')

		self.pieces[0][0] = Rook(0, 0, False, 'blackrook.png', 'r')
		self.pieces[0][1] = Knight(1, 0, False, 'blackknight.png', 'n')
		self.pieces[0][2] = Bishop(2, 0, False, 'blackbishop.png', 'b')
		self.pieces[0][3] = Queen(3, 0, False, 'blackqueen.png', 'q')
		self.pieces[0][4] = King(4, 0, False, 'blackking.png', 'k')
		self.pieces[0][5] = Bishop(5, 0, False, 'blackbishop.png', 'b')
		self.pieces[0][6] = Knight(6, 0, False, 'blackknight.png', 'n')
		self.pieces[0][7] = Rook(7, 0, False, 'blackrook.png', 'r')

		for i in range(0, 8):
			self.pieces[1][i] = Pawn(i, 1, False, 'blackpawn.png', 'p')



	def makeMove(self, x, y, newX, newY):
		p = self.pieces[y][x]

		#Empty square selected
		if(p is None):
			return False

		#Attempting to move to a square occupied by friendly piece
		if(self.pieces[newY][newX] is not None and self.pieces[newY][newX].white == p.white):
			return False

		#Piece cannot move in the manner described
		if(not self.pieces[y][x].makeMove(newX, newY, self.pieces)):
			return False

		#Path must be clear (barring knights)
		if(type(p) is not Knight and self.pathBlocked(x, y, newX, newY)):
			return False

		#Taking snapshot of board in case reversion is needed
		snapshot = copy.deepcopy(self.pieces)

		#Right castle - right corner rook needs to be updated
		if(type(p) is King and newX == x + 2):
			rook = self.pieces[y][7]
			rook.x -= 2
			rook.hasMoved = True

		#Left castle - left corner rook needs to be updated
		if(type(p) is King and newX == x - 2):
			rook = self.pieces[y][0]
			rook.x += 3
			rook.hasMoved = True

		#Update piece being moved and board in general
		p.hasMoved = True
		p.x = newX
		p.y = newY

        #If capturing, remove old piece from spot
		self.pieces[newY][newX] = None
        

        #Handle promotion
		if(type(p) is Pawn):
			if(p.white and p.y == 0):
				self.pieces[p.y][p.x] = Queen(p.x, p.y, p.white, 'whitequeen.png', 'Q')
			elif(not p.white and p.y == 7):
				self.pieces[p.y][p.x] = Queen(p.x, p.y, p.white, 'blackqueen.png', 'q')

		#Revert back to old state if king is in check
		if(self.checked(p.white)):
			self.pieces = snapshot
			return False
	
		#Move is valid
		self.update()
		return True



	
	def pathBlocked(self, x, y, newX, newY):
		dy = 0
		dx = 0
		if(newX < x):
			dx = -1
		if(newX > x):
			dx = 1
		if(newY < y):
			dy = -1
		if(newY > y):
			dy = 1

		x += dx
		y += dy

		while(x != newX or y != newY):
			if(self.pieces[y][x] is not None):
				return True
			x += dx
			y += dy

		return False


	def update(self):
		updated = []
		for i in range(0, 8):
			curr = [None for j in range(0, 8)]
			updated.append(curr)

		for row in self.pieces:
			for piece in row:
				if(piece is None):
					continue
				updated[piece.y][piece.x] = piece

		self.pieces = updated


	def checked(self, white):
		#Get location of king
		king = None
		for row in self.pieces:
			for piece in row:
				if(piece is None):
					continue
				if(type(piece) is King and piece.white == white):
					king = piece
		
		#Check for queens, rooks, and bishops
		for dx in range(-1, 2):
			for dy in range(-1, 2):
				if(dx == 0 and dy == 0):
					continue
				if(self.moveAlongPath(king.x, king.y, dx, dy, king.white)):
					return True

		#Check for knights
		if(king.y < 7 and king.x < 6 and self.pieces[king.y + 1][king.x + 2] is not None and type(self.pieces[king.y + 1][king.x + 2]) is Knight and self.pieces[king.y + 1][king.x + 2].white != white):
			return True
		if(king.y < 6 and king.x < 7 and self.pieces[king.y + 2][king.x + 1] is not None and type(self.pieces[king.y + 2][king.x + 1]) is Knight and self.pieces[king.y + 2][king.x + 1].white != white):
			return True
		if(king.y < 6 and king.y > 0 and self.pieces[king.y + 2][king.x - 1] is not None and type(self.pieces[king.y + 2][king.x - 1]) is Knight and self.pieces[king.y + 2][king.x - 1].white != white):
			return True
		if(king.y < 7 and king.x > 1 and self.pieces[king.y + 1][king.x - 2] is not None and type(self.pieces[king.y + 1][king.x - 2]) is Knight and self.pieces[king.y + 1][king.x - 2].white != white):
			return True
		if(king.y > 0 and king.x > 1 and self.pieces[king.y - 1][king.x - 2] is not None and type(self.pieces[king.y - 1][king.x - 2]) is Knight and self.pieces[king.y - 1][king.x - 2].white != white):
			return True
		if(king.y > 1 and king.x > 0 and self.pieces[king.y - 2][king.x - 1] is not None and type(self.pieces[king.y - 2][king.x - 1]) is Knight and self.pieces[king.y - 2][king.x - 1].white != white):
			return True
		if(king.y > 1 and king.x < 7 and self.pieces[king.y - 2][king.x + 1] is not None and type(self.pieces[king.y - 2][king.x + 1]) is Knight and self.pieces[king.y - 2][king.x + 1].white != white):
			return True
		if(king.y > 0 and king.x < 6 and self.pieces[king.y - 1][king.x + 2] is not None and type(self.pieces[king.y - 1][king.x + 2]) is Knight and self.pieces[king.y - 1][king.x + 2].white != white):
			return True
		
		#Check for pawns
		if(white):
			if(king.y - 1 > -1 and king.x - 1 > -1):
				if(self.pieces[king.y - 1][king.x - 1] is not None and type(self.pieces[king.y - 1][king.x - 1]) is Pawn and self.pieces[king.y - 1][king.x - 1].white != white):
					return True
			if(king.y - 1 > -1 and king.x + 1 < 8):
				if(self.pieces[king.y - 1][king.x + 1] is not None and type(self.pieces[king.y - 1][king.x - 1]) is Pawn and self.pieces[king.y - 1][king.x + 1].white != white):
					return True

		else:
			if(king.y + 1 < 8 and king.x - 1 > -1):
				if(self.pieces[king.y + 1][king.x - 1] is not None and type(self.pieces[king.y + 1][king.x - 1]) is Pawn and self.pieces[king.y + 1][king.x - 1].white != white):
					return True
			if(king.y + 1 < 8 and king.x + 1 < 8):
				if(self.pieces[king.y + 1][king.x + 1] is not None and type(self.pieces[king.y + 1][king.x + 1]) is Pawn and self.pieces[king.y + 1][king.x + 1].white != white):
					return True


		#Check for enemy king
		for dx in range(-1, 2):
			for dy in range(-1, 2):
				if(dx == 0 and dy == 0):
					continue
				if(king.y + dy < 0 or king.y + dy > 7 or king.x + dx < 0 or king.x + dx > 7):
					continue
				if(self.pieces[king.y + dy][king.x + dx] is None):
					continue
				if(self.pieces[king.y + dy][king.x + dx].white == white):
					continue
				if(type(self.pieces[king.y + dy][king.x + dx]) is King):
					return True

		return False




	def moveAlongPath(self, x, y, dx, dy, white):
		x += dx
		y += dy
		while(x > -1 and x < 8 and y > -1 and y < 8):
			#If either dx or dy is 0, we are looking for queens/rooks
			if(dx == 0 or dy == 0):
				if(self.pieces[y][x] is None):
					x += dx
					y += dy
					continue
				if((self.pieces[y][x].white != white) and (type(self.pieces[y][x]) is Queen or type(self.pieces[y][x]) is Rook)):
					return True
				break

			#Otherwise we are looking for queens/bishops
			else:
				if(self.pieces[y][x] is None):
					x += dx
					y += dy
					continue
				if((self.pieces[y][x].white != white) and (type(self.pieces[y][x]) is Queen or type(self.pieces[y][x]) is Bishop)):
					return True
				break

		return False

	def listPieces(self, w):
		for i in range(0, 8):
			for j in range(0, 8):
				if(self.pieces[i][j] is None or self.pieces[i][j].white != w):
					continue
				y = self.pieces[i][j].y
				x = self.pieces[i][j].x
				print(type(self.pieces[y][x]), 'on', y, x)	


	def allMoves(self, w):
		#Returns a list of all possible moves in the form (y, x, newY, newX)
		res = []
		for i in range(0, 8):
			for j in range(0, 8):
				if(self.pieces[i][j] is None or self.pieces[i][j].white != w):
					continue
				x = self.pieces[i][j].x
				y = self.pieces[i][j].y
				for a in range(0, 8):
					for b in range(0, 8):
						if(x == b and y == a):
							continue
						snapshot = copy.deepcopy(self.pieces)
						if(self.makeMove(x, y, b, a)):
							curr = [y, x, a, b]
							res.append(curr)
							self.pieces = snapshot
		return res
