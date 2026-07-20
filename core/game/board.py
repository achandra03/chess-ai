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

		self.turn = 0

	def has_queenside_castling_rights(self, side):
		# side 0 = black, side 1 = white.
		if(side == 0):
			if(self.pieces[0][4] is not None and self.pieces[0][4].filename == 'blackking.png' and not self.pieces[0][4].hasMoved):
				if(self.pieces[0][0] is not None and self.pieces[0][0].filename == 'blackrook.png' and not self.pieces[0][0].hasMoved):
					return 1
			return 0

		else:
			if(self.pieces[7][4] is not None and self.pieces[7][4].filename == 'whiteking.png' and not self.pieces[7][4].hasMoved):
				if(self.pieces[7][0] is not None and self.pieces[7][0].filename == 'whiterook.png' and not self.pieces[7][0].hasMoved):
					return 1
			return 0


	def has_kingside_castling_rights(self, side):
		if(side == 0):
			if(self.pieces[0][4] is not None and self.pieces[0][4].filename == 'blackking.png' and not self.pieces[0][4].hasMoved):
				if(self.pieces[0][7] is not None and self.pieces[0][7].filename == 'blackrook.png' and not self.pieces[0][7].hasMoved):
					return 1
			return 0

		else:
			if(self.pieces[7][4] is not None and self.pieces[7][4].filename == 'whiteking.png' and not self.pieces[7][4].hasMoved):
				if(self.pieces[7][7] is not None and self.pieces[7][7].filename == 'whiterook.png' and not self.pieces[7][7].hasMoved):
					return 1
			return 0


	def makeMove(self, x, y, newX, newY):
		p = self.pieces[y][x]

		if(p is None):
			return False

		if(self.pieces[newY][newX] is not None and self.pieces[newY][newX].white == p.white):
			return False

		if(not self.pieces[y][x].makeMove(newX, newY, self.pieces)):
			return False

		if(type(p) is not Knight and self.pathBlocked(x, y, newX, newY)):
			return False

		# A castling king may not start in check or pass through an
		# attacked square.
		if(type(p) is King and abs(newX - x) == 2):
			if(self.checked(p.white)):
				return False
			p.x = x + 1 if newX > x else x - 1
			self.update()
			passesThroughCheck = self.checked(p.white)
			p.x = x
			self.update()
			if(passesThroughCheck):
				return False

		snapshot = copy.deepcopy(self.pieces)

		if(type(p) is King and newX == x + 2):
			rook = self.pieces[y][7]
			rook.x -= 2
			rook.hasMoved = True

		if(type(p) is King and newX == x - 2):
			rook = self.pieces[y][0]
			rook.x += 3
			rook.hasMoved = True

		p.hasMoved = True
		p.x = newX
		p.y = newY
		self.pieces[newY][newX] = None
		self.update()

		# Promotion is always to a queen.
		if(type(p) is Pawn):
			if(p.white and p.y == 0):
				self.pieces[p.y][p.x] = Queen(p.x, p.y, p.white, 'whitequeen.png', 'Q')
			elif(not p.white and p.y == 7):
				self.pieces[p.y][p.x] = Queen(p.x, p.y, p.white, 'blackqueen.png', 'q')

		if(self.checked(p.white)):
			self.pieces = snapshot
			return False

		self.update()
		if(self.turn == 0):
			self.turn = 1
		else:
			self.turn = 0
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
		king = None
		for row in self.pieces:
			for piece in row:
				if(piece is None):
					continue
				if(type(piece) is King and piece.white == white):
					king = piece

		if(king is None):
			print('Cannot find king')
			exit(0)

		for dx in range(-1, 2):
			for dy in range(-1, 2):
				if(dx == 0 and dy == 0):
					continue
				if(self.moveAlongPath(king.x, king.y, dx, dy, king.white)):
					return True

		if(king.y < 7 and king.x < 6 and self.pieces[king.y + 1][king.x + 2] is not None and type(self.pieces[king.y + 1][king.x + 2]) is Knight and self.pieces[king.y + 1][king.x + 2].white != white):
			return True
		if(king.y < 6 and king.x < 7 and self.pieces[king.y + 2][king.x + 1] is not None and type(self.pieces[king.y + 2][king.x + 1]) is Knight and self.pieces[king.y + 2][king.x + 1].white != white):
			return True
		if(king.y < 6 and king.x > 0 and self.pieces[king.y + 2][king.x - 1] is not None and type(self.pieces[king.y + 2][king.x - 1]) is Knight and self.pieces[king.y + 2][king.x - 1].white != white):
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

		if(white):
			if(king.y - 1 > -1 and king.x - 1 > -1):
				if(self.pieces[king.y - 1][king.x - 1] is not None and type(self.pieces[king.y - 1][king.x - 1]) is Pawn and self.pieces[king.y - 1][king.x - 1].white != white):
					return True
			if(king.y - 1 > -1 and king.x + 1 < 8):
				if(self.pieces[king.y - 1][king.x + 1] is not None and type(self.pieces[king.y - 1][king.x + 1]) is Pawn and self.pieces[king.y - 1][king.x + 1].white != white):
					return True

		else:
			if(king.y + 1 < 8 and king.x - 1 > -1):
				if(self.pieces[king.y + 1][king.x - 1] is not None and type(self.pieces[king.y + 1][king.x - 1]) is Pawn and self.pieces[king.y + 1][king.x - 1].white != white):
					return True
			if(king.y + 1 < 8 and king.x + 1 < 8):
				if(self.pieces[king.y + 1][king.x + 1] is not None and type(self.pieces[king.y + 1][king.x + 1]) is Pawn and self.pieces[king.y + 1][king.x + 1].white != white):
					return True

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
			if(dx == 0 or dy == 0):
				if(self.pieces[y][x] is None):
					x += dx
					y += dy
					continue
				if((self.pieces[y][x].white != white) and (type(self.pieces[y][x]) is Queen or type(self.pieces[y][x]) is Rook)):
					return True
				break

			else:
				if(self.pieces[y][x] is None):
					x += dx
					y += dy
					continue
				if((self.pieces[y][x].white != white) and (type(self.pieces[y][x]) is Queen or type(self.pieces[y][x]) is Bishop)):
					return True
				break

		return False

	def candidateSquares(self, piece):
		# Geometric destination candidates (newY, newX): a small superset of
		# the piece's legal destinations. Legality (pins, checks, castling
		# conditions) is still decided by trial makeMove.
		x = piece.x
		y = piece.y
		res = []

		if(type(piece) is Pawn):
			dy = -1 if piece.white else 1
			if(y + dy < 0 or y + dy > 7):
				return res
			if(self.pieces[y + dy][x] is None):
				res.append((y + dy, x))
				if(not piece.hasMoved and y + 2 * dy > -1 and y + 2 * dy < 8 and self.pieces[y + 2 * dy][x] is None):
					res.append((y + 2 * dy, x))
			for dx in (-1, 1):
				if(x + dx < 0 or x + dx > 7):
					continue
				target = self.pieces[y + dy][x + dx]
				if(target is not None and target.white != piece.white):
					res.append((y + dy, x + dx))
			return res

		if(type(piece) is Knight):
			for dy, dx in ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)):
				a = y + dy
				b = x + dx
				if(a < 0 or a > 7 or b < 0 or b > 7):
					continue
				if(self.pieces[a][b] is None or self.pieces[a][b].white != piece.white):
					res.append((a, b))
			return res

		if(type(piece) is King):
			for dy in range(-1, 2):
				for dx in range(-1, 2):
					if(dy == 0 and dx == 0):
						continue
					a = y + dy
					b = x + dx
					if(a < 0 or a > 7 or b < 0 or b > 7):
						continue
					if(self.pieces[a][b] is None or self.pieces[a][b].white != piece.white):
						res.append((a, b))
			if(not piece.hasMoved):
				if(x + 2 < 8):
					res.append((y, x + 2))
				if(x - 2 > -1):
					res.append((y, x - 2))
			return res

		directions = []
		if(type(piece) is Rook or type(piece) is Queen):
			directions += [(0, 1), (0, -1), (1, 0), (-1, 0)]
		if(type(piece) is Bishop or type(piece) is Queen):
			directions += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
		for dy, dx in directions:
			a = y + dy
			b = x + dx
			while(a > -1 and a < 8 and b > -1 and b < 8):
				if(self.pieces[a][b] is None):
					res.append((a, b))
					a += dy
					b += dx
					continue
				if(self.pieces[a][b].white != piece.white):
					res.append((a, b))
				break
		return res


	def allMoves(self, w):
		"""All legal moves for side w as [y, x, newY, newX]."""
		res = []
		for i in range(0, 8):
			for j in range(0, 8):
				piece = self.pieces[i][j]
				if(piece is None or piece.white != w):
					continue
				x = piece.x
				y = piece.y
				for a, b in self.candidateSquares(piece):
					snapshot = copy.deepcopy(self.pieces), self.turn
					if(self.makeMove(x, y, b, a)):
						curr = [y, x, a, b]
						res.append(curr)
						self.pieces, self.turn = snapshot
		return res


	def noisyMoves(self, w, includeChecks=False):
		"""Legal captures and promotion pushes as [y, x, newY, newX].

		With includeChecks, quiet moves that give check are included too.
		"""
		res = []
		for i in range(0, 8):
			for j in range(0, 8):
				piece = self.pieces[i][j]
				if(piece is None or piece.white != w):
					continue
				x = piece.x
				y = piece.y
				for a, b in self.candidateSquares(piece):
					noisy = self.pieces[a][b] is not None or (type(piece) is Pawn and (a == 0 or a == 7))
					if(not noisy and not includeChecks):
						continue
					snapshot = copy.deepcopy(self.pieces), self.turn
					if(self.makeMove(x, y, b, a)):
						keep = noisy or self.checked(not w)
						self.pieces, self.turn = snapshot
						if(keep):
							curr = [y, x, a, b]
							res.append(curr)
		return res
