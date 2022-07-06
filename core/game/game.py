import pygame
from board import Board
import os
import time
import sys

class Game:

	def __init__(self):
		self.board = Board()
		self.whiteTurn = True
		pygame.init()
		size = width, height = 512, 512
		self.screen = pygame.display.set_mode(size)
		square_size = (64, 64)
		self.darksquare = pygame.image.load('/Users/arnavchandra/Desktop/chess/core/game/assets/darksquare.png')
		self.darksquare = pygame.transform.scale(self.darksquare, square_size)
		self.lightsquare = pygame.image.load('/Users/arnavchandra/Desktop/chess/core/game/assets/lightsquare.png')
		self.lightsquare = pygame.transform.scale(self.lightsquare, square_size)
		piece_size = (55, 55)
		self.piece_mapping = {}
		piece_directory = '/Users/arnavchandra/Desktop/chess/core/game/assets/pieces/'
		white = 255, 255, 255
		for file in os.listdir(piece_directory):
			if(not file.endswith('.png')):
				continue
			piece = pygame.image.load(piece_directory + file)
			piece.set_colorkey(white)
			piece.convert_alpha()
			piece = pygame.transform.scale(piece, piece_size)
			self.piece_mapping[file] = piece

		self.clickedOn = None


	def move(self, x, y, newX, newY):
		status = self.board.makeMove(x, y, newX, newY)
		if(status):
			self.whiteTurn = not self.whiteTurn
			return True
		return False

	def render(self):
		black = 0, 0, 0
		self.screen.fill(black)
		for i in range(0, 8):
			for j in range(0, 8):
				square = None
				if(i % 2 == j % 2):
					square = self.lightsquare
				else:
					square = self.darksquare

				self.screen.blit(square, (i * 64, j * 64))

		for i in range(0, 8):
			for j in range(0, 8):
				if(self.board.pieces[i][j] is None):
					continue
				piece = self.board.pieces[i][j]
				image = self.piece_mapping[piece.filename]
				self.screen.blit(image, (j * 64 + 5, i * 64 + 5))


		pygame.display.flip()

	def run(self):
		while True:
			self.render()
			for event in pygame.event.get():
				if(event.type == pygame.QUIT):
					sys.exit()
				if(event.type == pygame.MOUSEBUTTONDOWN):
					x, y = pygame.mouse.get_pos()
					boardY = y // 64
					boardX = x // 64

					if(self.board.pieces[boardY][boardX] is None):
						if(self.clickedOn is None):
							continue
						if(self.move(self.clickedOn.x, self.clickedOn.y, boardX, boardY)):
							self.clickedOn = None
						else:
							self.clickedOn = None

					else:
						if(self.board.pieces[boardY][boardX].white == self.whiteTurn):
							self.clickedOn = self.board.pieces[boardY][boardX]
						else:
							if(self.clickedOn is None):
								continue
							if(self.move(self.clickedOn.x, self.clickedOn.y, boardX, boardY)):
								self.clickedOn = None

game = Game()
game.run()
