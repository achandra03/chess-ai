import os
import sys
from pathlib import Path

GAME_DIR = Path(__file__).resolve().parent
AI_DIR = GAME_DIR.parent / "ai"
ASSET_DIR = GAME_DIR / "assets"

sys.path.append(str(AI_DIR))
import pygame
from board import Board
from board_features import board_position_key
from engine import Engine

SQUARE_SIZE = 64
BOARD_PIXELS = SQUARE_SIZE * 8
PIECE_SIZE = 55
PIECE_INSET = (SQUARE_SIZE - PIECE_SIZE) // 2
ENGINE_DEPTH = 2
# Depth 1 costs ~3s in a busy middlegame, so keep headroom above that:
# past the budget the engine falls back to a shallow ordered move.
ENGINE_TIME_BUDGET = 10.0
HIGHLIGHT = (246, 246, 105)


class Game:
	def __init__(self, human_white=True, depth=ENGINE_DEPTH, time_budget=ENGINE_TIME_BUDGET):
		self.board = Board()
		self.human_white = human_white
		self.time_budget = time_budget
		# Selection is a square, never a Piece: Board.makeMove mutates a
		# piece before testing legality and then restores the board from a
		# deepcopy, which leaves any held reference orphaned and carrying
		# the rejected destination's coordinates.
		self.selected_square = None
		self.history = {}

		pygame.init()
		self.screen = pygame.display.set_mode((BOARD_PIXELS, BOARD_PIXELS))
		self.clock = pygame.time.Clock()
		pygame.display.set_caption("Chess")

		square_size = (SQUARE_SIZE, SQUARE_SIZE)
		self.darksquare = pygame.transform.scale(
			pygame.image.load(str(ASSET_DIR / "darksquare.png")), square_size
		)
		self.lightsquare = pygame.transform.scale(
			pygame.image.load(str(ASSET_DIR / "lightsquare.png")), square_size
		)

		self.piece_mapping = {}
		piece_directory = ASSET_DIR / "pieces"
		for file in os.listdir(piece_directory):
			if not file.endswith(".png"):
				continue
			piece = pygame.image.load(str(piece_directory / file))
			piece.set_colorkey((255, 255, 255))
			piece.convert_alpha()
			self.piece_mapping[file] = pygame.transform.scale(
				piece, (PIECE_SIZE, PIECE_SIZE)
			)

		self.engine = Engine(
			self.board,
			model_path=os.environ.get("CHESS_AI_MODEL_PATH") or None,
			depth=depth,
		)
		self.record_position()

	# Board squares are stored with y=0 as rank 8. When the human plays
	# black the view is rotated so their pieces sit at the bottom.
	def board_to_screen(self, x, y):
		if not self.human_white:
			x, y = 7 - x, 7 - y
		return x * SQUARE_SIZE, y * SQUARE_SIZE

	def screen_to_board(self, position):
		x = position[0] // SQUARE_SIZE
		y = position[1] // SQUARE_SIZE
		if not self.human_white:
			x, y = 7 - x, 7 - y
		return int(x), int(y)

	def white_to_move(self):
		return self.board.turn == 0

	def human_to_move(self):
		return self.white_to_move() == self.human_white

	def record_position(self):
		key = board_position_key(self.board)
		self.history[key] = self.history.get(key, 0) + 1
		return self.history[key]

	def result_text(self):
		"""Result string once the side to move has no legal reply."""
		side = self.white_to_move()
		if self.board.allMoves(side):
			return None
		if self.board.checked(side):
			return "You win" if side != self.human_white else "Bot wins"
		return "Stalemate"

	def render(self, status=None):
		self.screen.fill((0, 0, 0))
		for x in range(8):
			for y in range(8):
				square = self.lightsquare if (x + y) % 2 == 0 else self.darksquare
				self.screen.blit(square, self.board_to_screen(x, y))

		if self.selected_square is not None:
			left, top = self.board_to_screen(*self.selected_square)
			highlight = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE))
			highlight.set_alpha(120)
			highlight.fill(HIGHLIGHT)
			self.screen.blit(highlight, (left, top))

		for row in self.board.pieces:
			for piece in row:
				if piece is None:
					continue
				left, top = self.board_to_screen(piece.x, piece.y)
				self.screen.blit(
					self.piece_mapping[piece.filename],
					(left + PIECE_INSET, top + PIECE_INSET),
				)

		pygame.display.set_caption(f"Chess - {status}" if status else "Chess")
		pygame.display.flip()

	def apply_move(self, x, y, newX, newY):
		if not self.board.makeMove(x, y, newX, newY):
			return False
		self.record_position()
		return True

	def handle_click(self, position):
		x, y = self.screen_to_board(position)
		if not (0 <= x < 8 and 0 <= y < 8):
			return
		target = self.board.pieces[y][x]

		# Clicking one of your own pieces always (re)selects it, so a
		# mis-click does not throw away the current selection.
		if target is not None and target.white == self.human_white:
			self.selected_square = (x, y)
			return
		if self.selected_square is None:
			return

		fromX, fromY = self.selected_square
		origin = self.board.pieces[fromY][fromX]
		if origin is None or origin.white != self.human_white:
			self.selected_square = None
			return
		if self.apply_move(fromX, fromY, x, y):
			self.selected_square = None

	def play_engine_move(self):
		self.render(status="thinking")
		pygame.event.pump()
		self.engine.set_game_history(self.history)
		move = self.engine.selectMove(time_budget=self.time_budget)
		if move is None:
			return False
		y, x, newY, newX = move
		self.apply_move(x, y, newX, newY)
		return True

	def run(self):
		self.render()
		while True:
			result = self.result_text()
			if result is not None:
				self.render(status=result)
				return self.wait_for_quit(result)
			if self.history.get(board_position_key(self.board), 0) >= 3:
				self.render(status="Draw by repetition")
				return self.wait_for_quit("Draw by repetition")

			if self.human_to_move():
				for event in pygame.event.get():
					if event.type == pygame.QUIT:
						return None
					if event.type == pygame.MOUSEBUTTONDOWN:
						self.handle_click(event.pos)
				self.clock.tick(60)
			elif not self.play_engine_move():
				return self.wait_for_quit("Bot has no move")

			self.render()

	def wait_for_quit(self, result):
		print(result)
		while True:
			for event in pygame.event.get():
				if event.type in (pygame.QUIT, pygame.KEYDOWN):
					return result
			self.clock.tick(30)


if __name__ == "__main__":
	Game().run()
