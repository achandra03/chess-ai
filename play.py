import random
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "core" / "ai"))
sys.path.insert(0, str(ROOT_DIR / "core" / "game"))


def choose_color():
	prompt = "Play as (w)hite, (b)lack, or (r)andom? "
	while True:
		try:
			choice = input(prompt).strip().lower()
		except (EOFError, KeyboardInterrupt):
			print()
			return True
		if choice in ("w", "white"):
			return True
		if choice in ("b", "black"):
			return False
		if choice in ("r", "random", ""):
			human_white = random.choice((True, False))
			print(f"Random: you play {'white' if human_white else 'black'}.")
			return human_white
		print("Please answer w, b, or r.")


def main():
	from game import Game

	print("Chess vs. the perspective-transformer evaluator.")
	human_white = choose_color()

	try:
		game = Game(human_white=human_white)
	except FileNotFoundError as error:
		print(f"\nCannot start: {error}", file=sys.stderr)
		print(
			"Train an evaluator with `python train.py`, or point "
			"CHESS_AI_MODEL_PATH at a .keras model.",
			file=sys.stderr,
		)
		raise SystemExit(1)

	print("Click a piece, then click where it should go. Close the window to quit.")
	game.run()


if __name__ == "__main__":
	try:
		main()
	except KeyboardInterrupt:
		print("\nGame abandoned.")
		raise SystemExit(130)
