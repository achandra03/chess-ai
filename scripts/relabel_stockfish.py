#!/usr/bin/env python3

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT_DIR / "data" / "chessData.csv"
DEFAULT_OUTPUT = ROOT_DIR / "data" / "chessData_depth8.csv"
DEFAULT_STOCKFISH = ROOT_DIR / "stockfish" / "stockfish-macos-m1-apple-silicon"


def parse_args():
	parser = argparse.ArgumentParser(
		description="Re-evaluate CSV chess positions with Stockfish through UCI."
	)
	parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
	parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
	parser.add_argument(
		"--stockfish",
		default=os.environ.get("STOCKFISH_PATH", DEFAULT_STOCKFISH),
		help=(
			"Stockfish executable path. Defaults to STOCKFISH_PATH or the "
			"repository's Apple Silicon binary."
		),
	)
	parser.add_argument(
		"--depth",
		type=int,
		default=8,
		help="Fixed Stockfish search depth. The paper used depth 8.",
	)
	parser.add_argument("--threads", type=int, default=1)
	parser.add_argument("--hash-mb", type=int, default=128)
	parser.add_argument(
		"--max-positions",
		type=int,
		default=None,
		help="Optional number of new rows to evaluate, useful for testing.",
	)
	parser.add_argument(
		"--progress-every",
		type=int,
		default=1000,
		help="Print progress after this many newly evaluated positions.",
	)
	parser.add_argument(
		"--flush-every",
		type=int,
		default=100,
		help="Flush output after this many rows to make interruption resumable.",
	)
	parser.add_argument(
		"--overwrite",
		action="store_true",
		help="Replace an existing output file instead of resuming it.",
	)
	return parser.parse_args()


class StockfishUCI:
	def __init__(self, executable, threads, hash_mb):
		self.executable = resolve_executable(executable)
		self.process = subprocess.Popen(
			[str(self.executable)],
			stdin=subprocess.PIPE,
			stdout=subprocess.PIPE,
			stderr=None,
			text=True,
			bufsize=1,
		)
		try:
			self._send("uci")
			self._wait_for("uciok")
			self._send(f"setoption name Threads value {threads}")
			self._send(f"setoption name Hash value {hash_mb}")
			self._send("setoption name MultiPV value 1")
			self._send("setoption name UCI_AnalyseMode value true")
			self._send("isready")
			self._wait_for("readyok")
		except Exception:
			self.close(force=True)
			raise

	def evaluate(self, fen, depth):
		self._send(f"position fen {fen}")
		self._send(f"go depth {depth}")

		best_score = None
		best_depth = -1
		while True:
			line = self._readline()
			if line.startswith("bestmove"):
				break
			if not line.startswith("info "):
				continue

			score = parse_info_score(line)
			info_depth = parse_info_depth(line)
			if score is not None and info_depth >= best_depth:
				best_score = score
				best_depth = info_depth

		if best_score is None:
			raise RuntimeError(f"Stockfish returned no score for FEN: {fen}")
		return format_evaluation(*best_score)

	def close(self, force=False):
		if self.process.poll() is not None:
			return
		if not force:
			try:
				self._send("quit")
				self.process.wait(timeout=5)
				return
			except (BrokenPipeError, subprocess.TimeoutExpired):
				pass
		self.process.kill()
		self.process.wait()

	def _send(self, command):
		if self.process.stdin is None:
			raise RuntimeError("Stockfish stdin is unavailable")
		self.process.stdin.write(command + "\n")
		self.process.stdin.flush()

	def _readline(self):
		if self.process.stdout is None:
			raise RuntimeError("Stockfish stdout is unavailable")
		line = self.process.stdout.readline()
		if line:
			return line.strip()
		return_code = self.process.poll()
		raise RuntimeError(
			f"Stockfish exited unexpectedly with code {return_code}"
		)

	def _wait_for(self, expected):
		while True:
			line = self._readline()
			if line == expected:
				return


def resolve_executable(executable):
	path = Path(executable).expanduser()
	if path.parent != Path(".") or path.is_absolute():
		if path.is_file():
			return path.resolve()
		raise FileNotFoundError(f"Stockfish executable not found: {path}")

	resolved = shutil.which(executable)
	if resolved is None:
		raise FileNotFoundError(
			"Stockfish executable not found. Install Stockfish or pass "
			"--stockfish /path/to/stockfish."
		)
	return Path(resolved)


def parse_info_score(line):
	tokens = line.split()
	try:
		score_index = tokens.index("score")
	except ValueError:
		return None

	if score_index + 2 >= len(tokens):
		return None
	score_type = tokens[score_index + 1]
	if score_type not in ("cp", "mate"):
		return None
	try:
		return score_type, int(tokens[score_index + 2])
	except ValueError:
		return None


def parse_info_depth(line):
	tokens = line.split()
	try:
		return int(tokens[tokens.index("depth") + 1])
	except (ValueError, IndexError):
		return -1


def format_evaluation(score_type, value):
	if score_type == "mate":
		return f"#{value:+d}"
	if value > 0:
		return f"+{value}"
	return str(value)


def repair_incomplete_output(path):
	if not path.exists() or path.stat().st_size == 0:
		return

	with path.open("rb+") as file:
		file.seek(-1, os.SEEK_END)
		if file.read(1) == b"\n":
			return

		position = file.tell()
		while position > 0:
			position -= 1
			file.seek(position)
			if file.read(1) == b"\n":
				file.truncate(position + 1)
				return
		file.truncate(0)


def inspect_existing_output(path, expected_fields):
	if not path.exists() or path.stat().st_size == 0:
		return 0

	repair_incomplete_output(path)
	with path.open(newline="") as file:
		reader = csv.DictReader(file)
		if reader.fieldnames != expected_fields:
			raise ValueError(
				f"output header {reader.fieldnames} does not match input "
				f"header {expected_fields}"
			)
		return sum(1 for _ in reader)


def validate_args(args):
	if args.depth < 1:
		raise ValueError("--depth must be positive")
	if args.threads < 1:
		raise ValueError("--threads must be positive")
	if args.hash_mb < 1:
		raise ValueError("--hash-mb must be positive")
	if args.max_positions is not None and args.max_positions < 1:
		raise ValueError("--max-positions must be positive")
	if args.progress_every < 1:
		raise ValueError("--progress-every must be positive")
	if args.flush_every < 1:
		raise ValueError("--flush-every must be positive")
	if not args.input.is_file():
		raise FileNotFoundError(args.input)
	if args.input.resolve() == args.output.resolve():
		raise ValueError("--output must be different from --input")


def main():
	args = parse_args()
	validate_args(args)
	args.output.parent.mkdir(parents=True, exist_ok=True)

	if args.overwrite and args.output.exists():
		args.output.unlink()

	with args.input.open(newline="") as input_file:
		reader = csv.DictReader(input_file)
		if reader.fieldnames is None:
			raise ValueError("input CSV has no header")
		if "FEN" not in reader.fieldnames or "Evaluation" not in reader.fieldnames:
			raise ValueError("input CSV must contain FEN and Evaluation columns")

		completed_rows = inspect_existing_output(args.output, reader.fieldnames)
		for _ in range(completed_rows):
			if next(reader, None) is None:
				raise ValueError("output contains more rows than input")

		file_exists = args.output.exists() and args.output.stat().st_size > 0
		with args.output.open("a", newline="") as output_file:
			writer = csv.DictWriter(output_file, fieldnames=reader.fieldnames)
			if not file_exists:
				writer.writeheader()
				output_file.flush()

			print(f"Input: {args.input}")
			print(f"Output: {args.output}")
			print(f"Stockfish: {resolve_executable(args.stockfish)}")
			print(f"Depth: {args.depth}")
			print(f"Already completed: {completed_rows} positions")

			engine = StockfishUCI(
				executable=args.stockfish,
				threads=args.threads,
				hash_mb=args.hash_mb,
			)
			start_time = time.monotonic()
			new_rows = 0
			try:
				for row in reader:
					row["Evaluation"] = engine.evaluate(row["FEN"], args.depth)
					writer.writerow(row)
					new_rows += 1

					if new_rows % args.flush_every == 0:
						output_file.flush()
					if new_rows % args.progress_every == 0:
						elapsed = time.monotonic() - start_time
						rate = new_rows / elapsed if elapsed else 0.0
						print(
							f"Evaluated {completed_rows + new_rows} total "
							f"({new_rows} this run, {rate:.2f} positions/s)",
							flush=True,
						)
					if (
						args.max_positions is not None
						and new_rows >= args.max_positions
					):
						break
			finally:
				output_file.flush()
				engine.close()

	elapsed = time.monotonic() - start_time
	rate = new_rows / elapsed if elapsed else 0.0
	print(
		f"Finished {new_rows} new positions in {elapsed:.1f}s "
		f"({rate:.2f} positions/s)."
	)


if __name__ == "__main__":
	try:
		main()
	except KeyboardInterrupt:
		print(
			"\nInterrupted; rerun the same command to resume completed rows.",
			file=sys.stderr,
		)
		raise SystemExit(130)
