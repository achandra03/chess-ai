"""Pre-encode a FEN/Evaluation CSV into a packed on-disk training cache.

Encoding FENs in Python is the training bottleneck, so this runs once,
in parallel, and training then streams the packed rows from memmaps.

Usage:
  python core/ai/encode_dataset.py                      # depth8 dataset
  python core/ai/encode_dataset.py --data-file data/tactic_evals.csv \
      --out-dir data/encoded/tactic_evals_perspective_v2 --train-only
"""

import argparse
import csv
import multiprocessing
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from board_features import (
	encode_fen_perspective_v2,
	evaluation_to_paper_pawns,
	hash_fraction,
	position_key,
	split_for_fen,
)
from encoded_cache import (
	CACHE_ARCHITECTURE,
	FEATURE_SIZE,
	MIRROR_PERMUTATION,
	CacheWriter,
	SPLIT_CODES,
	pack_features,
	unpack_features,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_FILE = ROOT_DIR / "data" / "chessData_depth8.csv"
DEFAULT_SPLIT_SEED = "sabatelli-dataset4-split-v1"
DEFAULT_SAMPLE_SEED = "sabatelli-dataset4-sample-v1"
DEFAULT_MAX_ABS_PAWNS = 10.0

_WORKER_CONFIG = None


def parse_args():
	parser = argparse.ArgumentParser(
		description=f"Pre-encode positions for the {CACHE_ARCHITECTURE} trainer."
	)
	parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
	parser.add_argument(
		"--out-dir",
		type=Path,
		default=None,
		help="Cache directory. Defaults to data/encoded/<csv stem>_perspective_v2.",
	)
	parser.add_argument(
		"--workers",
		type=int,
		default=max(1, multiprocessing.cpu_count() - 2),
	)
	parser.add_argument("--chunk-rows", type=int, default=8192)
	parser.add_argument(
		"--max-rows",
		type=int,
		default=None,
		help="Optional source-row cap for smoke tests.",
	)
	parser.add_argument(
		"--train-only",
		action="store_true",
		help="Mark every row as training data (for extra datasets).",
	)
	parser.add_argument("--split-seed", default=DEFAULT_SPLIT_SEED)
	parser.add_argument("--sample-seed", default=DEFAULT_SAMPLE_SEED)
	parser.add_argument("--max-abs-pawns", type=float, default=DEFAULT_MAX_ABS_PAWNS)
	parser.add_argument(
		"--verify-rows",
		type=int,
		default=200,
		help="Rows used to self-check packing and the mirror permutation.",
	)
	args = parser.parse_args()
	if args.out_dir is None:
		args.out_dir = (
			ROOT_DIR / "data" / "encoded" / f"{args.data_file.stem}_perspective_v2"
		)
	if args.workers < 1:
		raise ValueError("--workers must be positive")
	if args.chunk_rows < 1:
		raise ValueError("--chunk-rows must be positive")
	if args.max_abs_pawns <= 0:
		raise ValueError("--max-abs-pawns must be positive")
	return args


def open_csv_columns(path):
	file = path.open(newline="", encoding="utf-8")
	reader = csv.reader(file)
	header = next(reader, None)
	if header is None or "FEN" not in header or "Evaluation" not in header:
		file.close()
		raise ValueError("input CSV must contain FEN and Evaluation columns")
	return file, reader, header.index("FEN"), header.index("Evaluation")


def count_rows(path, max_rows):
	file, reader, _, _ = open_csv_columns(path)
	with file:
		count = 0
		for _ in reader:
			count += 1
			if max_rows is not None and count >= max_rows:
				break
	return count


def iter_chunks(path, max_rows, chunk_rows):
	file, reader, fen_index, eval_index = open_csv_columns(path)
	with file:
		chunk = []
		seen = 0
		for row in reader:
			chunk.append((row[fen_index], row[eval_index]))
			seen += 1
			if len(chunk) >= chunk_rows:
				yield chunk
				chunk = []
			if max_rows is not None and seen >= max_rows:
				break
		if chunk:
			yield chunk


def init_worker(config):
	global _WORKER_CONFIG
	_WORKER_CONFIG = config


def encode_chunk(chunk):
	config = _WORKER_CONFIG
	count = len(chunk)
	features = np.empty((count, FEATURE_SIZE), dtype=np.float32)
	targets = np.empty(count, dtype=np.float32)
	splits = np.zeros(count, dtype=np.uint8)
	sample = np.empty(count, dtype=np.float32)

	for index, (fen, evaluation) in enumerate(chunk):
		features[index] = encode_fen_perspective_v2(fen)
		targets[index] = evaluation_to_paper_pawns(
			evaluation, max_abs_pawns=config["max_abs_pawns"]
		)
		key = position_key(fen)
		sample[index] = hash_fraction(config["sample_seed"], key)
		if not config["train_only"]:
			splits[index] = SPLIT_CODES[split_for_fen(fen, config["split_seed"])]

	boards, metadata = pack_features(features)
	return boards, metadata, targets, splits, sample


def verify_encoding(path, max_rows, verify_rows):
	"""Check pack/unpack and the mirror permutation against direct encoding."""
	file, reader, fen_index, _ = open_csv_columns(path)
	with file:
		checked = 0
		for row in reader:
			if checked >= verify_rows or (max_rows is not None and checked >= max_rows):
				break
			fen = row[fen_index]
			direct = encode_fen_perspective_v2(fen)
			boards, metadata = pack_features(direct[np.newaxis, :])
			round_trip = unpack_features(boards, metadata)[0]
			if not np.allclose(direct, round_trip, atol=0.01):
				raise AssertionError(f"pack/unpack mismatch for FEN: {fen}")
			mirrored_direct = encode_fen_perspective_v2(fen, mirror_files=True)
			mirrored_cached = round_trip[MIRROR_PERMUTATION]
			if not np.allclose(mirrored_direct, mirrored_cached, atol=0.01):
				raise AssertionError(f"mirror permutation mismatch for FEN: {fen}")
			checked += 1
	if checked == 0:
		raise ValueError("input CSV contains no data rows")
	print(f"Verified packing and mirror permutation on {checked} rows.")


def main():
	args = parse_args()
	if not args.data_file.is_file():
		raise FileNotFoundError(args.data_file)

	print(f"Source: {args.data_file}")
	print(f"Cache: {args.out_dir}")
	verify_encoding(args.data_file, args.max_rows, args.verify_rows)

	print("Counting source rows...", flush=True)
	rows = count_rows(args.data_file, args.max_rows)
	print(f"Encoding {rows} rows with {args.workers} workers.", flush=True)

	writer = CacheWriter(args.out_dir, rows)
	split_counts = {name: 0 for name in SPLIT_CODES}
	config = {
		"train_only": args.train_only,
		"split_seed": args.split_seed,
		"sample_seed": args.sample_seed,
		"max_abs_pawns": args.max_abs_pawns,
	}

	start = time.monotonic()
	last_report = start
	with multiprocessing.Pool(
		processes=args.workers,
		initializer=init_worker,
		initargs=(config,),
	) as pool:
		results = pool.imap(
			encode_chunk,
			iter_chunks(args.data_file, args.max_rows, args.chunk_rows),
		)
		for boards, metadata, targets, splits, sample in results:
			writer.append(boards, metadata, targets, splits, sample)
			for name, code in SPLIT_CODES.items():
				split_counts[name] += int(np.count_nonzero(splits == code))

			now = time.monotonic()
			if now - last_report >= 5.0 or writer.cursor == rows:
				rate = writer.cursor / max(now - start, 1e-9)
				remaining = (rows - writer.cursor) / max(rate, 1e-9)
				print(
					f"  {writer.cursor}/{rows} rows "
					f"({writer.cursor / rows:.1%}), "
					f"{rate:,.0f} rows/s, ETA {remaining / 60:.1f} min",
					flush=True,
				)
				last_report = now

	source_stat = args.data_file.stat()
	writer.finalize(
		{
			"source_file": str(args.data_file),
			"source_bytes": source_stat.st_size,
			"source_mtime": source_stat.st_mtime,
			"max_abs_pawns": args.max_abs_pawns,
			"split_seed": args.split_seed,
			"sample_seed": args.sample_seed,
			"train_only": args.train_only,
			"split_counts": split_counts,
			"created": datetime.now(timezone.utc).isoformat(),
		}
	)
	elapsed = time.monotonic() - start
	print(f"Encoded {rows} rows in {elapsed / 60:.1f} min.")
	print(
		"Split counts: "
		+ ", ".join(f"{name}={count}" for name, count in split_counts.items())
	)
	print(f"Cache written to {args.out_dir}")


if __name__ == "__main__":
	main()
