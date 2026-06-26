import argparse
import csv
import hashlib
import math
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf

from position_encoding import (
	PAPER_BITMAP_FEATURE_SIZE,
	encode_fen_paper_bitmap,
	evaluation_to_paper_pawns,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
AI_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_FILE = ROOT_DIR / "data" / "chessData_depth8.csv"
DEFAULT_MODEL_OUT = AI_DIR / "position_evaluator_paper_mlp_mae.keras"
DEFAULT_WEIGHTS_OUT = AI_DIR / "position_evaluator_paper_mlp_mae.weights.h5"
DEFAULT_LEARNING_RATE = 0.0001
DEFAULT_MAX_ABS_PAWNS = 10.0


def parse_args():
	parser = argparse.ArgumentParser(
		description="Train the Sabatelli-style depth-8 bitmap MLP evaluator."
	)
	parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
	parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_OUT)
	parser.add_argument("--weights-out", type=Path, default=DEFAULT_WEIGHTS_OUT)
	parser.add_argument(
		"--epochs",
		type=int,
		default=600,
		help=(
			"The paper does not report an exact epoch count. 600 approximates "
			"its reported three-day convergence time at 440 seconds per epoch."
		),
	)
	parser.add_argument("--batch-size", type=int, default=248)
	parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
	parser.add_argument("--momentum", type=float, default=0.7)
	parser.add_argument(
		"--sample-size",
		type=int,
		default=3_000_000,
		help=(
			"Approximate number of positions selected deterministically from "
			"the source CSV. Set to 0 to use every row."
		),
	)
	parser.add_argument(
		"--sample-seed",
		default="sabatelli-dataset4-sample-v1",
		help="Seed for deterministic position sampling.",
	)
	parser.add_argument(
		"--split-seed",
		default="sabatelli-dataset4-split-v1",
		help="Seed for the deterministic 80/10/10 position split.",
	)
	parser.add_argument(
		"--max-abs-pawns",
		type=float,
		default=DEFAULT_MAX_ABS_PAWNS,
		help="Clip evaluations to this pawn range before training.",
	)
	parser.add_argument("--shuffle-buffer", type=int, default=100_000)
	parser.add_argument(
		"--max-rows",
		type=int,
		default=None,
		help="Optional source-row cap for smoke tests.",
	)
	parser.add_argument(
		"--steps-per-epoch",
		type=int,
		default=None,
		help="Optional cap on training steps per epoch.",
	)
	parser.add_argument(
		"--validation-steps",
		type=int,
		default=None,
		help="Optional cap on validation steps.",
	)
	parser.add_argument(
		"--test-steps",
		type=int,
		default=None,
		help="Optional cap on final test steps.",
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Inspect one batch and the model without training.",
	)
	return parser.parse_args()


def configure_tensorflow():
	for gpu in tf.config.list_physical_devices("GPU"):
		try:
			tf.config.experimental.set_memory_growth(gpu, True)
		except RuntimeError:
			pass


def validate_args(args):
	if not args.data_file.is_file():
		raise FileNotFoundError(args.data_file)
	if args.epochs < 1:
		raise ValueError("--epochs must be positive")
	if args.batch_size < 1:
		raise ValueError("--batch-size must be positive")
	if args.learning_rate <= 0:
		raise ValueError("--learning-rate must be positive")
	if args.momentum < 0 or args.momentum >= 1:
		raise ValueError("--momentum must be in [0, 1)")
	if args.sample_size < 0:
		raise ValueError("--sample-size cannot be negative")
	if args.max_abs_pawns <= 0:
		raise ValueError("--max-abs-pawns must be positive")
	if args.shuffle_buffer < 0:
		raise ValueError("--shuffle-buffer cannot be negative")
	if args.max_rows is not None and args.max_rows < 1:
		raise ValueError("--max-rows must be positive")
	for name in ("steps_per_epoch", "validation_steps", "test_steps"):
		value = getattr(args, name)
		if value is not None and value < 1:
			raise ValueError(f"--{name.replace('_', '-')} must be positive")
	if not str(args.weights_out).endswith(".weights.h5"):
		raise ValueError("--weights-out must end with .weights.h5")


def build_model(
	learning_rate=DEFAULT_LEARNING_RATE,
	momentum=0.7,
	max_abs_pawns=DEFAULT_MAX_ABS_PAWNS,
):
	inputs = tf.keras.Input(
		shape=(PAPER_BITMAP_FEATURE_SIZE,), name="bitmap_position"
	)
	x = tf.keras.layers.Dense(2048, activation="elu", name="hidden_1")(inputs)
	x = tf.keras.layers.BatchNormalization(name="batch_norm_1")(x)
	x = tf.keras.layers.Dense(2048, activation="elu", name="hidden_2")(x)
	x = tf.keras.layers.BatchNormalization(name="batch_norm_2")(x)
	x = tf.keras.layers.Dense(2048, activation="elu", name="hidden_3")(x)
	unit_score = tf.keras.layers.Dense(
		1,
		activation="tanh",
		kernel_initializer="zeros",
		bias_initializer="zeros",
		name="bounded_unit_score",
	)(x)
	outputs = tf.keras.layers.Rescaling(
		scale=max_abs_pawns, name="side_to_move_pawn_score"
	)(unit_score)

	model = tf.keras.Model(
		inputs=inputs,
		outputs=outputs,
		name="sabatelli_bitmap_mlp",
	)
	model.compile(
		optimizer=tf.keras.optimizers.SGD(
			learning_rate=learning_rate,
			momentum=momentum,
			nesterov=True,
		),
		loss=tf.keras.losses.MeanAbsoluteError(),
		metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
	)
	return model


def position_key(fen):
	fields = str(fen).strip().split()
	if len(fields) < 4:
		raise ValueError(f"invalid FEN: {fen}")
	return " ".join(fields[:4])


def hash_fraction(seed, value):
	digest = hashlib.blake2b(
		f"{seed}\0{value}".encode("utf-8"), digest_size=8
	).digest()
	return int.from_bytes(digest, "big") / 2**64


def split_for_fen(fen, split_seed):
	value = hash_fraction(split_seed, position_key(fen))
	if value < 0.8:
		return "train"
	if value < 0.9:
		return "validation"
	return "test"


def selected_for_sample(fen, sample_fraction, sample_seed):
	if sample_fraction >= 1.0:
		return True
	return hash_fraction(sample_seed, position_key(fen)) < sample_fraction


def count_source_rows(path, max_rows):
	count = 0
	with path.open(newline="") as file:
		reader = csv.DictReader(file)
		validate_csv_header(reader.fieldnames)
		for _ in reader:
			if max_rows is not None and count >= max_rows:
				break
			count += 1
	return count


def sample_fraction_for(raw_rows, sample_size):
	if sample_size == 0 or sample_size >= raw_rows:
		return 1.0
	return sample_size / raw_rows


def count_split_rows(path, raw_rows, sample_fraction, sample_seed, split_seed):
	counts = {"train": 0, "validation": 0, "test": 0}
	with path.open(newline="") as file:
		reader = csv.DictReader(file)
		validate_csv_header(reader.fieldnames)
		for row_index, row in enumerate(reader):
			if row_index >= raw_rows:
				break
			fen = row["FEN"]
			if not selected_for_sample(fen, sample_fraction, sample_seed):
				continue
			counts[split_for_fen(fen, split_seed)] += 1
	return counts


def validate_csv_header(fieldnames):
	if fieldnames is None:
		raise ValueError("input CSV has no header")
	if "FEN" not in fieldnames or "Evaluation" not in fieldnames:
		raise ValueError("input CSV must contain FEN and Evaluation columns")


def iter_examples(
	path,
	raw_rows,
	split,
	sample_fraction,
	sample_seed,
	split_seed,
	max_abs_pawns,
):
	with path.open(newline="") as file:
		reader = csv.DictReader(file)
		for row_index, row in enumerate(reader):
			if row_index >= raw_rows:
				break

			fen = row["FEN"]
			if not selected_for_sample(fen, sample_fraction, sample_seed):
				continue
			if split_for_fen(fen, split_seed) != split:
				continue

			features = encode_fen_paper_bitmap(fen)
			target = evaluation_to_paper_pawns(
				row["Evaluation"],
				max_abs_pawns=max_abs_pawns,
			)
			yield features, np.array([target], dtype=np.float32)


def make_dataset(args, raw_rows, split, sample_fraction, repeat=False):
	dataset = tf.data.Dataset.from_generator(
		lambda: iter_examples(
			path=args.data_file,
			raw_rows=raw_rows,
			split=split,
			sample_fraction=sample_fraction,
			sample_seed=args.sample_seed,
			split_seed=args.split_seed,
			max_abs_pawns=args.max_abs_pawns,
		),
		output_signature=(
			tf.TensorSpec(
				shape=(PAPER_BITMAP_FEATURE_SIZE,), dtype=tf.float32
			),
			tf.TensorSpec(shape=(1,), dtype=tf.float32),
		),
	)
	if split == "train" and args.shuffle_buffer > 0:
		dataset = dataset.shuffle(
			args.shuffle_buffer,
			seed=int(hash_fraction(args.split_seed, "shuffle") * (2**31 - 1)),
			reshuffle_each_iteration=True,
		)
	if repeat:
		dataset = dataset.repeat()
	return dataset.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)


def capped_steps(row_count, batch_size, requested_steps):
	steps = math.ceil(row_count / batch_size)
	if requested_steps is not None:
		steps = min(steps, requested_steps)
	return steps


def main():
	args = parse_args()
	validate_args(args)
	configure_tensorflow()

	raw_rows = count_source_rows(args.data_file, args.max_rows)
	if raw_rows == 0:
		raise ValueError("input CSV contains no data rows")

	sample_fraction = sample_fraction_for(raw_rows, args.sample_size)
	split_counts = count_split_rows(
		path=args.data_file,
		raw_rows=raw_rows,
		sample_fraction=sample_fraction,
		sample_seed=args.sample_seed,
		split_seed=args.split_seed,
	)
	if split_counts["train"] == 0:
		raise ValueError("deterministic sample contains no training rows")

	print("Experiment: Sabatelli Dataset 4 bitmap MLP with pawn-scale MAE")
	print(f"Data file: {args.data_file}")
	print(f"Source rows considered: {raw_rows}")
	print(
		f"Deterministic sample: {sum(split_counts.values())} positions "
		f"({sample_fraction:.4%} of considered rows)"
	)
	print(
		"Split: "
		f"{split_counts['train']} train, "
		f"{split_counts['validation']} validation, "
		f"{split_counts['test']} test"
	)
	print(f"Input: {PAPER_BITMAP_FEATURE_SIZE} binary bitmap values")
	print(f"Target: clipped to [-{args.max_abs_pawns:g}, +{args.max_abs_pawns:g}] pawns")
	print("Loss/metric: MAE in pawns")
	print(
		f"Optimizer: SGD(lr={args.learning_rate:g}, momentum={args.momentum:g}, "
		"nesterov=True)"
	)
	print(f"Batch size: {args.batch_size}")
	print(f"Model checkpoint: {args.model_out}")
	print(f"Weights checkpoint: {args.weights_out}")

	train_dataset = make_dataset(
		args, raw_rows, "train", sample_fraction, repeat=True
	)
	validation_dataset = make_dataset(
		args, raw_rows, "validation", sample_fraction
	)
	test_dataset = make_dataset(args, raw_rows, "test", sample_fraction)

	model = build_model(
		args.learning_rate,
		args.momentum,
		max_abs_pawns=args.max_abs_pawns,
	)
	if args.dry_run:
		for features, targets in train_dataset.take(1):
			print(f"features: {features.shape} {features.dtype}")
			print(f"targets: {targets.shape} {targets.dtype}")
			print(
				f"target range: {targets.numpy().min():.5f} "
				f"to {targets.numpy().max():.5f}"
			)
		print(f"Model parameters: {model.count_params()}")
		return

	args.model_out.parent.mkdir(parents=True, exist_ok=True)
	args.weights_out.parent.mkdir(parents=True, exist_ok=True)

	steps_per_epoch = capped_steps(
		split_counts["train"], args.batch_size, args.steps_per_epoch
	)
	validation_steps = capped_steps(
		split_counts["validation"], args.batch_size, args.validation_steps
	)
	test_steps = capped_steps(
		split_counts["test"], args.batch_size, args.test_steps
	)

	callbacks = [
		tf.keras.callbacks.ModelCheckpoint(
			filepath=str(args.model_out),
			monitor="val_mae",
			mode="min",
			save_best_only=True,
		),
		tf.keras.callbacks.ModelCheckpoint(
			filepath=str(args.weights_out),
			monitor="val_mae",
			mode="min",
			save_best_only=True,
			save_weights_only=True,
		),
	]

	model.fit(
		train_dataset,
		epochs=args.epochs,
		steps_per_epoch=steps_per_epoch,
		validation_data=validation_dataset,
		validation_steps=validation_steps,
		callbacks=callbacks,
	)

	model.load_weights(args.weights_out)
	results = model.evaluate(
		test_dataset,
		steps=test_steps,
		return_dict=True,
	)
	print(f"Saved best evaluator to {args.model_out}")
	print(f"Saved best weights to {args.weights_out}")
	print(
		"Test metrics: "
		+ ", ".join(f"{name}={value:.6f}" for name, value in results.items())
	)


if __name__ == "__main__":
	main()
