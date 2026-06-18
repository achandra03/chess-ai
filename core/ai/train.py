import argparse
import csv
import math
import os
import zlib
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf

from position_encoding import (
	ATTACK_PLANE_COUNT,
	ATTACK_BOARD_FEATURE_SIZE,
	ATTACK_FEATURE_SIZE,
	FEATURE_SIZE,
	METADATA_SIZE,
	PIECE_PLANE_COUNT,
	encode_fen,
	evaluation_to_pawns,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
AI_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
DEFAULT_DATA_FILES = [
	DATA_DIR / "chessData.csv",
	DATA_DIR / "random_evals.csv",
	DATA_DIR / "tactic_evals.csv",
]
DEFAULT_CNN_MODEL_OUT = AI_DIR / "position_evaluator_cnn_attacks.keras"
DEFAULT_CNN_WEIGHTS_OUT = AI_DIR / "position_evaluator_cnn_attacks.weights.h5"
DEFAULT_MLP_MODEL_OUT = AI_DIR / "position_evaluator.keras"
DEFAULT_MLP_WEIGHTS_OUT = AI_DIR / "position_evaluator.weights.h5"


def parse_args():
	parser = argparse.ArgumentParser(description="Train the chess position evaluator.")
	parser.add_argument(
		"--data-files",
		nargs="+",
		type=Path,
		default=DEFAULT_DATA_FILES,
		help="CSV files with FEN and Evaluation columns.",
	)
	parser.add_argument(
		"--model-out",
		type=Path,
		default=None,
		help="Path for the trained Keras evaluator.",
	)
	parser.add_argument(
		"--weights-out",
		type=Path,
		default=None,
		help="Optional path for best weights-only checkpoint.",
	)
	parser.add_argument(
		"--architecture",
		choices=("cnn", "mlp"),
		default="cnn",
		help="Evaluator architecture to train.",
	)
	parser.add_argument("--epochs", type=int, default=40)
	parser.add_argument("--batch-size", type=int, default=2048)
	parser.add_argument("--learning-rate", type=float, default=3e-4)
	parser.add_argument("--validation-fraction", type=float, default=0.05)
	parser.add_argument(
		"--split-seed",
		default="chess-ai-position-evaluator-v1",
		help="Seed string for the deterministic FEN hash train/validation split.",
	)
	parser.add_argument("--shuffle-buffer", type=int, default=100000)
	parser.add_argument(
		"--max-rows-per-file",
		type=int,
		default=None,
		help="Optional cap per CSV, useful for smoke tests.",
	)
	parser.add_argument(
		"--max-abs-pawns",
		type=float,
		default=10.0,
		help="Clip training targets to this absolute pawn score.",
	)
	parser.add_argument(
		"--focus-loss-pawns",
		type=float,
		default=3.0,
		help="Give positions within this absolute pawn score full loss weight. Set to 0 to disable weighting.",
	)
	parser.add_argument(
		"--min-sample-weight",
		type=float,
		default=0.25,
		help="Minimum loss weight for extreme positions when --focus-loss-pawns is enabled.",
	)
	parser.add_argument(
		"--steps-per-epoch",
		type=int,
		default=None,
		help="Optional cap on train steps per epoch.",
	)
	parser.add_argument(
		"--validation-steps",
		type=int,
		default=None,
		help="Optional cap on validation steps.",
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Build one batch and print its shape without training.",
	)
	return parser.parse_args()


def configure_tensorflow():
	for gpu in tf.config.list_physical_devices("GPU"):
		try:
			tf.config.experimental.set_memory_growth(gpu, True)
		except RuntimeError:
			pass


def resolve_output_paths(args):
	if args.model_out is None:
		if args.architecture == "cnn":
			args.model_out = DEFAULT_CNN_MODEL_OUT
		else:
			args.model_out = DEFAULT_MLP_MODEL_OUT

	if args.weights_out is None:
		if args.architecture == "cnn":
			args.weights_out = DEFAULT_CNN_WEIGHTS_OUT
		else:
			args.weights_out = DEFAULT_MLP_WEIGHTS_OUT


@tf.keras.utils.register_keras_serializable(package="ChessAI")
class TargetRangeMAE(tf.keras.metrics.Metric):
	def __init__(self, min_abs=None, max_abs=None, name="target_range_mae", **kwargs):
		super().__init__(name=name, **kwargs)
		self.min_abs = min_abs
		self.max_abs = max_abs
		self.total_error = self.add_weight(name="total_error", initializer="zeros")
		self.count = self.add_weight(name="count", initializer="zeros")

	def update_state(self, y_true, y_pred, sample_weight=None):
		y_true = tf.cast(y_true, self.dtype)
		y_pred = tf.cast(y_pred, self.dtype)
		target_abs = tf.abs(y_true)
		mask = tf.ones_like(target_abs, dtype=tf.bool)
		if self.min_abs is not None:
			mask = tf.logical_and(mask, target_abs > self.min_abs)
		if self.max_abs is not None:
			mask = tf.logical_and(mask, target_abs <= self.max_abs)

		mask = tf.cast(mask, self.dtype)
		error = tf.abs(y_true - y_pred) * mask
		self.total_error.assign_add(tf.reduce_sum(error))
		self.count.assign_add(tf.reduce_sum(mask))

	def result(self):
		return tf.math.divide_no_nan(self.total_error, self.count)

	def reset_state(self):
		self.total_error.assign(0.0)
		self.count.assign(0.0)

	def get_config(self):
		config = super().get_config()
		config.update({"min_abs": self.min_abs, "max_abs": self.max_abs})
		return config


def compile_model(model, learning_rate):
	model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
		loss=tf.keras.losses.Huber(delta=1.0),
		metrics=[
			tf.keras.metrics.MeanAbsoluteError(name="mae"),
			TargetRangeMAE(max_abs=1.0, name="mae_abs_le_1"),
			TargetRangeMAE(max_abs=3.0, name="mae_abs_le_3"),
			TargetRangeMAE(max_abs=5.0, name="mae_abs_le_5"),
			TargetRangeMAE(min_abs=5.0, name="mae_abs_gt_5"),
		],
	)
	return model


def build_mlp_model(learning_rate):
	inputs = tf.keras.Input(shape=(FEATURE_SIZE,), name="position")
	x = tf.keras.layers.Dense(
		1024,
		activation="relu",
		kernel_regularizer=tf.keras.regularizers.l2(1e-5),
	)(inputs)
	x = tf.keras.layers.Dropout(0.10)(x)
	x = tf.keras.layers.Dense(
		512,
		activation="relu",
		kernel_regularizer=tf.keras.regularizers.l2(1e-5),
	)(x)
	x = tf.keras.layers.Dropout(0.10)(x)
	x = tf.keras.layers.Dense(256, activation="relu")(x)
	outputs = tf.keras.layers.Dense(1, name="side_to_move_pawn_score")(x)

	model = tf.keras.Model(inputs=inputs, outputs=outputs, name="position_evaluator")
	return compile_model(model, learning_rate)


def residual_block(x, filters, name):
	shortcut = x
	x = tf.keras.layers.Conv2D(
		filters,
		kernel_size=3,
		padding="same",
		use_bias=False,
		kernel_regularizer=tf.keras.regularizers.l2(1e-5),
		name=f"{name}_conv_1",
	)(x)
	x = tf.keras.layers.BatchNormalization(name=f"{name}_bn_1")(x)
	x = tf.keras.layers.Activation("relu", name=f"{name}_relu_1")(x)
	x = tf.keras.layers.Conv2D(
		filters,
		kernel_size=3,
		padding="same",
		use_bias=False,
		kernel_regularizer=tf.keras.regularizers.l2(1e-5),
		name=f"{name}_conv_2",
	)(x)
	x = tf.keras.layers.BatchNormalization(name=f"{name}_bn_2")(x)

	if shortcut.shape[-1] != filters:
		shortcut = tf.keras.layers.Conv2D(
			filters,
			kernel_size=1,
			padding="same",
			use_bias=False,
			kernel_regularizer=tf.keras.regularizers.l2(1e-5),
			name=f"{name}_projection",
		)(shortcut)
		shortcut = tf.keras.layers.BatchNormalization(name=f"{name}_projection_bn")(shortcut)

	x = tf.keras.layers.Add(name=f"{name}_add")([shortcut, x])
	return tf.keras.layers.Activation("relu", name=f"{name}_relu_2")(x)


def build_cnn_model(learning_rate):
	inputs = tf.keras.Input(shape=(ATTACK_FEATURE_SIZE,), name="position")
	sequence = tf.keras.layers.Reshape(
		(ATTACK_FEATURE_SIZE, 1), name="feature_sequence"
	)(inputs)

	board = tf.keras.layers.Cropping1D(
		cropping=(0, METADATA_SIZE), name="board_feature_slice"
	)(sequence)
	board = tf.keras.layers.Reshape(
		(8, 8, PIECE_PLANE_COUNT + ATTACK_PLANE_COUNT), name="board_planes"
	)(board)

	metadata = tf.keras.layers.Cropping1D(
		cropping=(ATTACK_BOARD_FEATURE_SIZE, 0), name="metadata_feature_slice"
	)(sequence)
	metadata = tf.keras.layers.Flatten(name="metadata_flatten")(metadata)
	metadata = tf.keras.layers.Dense(64, activation="relu", name="metadata_dense")(metadata)

	x = tf.keras.layers.Conv2D(
		64,
		kernel_size=3,
		padding="same",
		use_bias=False,
		kernel_regularizer=tf.keras.regularizers.l2(1e-5),
		name="stem_conv",
	)(board)
	x = tf.keras.layers.BatchNormalization(name="stem_bn")(x)
	x = tf.keras.layers.Activation("relu", name="stem_relu")(x)

	x = residual_block(x, 64, "res_64_1")
	x = residual_block(x, 64, "res_64_2")
	x = residual_block(x, 128, "res_128_1")
	x = residual_block(x, 128, "res_128_2")

	spatial = tf.keras.layers.Flatten(name="spatial_flatten")(x)
	x = tf.keras.layers.Concatenate(name="spatial_metadata")([spatial, metadata])
	x = tf.keras.layers.Dense(
		512,
		activation="relu",
		kernel_regularizer=tf.keras.regularizers.l2(1e-5),
		name="value_dense_1",
	)(x)
	x = tf.keras.layers.Dropout(0.15, name="value_dropout")(x)
	x = tf.keras.layers.Dense(128, activation="relu", name="value_dense_2")(x)
	outputs = tf.keras.layers.Dense(1, name="side_to_move_pawn_score")(x)

	model = tf.keras.Model(
		inputs=inputs,
		outputs=outputs,
		name="position_evaluator_cnn_attacks",
	)
	return compile_model(model, learning_rate)


def build_model(architecture, learning_rate):
	if architecture == "cnn":
		return build_cnn_model(learning_rate)
	if architecture == "mlp":
		return build_mlp_model(learning_rate)
	raise ValueError(f"unsupported architecture: {architecture}")


def is_validation_fen(fen, validation_fraction, split_seed):
	if validation_fraction <= 0:
		return False
	key = f"{split_seed}\0{fen}".encode("utf-8")
	return zlib.crc32(key) / 2**32 < validation_fraction


def count_split_rows(paths, max_rows, validation_fraction, split_seed):
	row_counts = []
	train_counts = []
	validation_counts = []

	for path in paths:
		row_count = 0
		train_count = 0
		validation_count = 0

		with path.open(newline="") as file:
			reader = csv.DictReader(file)
			for row in reader:
				if max_rows is not None and row_count >= max_rows:
					break
				row_count += 1
				if is_validation_fen(row["FEN"], validation_fraction, split_seed):
					validation_count += 1
				else:
					train_count += 1

		row_counts.append(row_count)
		train_counts.append(train_count)
		validation_counts.append(validation_count)

	return row_counts, train_counts, validation_counts


def sample_weight_for_target(target, focus_loss_pawns, min_sample_weight):
	if focus_loss_pawns <= 0:
		return 1.0
	target_abs = abs(float(target))
	if target_abs <= focus_loss_pawns:
		return 1.0
	return max(min_sample_weight, focus_loss_pawns / target_abs)


def iter_examples(
	paths,
	row_counts,
	split,
	include_attack_maps,
	max_abs_pawns,
	validation_fraction,
	split_seed,
	focus_loss_pawns,
	min_sample_weight,
):
	for path, row_count in zip(paths, row_counts):
		with path.open(newline="") as file:
			reader = csv.DictReader(file)
			for row_index, row in enumerate(reader):
				if row_index >= row_count:
					break

				is_validation = is_validation_fen(
					row["FEN"],
					validation_fraction=validation_fraction,
					split_seed=split_seed,
				)
				if split == "train" and is_validation:
					continue
				if split == "validation" and not is_validation:
					continue

				features = encode_fen(
					row["FEN"], include_attack_maps=include_attack_maps
				)
				target = evaluation_to_pawns(row["Evaluation"], max_abs_pawns=max_abs_pawns)
				weight = sample_weight_for_target(
					target=target,
					focus_loss_pawns=focus_loss_pawns,
					min_sample_weight=min_sample_weight,
				)
				yield features, np.array([target], dtype=np.float32), np.float32(weight)


def make_dataset(paths, row_counts, split, args):
	include_attack_maps = args.architecture == "cnn"
	feature_size = ATTACK_FEATURE_SIZE if include_attack_maps else FEATURE_SIZE
	dataset = tf.data.Dataset.from_generator(
		lambda: iter_examples(
			paths=paths,
			row_counts=row_counts,
			split=split,
			include_attack_maps=include_attack_maps,
			max_abs_pawns=args.max_abs_pawns,
			validation_fraction=args.validation_fraction,
			split_seed=args.split_seed,
			focus_loss_pawns=args.focus_loss_pawns,
			min_sample_weight=args.min_sample_weight,
		),
		output_signature=(
			tf.TensorSpec(shape=(feature_size,), dtype=tf.float32),
			tf.TensorSpec(shape=(1,), dtype=tf.float32),
			tf.TensorSpec(shape=(), dtype=tf.float32),
		),
	)
	if split == "train" and args.shuffle_buffer > 0:
		dataset = dataset.shuffle(args.shuffle_buffer, reshuffle_each_iteration=True)
	if split == "train":
		dataset = dataset.repeat()
	return dataset.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)


def validate_args(args):
	if args.epochs < 1:
		raise ValueError("--epochs must be positive")
	if args.batch_size < 1:
		raise ValueError("--batch-size must be positive")
	if args.validation_fraction < 0 or args.validation_fraction >= 1:
		raise ValueError("--validation-fraction must be in [0, 1)")
	if args.max_abs_pawns <= 0:
		raise ValueError("--max-abs-pawns must be positive")
	if args.focus_loss_pawns < 0:
		raise ValueError("--focus-loss-pawns cannot be negative")
	if args.min_sample_weight <= 0 or args.min_sample_weight > 1:
		raise ValueError("--min-sample-weight must be in (0, 1]")
	if args.max_rows_per_file is not None and args.max_rows_per_file < 1:
		raise ValueError("--max-rows-per-file must be positive")
	if args.steps_per_epoch is not None and args.steps_per_epoch < 1:
		raise ValueError("--steps-per-epoch must be positive")
	if args.validation_steps is not None and args.validation_steps < 1:
		raise ValueError("--validation-steps must be positive")
	if args.weights_out is not None and not str(args.weights_out).endswith(".weights.h5"):
		raise ValueError("--weights-out must end with .weights.h5")

	for path in args.data_files:
		if not path.exists():
			raise FileNotFoundError(path)


def main():
	args = parse_args()
	resolve_output_paths(args)
	validate_args(args)
	configure_tensorflow()

	row_counts, train_counts, validation_counts = count_split_rows(
		paths=args.data_files,
		max_rows=args.max_rows_per_file,
		validation_fraction=args.validation_fraction,
		split_seed=args.split_seed,
	)
	train_rows = sum(train_counts)
	validation_rows = sum(validation_counts)

	if train_rows == 0:
		raise ValueError("no training rows available")

	print(f"Architecture: {args.architecture}")
	print(
		f"Input features: "
		f"{ATTACK_FEATURE_SIZE if args.architecture == 'cnn' else FEATURE_SIZE}"
	)
	print(f"Model checkpoint: {args.model_out}")
	print(f"Weights checkpoint: {args.weights_out}")
	print(f"Split: deterministic FEN hash ({args.validation_fraction:.1%} validation)")
	if args.focus_loss_pawns > 0:
		print(
			f"Loss weighting: full weight for |eval| <= {args.focus_loss_pawns:g} pawns, "
			f"minimum weight {args.min_sample_weight:g}"
		)
	else:
		print("Loss weighting: disabled")
	print("Training files:")
	for path, rows, train_count, validation_count in zip(
		args.data_files, row_counts, train_counts, validation_counts
	):
		print(f"  {path}: {rows} rows ({train_count} train, {validation_count} validation)")

	train_dataset = make_dataset(args.data_files, row_counts, "train", args)
	validation_dataset = None
	if validation_rows > 0:
		validation_dataset = make_dataset(args.data_files, row_counts, "validation", args)

	if args.dry_run:
		for features, targets, sample_weights in train_dataset.take(1):
			print(f"features: {features.shape} {features.dtype}")
			print(f"targets: {targets.shape} {targets.dtype}")
			print(f"sample_weights: {sample_weights.shape} {sample_weights.dtype}")
			print(f"target range: {targets.numpy().min():.3f} to {targets.numpy().max():.3f}")
			print(
				f"sample weight range: {sample_weights.numpy().min():.3f} "
				f"to {sample_weights.numpy().max():.3f}"
			)
		return

	model = build_model(args.architecture, args.learning_rate)
	args.model_out.parent.mkdir(parents=True, exist_ok=True)
	if args.weights_out is not None:
		args.weights_out.parent.mkdir(parents=True, exist_ok=True)

	steps_per_epoch = math.ceil(train_rows / args.batch_size)
	if args.steps_per_epoch is not None:
		steps_per_epoch = min(steps_per_epoch, args.steps_per_epoch)

	validation_steps = None
	if validation_rows > 0:
		validation_steps = math.ceil(validation_rows / args.batch_size)
		if args.validation_steps is not None:
			validation_steps = min(validation_steps, args.validation_steps)

	callbacks = [
		tf.keras.callbacks.ModelCheckpoint(
			filepath=str(args.model_out),
			monitor="val_loss" if validation_dataset is not None else "loss",
			save_best_only=True,
		),
	]
	if args.weights_out is not None:
		callbacks.append(
			tf.keras.callbacks.ModelCheckpoint(
				filepath=str(args.weights_out),
				monitor="val_loss" if validation_dataset is not None else "loss",
				save_best_only=True,
				save_weights_only=True,
			)
		)

	model.fit(
		train_dataset,
		epochs=args.epochs,
		steps_per_epoch=steps_per_epoch,
		validation_data=validation_dataset,
		validation_steps=validation_steps,
		callbacks=callbacks,
	)
	print(f"Saved best evaluator checkpoint to {args.model_out}")
	if args.weights_out is not None:
		print(f"Saved best weights checkpoint to {args.weights_out}")


if __name__ == "__main__":
	main()
