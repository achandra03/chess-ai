import argparse
import csv
import inspect
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf

from board_features import (
	BOARD_SIZE,
	PERSPECTIVE_V3_BOARD_FEATURE_SIZE,
	PERSPECTIVE_V3_FEATURE_SIZE,
	PERSPECTIVE_V3_METADATA_SIZE,
	encode_fen_perspective_v3,
	evaluation_to_pawns,
	hash_fraction,
	selected_for_sample,
	split_for_fen,
)
from encoded_cache import (
	MANIFEST_NAME as CACHE_MANIFEST_NAME,
	EncodedCache,
	unpack_features,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
AI_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_FILE = ROOT_DIR / "data" / "chessData_depth8.csv"
ARCHITECTURE = "perspective-transformer-v3"
DEFAULT_MODEL_OUT = AI_DIR / "position_evaluator_perspective_transformer_v3_mae.keras"
DEFAULT_WEIGHTS_OUT = AI_DIR / "position_evaluator_perspective_transformer_v3_mae.weights.h5"
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_MIN_LEARNING_RATE = 0.000001
DEFAULT_MAX_ABS_PAWNS = 10.0
DEFAULT_EPOCHS = 50
DEFAULT_SOURCE_ROWS = 12_958_035
DEFAULT_SAMPLE_SIZE = 0
DEFAULT_SPLIT_COUNTS_BY_SAMPLE = {
	3_000_000: {
		"train": 2_402_392,
		"validation": 300_013,
		"test": 300_099,
	},
}
DEFAULT_TRAIN_SPLIT_EVAL_EXAMPLES = 100_000


def default_cache_dir_for(data_file):
	return ROOT_DIR / "data" / "encoded" / f"{Path(data_file).stem}_perspective_v3"


def completed_epochs_from_history(log_path):
	log_path = Path(log_path)
	if not log_path.is_file():
		return 0
	last_epoch = -1
	with log_path.open(newline="") as file:
		for row in csv.DictReader(file):
			try:
				last_epoch = max(last_epoch, int(row["epoch"]))
			except (KeyError, TypeError, ValueError):
				continue
	return last_epoch + 1


def best_metric_from_history(log_path, metric):
	log_path = Path(log_path)
	if not log_path.is_file():
		return None
	best = None
	with log_path.open(newline="") as file:
		for row in csv.DictReader(file):
			try:
				value = float(row[metric])
			except (KeyError, TypeError, ValueError):
				continue
			if best is None or value < best:
				best = value
	return best


def parse_args():
	parser = argparse.ArgumentParser(
		description="Train the perspective-transformer-v3 position evaluator."
	)
	parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
	parser.add_argument(
		"--architecture",
		choices=(ARCHITECTURE,),
		default=ARCHITECTURE,
		help="Model architecture (only perspective-transformer-v3 remains).",
	)
	parser.add_argument("--model-out", type=Path, default=None)
	parser.add_argument("--weights-out", type=Path, default=None)
	parser.add_argument(
		"--initial-weights",
		type=Path,
		default=None,
		help="Optional .weights.h5 checkpoint to load before training.",
	)
	parser.add_argument(
		"--initial-epoch",
		type=int,
		default=0,
		help=(
			"Epoch to resume from (0-based). Offsets the warmup-cosine LR "
			"schedule and appends to the history CSV instead of overwriting."
		),
	)
	parser.add_argument(
		"--resume",
		action="store_true",
		help=(
			"Continue an interrupted run: load --weights-out when it exists "
			"and derive --initial-epoch from the history CSV. The checkpoint "
			"holds the best epoch's weights, not the last epoch's."
		),
	)
	parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
	parser.add_argument("--batch-size", type=int, default=512)
	parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
	parser.add_argument("--min-learning-rate", type=float, default=DEFAULT_MIN_LEARNING_RATE)
	parser.add_argument(
		"--lr-schedule",
		choices=("constant", "warmup-cosine"),
		default="warmup-cosine",
	)
	parser.add_argument(
		"--warmup-ratio",
		type=float,
		default=0.05,
		help="Fraction of total training steps spent warming up.",
	)
	parser.add_argument(
		"--warmup-epochs",
		type=float,
		default=2.0,
		help="Epochs spent warming up. Overrides --warmup-ratio unless --warmup-steps is set.",
	)
	parser.add_argument(
		"--warmup-steps",
		type=int,
		default=None,
		help="Exact warmup steps. Overrides --warmup-epochs and --warmup-ratio.",
	)
	parser.add_argument(
		"--optimizer",
		choices=("adamw", "adam"),
		default="adamw",
	)
	parser.add_argument("--weight-decay", type=float, default=1e-4)
	parser.add_argument(
		"--gradient-clipnorm",
		type=float,
		default=1.0,
		help="Global gradient norm clip. Set to 0 to disable.",
	)
	parser.add_argument("--transformer-d-model", type=int, default=384)
	parser.add_argument("--transformer-heads", type=int, default=8)
	parser.add_argument("--transformer-layers", type=int, default=6)
	parser.add_argument("--transformer-ff-dim", type=int, default=1536)
	parser.add_argument("--transformer-dropout", type=float, default=0.05)
	parser.add_argument(
		"--loss",
		choices=("mae", "huber"),
		default="huber",
		help="Training loss. MAE is always reported as a metric.",
	)
	parser.add_argument("--huber-delta", type=float, default=0.75)
	parser.add_argument(
		"--mixed-precision",
		choices=("auto", "on", "off"),
		default="auto",
		help="Use mixed_float16 on GPU. auto enables it when a GPU is present.",
	)
	parser.add_argument(
		"--extra-data-file",
		type=Path,
		action="append",
		default=None,
		help="Additional CSV used for training only. May be passed more than once.",
	)
	parser.add_argument(
		"--encoded-cache-dir",
		type=Path,
		default=None,
		help=(
			"Directory of pre-encoded positions built by encode_dataset.py. "
			"Auto-detected when present."
		),
	)
	parser.add_argument(
		"--no-encoded-cache",
		action="store_true",
		help="Stream from the CSV even if an encoded cache exists.",
	)
	parser.add_argument(
		"--extra-cache-dir",
		type=Path,
		action="append",
		default=None,
		help=(
			"Pre-encoded train-only extra dataset (encode_dataset.py "
			"--train-only). May be passed more than once."
		),
	)
	parser.add_argument(
		"--train-split-eval-examples",
		type=int,
		default=DEFAULT_TRAIN_SPLIT_EVAL_EXAMPLES,
		help=(
			"Fixed training-split subsample evaluated in inference mode after "
			"each epoch and logged as train_split_mae. Set to 0 to disable."
		),
	)
	parser.add_argument(
		"--mirror-augmentation",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="Train with horizontal mirror augmentation.",
	)
	parser.add_argument(
		"--sample-size",
		type=int,
		default=DEFAULT_SAMPLE_SIZE,
		help=(
			"Approximate number of positions selected deterministically from "
			"the source CSV. Set to 0 to use every row."
		),
	)
	parser.add_argument(
		"--scan-data",
		action="store_true",
		help="Count and split-scan the CSV instead of using known default counts.",
	)
	parser.add_argument("--sample-seed", default="sabatelli-dataset4-sample-v1")
	parser.add_argument("--split-seed", default="sabatelli-dataset4-split-v1")
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
	parser.add_argument("--steps-per-epoch", type=int, default=None)
	parser.add_argument("--validation-steps", type=int, default=None)
	parser.add_argument("--test-steps", type=int, default=None)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Inspect one batch and the model without training.",
	)
	parser.add_argument(
		"--target-mae",
		type=float,
		default=0.5,
		help="Stop after an epoch where train and validation MAE reach this value. 0 disables.",
	)
	parser.add_argument("--early-stopping-patience", type=int, default=18)
	parser.add_argument("--early-stopping-min-delta", type=float, default=0.001)
	parser.add_argument("--report-out", type=Path, default=None)
	parser.add_argument("--log-out", type=Path, default=None)
	args = parser.parse_args()
	if args.model_out is None:
		args.model_out = DEFAULT_MODEL_OUT
	if args.weights_out is None:
		args.weights_out = DEFAULT_WEIGHTS_OUT
	if args.report_out is None:
		args.report_out = args.model_out.with_suffix(".training_report.json")
	if args.log_out is None:
		args.log_out = args.model_out.with_suffix(".history.csv")
	if args.resume:
		if args.initial_weights is not None:
			parser.error("--resume cannot be combined with --initial-weights")
		if args.initial_epoch != 0:
			parser.error("--resume cannot be combined with --initial-epoch")
		if args.weights_out.is_file():
			args.initial_weights = args.weights_out
			args.initial_epoch = completed_epochs_from_history(args.log_out)
	if args.extra_data_file is None:
		args.extra_data_file = []
	args.extra_data_file = list(dict.fromkeys(path.resolve() for path in args.extra_data_file))
	if args.extra_cache_dir is None:
		args.extra_cache_dir = []
	if args.no_encoded_cache:
		args.encoded_cache_dir = None
	elif args.encoded_cache_dir is None:
		candidate = default_cache_dir_for(args.data_file)
		if (candidate / CACHE_MANIFEST_NAME).is_file():
			args.encoded_cache_dir = candidate
	return args


def configure_tensorflow(args):
	gpus = tf.config.list_physical_devices("GPU")
	for gpu in gpus:
		try:
			tf.config.experimental.set_memory_growth(gpu, True)
		except RuntimeError:
			pass
	use_mixed_precision = args.mixed_precision == "on" or (
		args.mixed_precision == "auto" and bool(gpus)
	)
	policy = "mixed_float16" if use_mixed_precision else "float32"
	tf.keras.mixed_precision.set_global_policy(policy)
	args.mixed_precision_policy = policy


def validate_args(args):
	if not args.data_file.is_file():
		raise FileNotFoundError(args.data_file)
	for extra_data_file in args.extra_data_file:
		if not extra_data_file.is_file():
			raise FileNotFoundError(extra_data_file)
	if args.encoded_cache_dir is not None:
		if not (args.encoded_cache_dir / CACHE_MANIFEST_NAME).is_file():
			raise FileNotFoundError(
				f"{args.encoded_cache_dir / CACHE_MANIFEST_NAME}; "
				"build the cache with encode_dataset.py"
			)
		if args.extra_data_file:
			raise ValueError(
				"CSV extra data cannot be mixed with an encoded cache; "
				"encode extras with encode_dataset.py --train-only and pass "
				"--extra-cache-dir"
			)
		for extra_cache_dir in args.extra_cache_dir:
			if not (extra_cache_dir / CACHE_MANIFEST_NAME).is_file():
				raise FileNotFoundError(extra_cache_dir / CACHE_MANIFEST_NAME)
	elif args.extra_cache_dir:
		raise ValueError("--extra-cache-dir requires an encoded cache")
	if args.train_split_eval_examples < 0:
		raise ValueError("--train-split-eval-examples cannot be negative")
	if args.initial_weights is not None and not args.initial_weights.is_file():
		raise FileNotFoundError(args.initial_weights)
	if args.initial_epoch < 0:
		raise ValueError("--initial-epoch cannot be negative")
	if args.epochs < 1:
		raise ValueError("--epochs must be positive")
	if args.batch_size < 1:
		raise ValueError("--batch-size must be positive")
	if args.learning_rate <= 0:
		raise ValueError("--learning-rate must be positive")
	if args.min_learning_rate < 0:
		raise ValueError("--min-learning-rate cannot be negative")
	if args.min_learning_rate > args.learning_rate:
		raise ValueError("--min-learning-rate cannot exceed --learning-rate")
	if args.warmup_ratio < 0 or args.warmup_ratio >= 1:
		raise ValueError("--warmup-ratio must be in [0, 1)")
	if args.warmup_epochs is not None and args.warmup_epochs < 0:
		raise ValueError("--warmup-epochs cannot be negative")
	if args.warmup_steps is not None and args.warmup_steps < 0:
		raise ValueError("--warmup-steps cannot be negative")
	if args.weight_decay < 0:
		raise ValueError("--weight-decay cannot be negative")
	if args.gradient_clipnorm < 0:
		raise ValueError("--gradient-clipnorm cannot be negative")
	if args.transformer_d_model < 1:
		raise ValueError("--transformer-d-model must be positive")
	if args.transformer_heads < 1:
		raise ValueError("--transformer-heads must be positive")
	if args.transformer_d_model % args.transformer_heads != 0:
		raise ValueError("--transformer-d-model must be divisible by --transformer-heads")
	if args.transformer_layers < 1:
		raise ValueError("--transformer-layers must be positive")
	if args.transformer_ff_dim < 1:
		raise ValueError("--transformer-ff-dim must be positive")
	if args.transformer_dropout < 0 or args.transformer_dropout >= 1:
		raise ValueError("--transformer-dropout must be in [0, 1)")
	if args.huber_delta <= 0:
		raise ValueError("--huber-delta must be positive")
	if args.sample_size < 0:
		raise ValueError("--sample-size cannot be negative")
	if args.max_abs_pawns <= 0:
		raise ValueError("--max-abs-pawns must be positive")
	if args.target_mae < 0:
		raise ValueError("--target-mae cannot be negative")
	if args.early_stopping_patience < 0:
		raise ValueError("--early-stopping-patience cannot be negative")
	if args.early_stopping_min_delta < 0:
		raise ValueError("--early-stopping-min-delta cannot be negative")
	if args.shuffle_buffer < 0:
		raise ValueError("--shuffle-buffer cannot be negative")
	if args.max_rows is not None and args.max_rows < 1:
		raise ValueError("--max-rows must be positive")
	if args.steps_per_epoch is not None and args.steps_per_epoch < 1:
		raise ValueError("--steps-per-epoch must be positive")
	for name in ("validation_steps", "test_steps"):
		value = getattr(args, name)
		if value is not None and value < 0:
			raise ValueError(f"--{name.replace('_', '-')} cannot be negative")
	if not str(args.weights_out).endswith(".weights.h5"):
		raise ValueError("--weights-out must end with .weights.h5")


@tf.keras.utils.register_keras_serializable(package="ChessAI")
class SquarePositionEmbedding(tf.keras.layers.Layer):
	"""Add a learned absolute embedding for each of the 64 board squares."""

	def __init__(self, square_count=BOARD_SIZE * BOARD_SIZE, **kwargs):
		super().__init__(**kwargs)
		self.square_count = square_count

	def build(self, input_shape):
		self.position_embeddings = self.add_weight(
			name="position_embeddings",
			shape=(self.square_count, int(input_shape[-1])),
			initializer="random_normal",
			trainable=True,
		)

	def call(self, inputs):
		return inputs + self.position_embeddings[tf.newaxis, :, :]

	def get_config(self):
		config = super().get_config()
		config.update({"square_count": self.square_count})
		return config


def transformer_encoder_block(
	x,
	d_model,
	heads,
	ff_dim,
	dropout,
	block_index,
):
	attention_input = tf.keras.layers.LayerNormalization(
		epsilon=1e-6, name=f"transformer_{block_index}_attention_norm"
	)(x)
	attention_output = tf.keras.layers.MultiHeadAttention(
		num_heads=heads,
		key_dim=d_model // heads,
		dropout=dropout,
		name=f"transformer_{block_index}_attention",
	)(attention_input, attention_input)
	attention_output = tf.keras.layers.Dropout(
		dropout, name=f"transformer_{block_index}_attention_dropout"
	)(attention_output)
	x = tf.keras.layers.Add(name=f"transformer_{block_index}_attention_residual")(
		[x, attention_output]
	)

	ffn_input = tf.keras.layers.LayerNormalization(
		epsilon=1e-6, name=f"transformer_{block_index}_ffn_norm"
	)(x)
	ffn_output = tf.keras.layers.Dense(
		ff_dim, activation="gelu", name=f"transformer_{block_index}_ffn_expand"
	)(ffn_input)
	ffn_output = tf.keras.layers.Dropout(
		dropout, name=f"transformer_{block_index}_ffn_dropout_1"
	)(ffn_output)
	ffn_output = tf.keras.layers.Dense(
		d_model, name=f"transformer_{block_index}_ffn_project"
	)(ffn_output)
	ffn_output = tf.keras.layers.Dropout(
		dropout, name=f"transformer_{block_index}_ffn_dropout_2"
	)(ffn_output)
	return tf.keras.layers.Add(name=f"transformer_{block_index}_ffn_residual")(
		[x, ffn_output]
	)


def build_perspective_transformer_v3_model(
	d_model=384,
	heads=8,
	layers=6,
	ff_dim=1536,
	dropout=0.05,
):
	board_feature_size = PERSPECTIVE_V3_BOARD_FEATURE_SIZE
	inputs = tf.keras.Input(
		shape=(PERSPECTIVE_V3_FEATURE_SIZE,), name="position_features"
	)
	board_inputs = inputs[:, :board_feature_size]
	x = tf.keras.layers.Reshape(
		(64, board_feature_size // 64), name="square_feature_planes"
	)(board_inputs)
	x = tf.keras.layers.Dense(d_model, name="square_projection")(x)
	x = SquarePositionEmbedding(name="square_position_embedding")(x)
	x = tf.keras.layers.Dropout(dropout, name="input_dropout")(x)

	for block_index in range(1, layers + 1):
		x = transformer_encoder_block(
			x=x,
			d_model=d_model,
			heads=heads,
			ff_dim=ff_dim,
			dropout=dropout,
			block_index=block_index,
		)

	x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="final_norm")(x)
	mean_pool = tf.keras.layers.GlobalAveragePooling1D(name="square_mean_pool")(x)
	max_pool = tf.keras.layers.GlobalMaxPooling1D(name="square_max_pool")(x)

	metadata_inputs = inputs[:, board_feature_size:]
	metadata = tf.keras.layers.LayerNormalization(
		epsilon=1e-6, name="metadata_norm"
	)(metadata_inputs)
	metadata = tf.keras.layers.Dense(
		max(32, d_model // 2),
		activation="gelu",
		name="metadata_projection",
	)(metadata)

	x = tf.keras.layers.Concatenate(name="pooled_features")(
		[mean_pool, max_pool, metadata]
	)
	x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="head_norm")(x)
	x = tf.keras.layers.Dense(
		ff_dim, activation="gelu", name="evaluation_head_hidden_1"
	)(x)
	x = tf.keras.layers.Dropout(dropout, name="evaluation_head_dropout_1")(x)
	x = tf.keras.layers.Dense(
		max(64, ff_dim // 2),
		activation="gelu",
		name="evaluation_head_hidden_2",
	)(x)
	x = tf.keras.layers.Dropout(dropout, name="evaluation_head_dropout_2")(x)
	outputs = tf.keras.layers.Dense(
		1,
		dtype="float32",
		name="side_to_move_pawn_score",
	)(x)
	return tf.keras.Model(
		inputs=inputs, outputs=outputs, name="perspective_transformer_v3"
	)


@tf.keras.utils.register_keras_serializable(package="ChessAI")
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
	def __init__(
		self,
		initial_learning_rate,
		decay_steps,
		warmup_steps=0,
		min_learning_rate=0.0,
		step_offset=0,
		name=None,
	):
		super().__init__()
		self.initial_learning_rate = float(initial_learning_rate)
		self.decay_steps = max(1, int(decay_steps))
		self.warmup_steps = max(0, int(warmup_steps))
		self.min_learning_rate = float(min_learning_rate)
		self.step_offset = max(0, int(step_offset))
		self.name = name

	def __call__(self, step):
		with tf.name_scope(self.name or "WarmupCosineDecay"):
			step = tf.cast(step, tf.float32) + tf.cast(
				self.step_offset, tf.float32
			)
			initial_lr = tf.cast(self.initial_learning_rate, tf.float32)
			min_lr = tf.cast(self.min_learning_rate, tf.float32)
			lr_range = initial_lr - min_lr

			warmup_steps = tf.cast(self.warmup_steps, tf.float32)
			warmup_denominator = tf.maximum(warmup_steps, 1.0)
			warmup_progress = tf.minimum((step + 1.0) / warmup_denominator, 1.0)
			warmup_lr = min_lr + lr_range * warmup_progress

			cosine_steps = tf.maximum(step - warmup_steps, 0.0)
			cosine_total = tf.maximum(
				tf.cast(self.decay_steps - self.warmup_steps, tf.float32),
				1.0,
			)
			cosine_progress = tf.minimum(cosine_steps / cosine_total, 1.0)
			cosine_decay = 0.5 * (
				1.0 + tf.cos(tf.constant(math.pi, dtype=tf.float32) * cosine_progress)
			)
			cosine_lr = min_lr + lr_range * cosine_decay

			return tf.where(step < warmup_steps, warmup_lr, cosine_lr)

	def get_config(self):
		return {
			"initial_learning_rate": self.initial_learning_rate,
			"decay_steps": self.decay_steps,
			"warmup_steps": self.warmup_steps,
			"min_learning_rate": self.min_learning_rate,
			"step_offset": self.step_offset,
			"name": self.name,
		}


def warmup_steps_for_training(args):
	total_steps = getattr(args, "total_train_steps", None)
	steps_per_epoch = getattr(args, "train_steps_per_epoch", None)
	if total_steps is None or total_steps < 1:
		return 0
	if args.warmup_steps is not None:
		return min(args.warmup_steps, max(total_steps - 1, 0))
	if args.warmup_epochs is not None and steps_per_epoch is not None:
		return min(int(round(args.warmup_epochs * steps_per_epoch)), max(total_steps - 1, 0))
	return min(int(round(args.warmup_ratio * total_steps)), max(total_steps - 1, 0))


def learning_rate_for_optimizer(args):
	if args.lr_schedule == "constant":
		return args.learning_rate

	total_steps = getattr(args, "total_train_steps", None)
	if total_steps is None or total_steps < 1:
		return args.learning_rate

	return WarmupCosineDecay(
		initial_learning_rate=args.learning_rate,
		decay_steps=total_steps,
		warmup_steps=warmup_steps_for_training(args),
		min_learning_rate=args.min_learning_rate,
		step_offset=getattr(args, "lr_schedule_step_offset", 0),
	)


def build_optimizer(args):
	learning_rate = learning_rate_for_optimizer(args)
	clip_kwargs = {}
	if args.gradient_clipnorm > 0:
		clip_kwargs["clipnorm"] = args.gradient_clipnorm
	if args.optimizer == "adamw":
		adamw_optimizer = getattr(tf.keras.optimizers, "AdamW", None)
		if adamw_optimizer is None:
			experimental_optimizers = getattr(
				tf.keras.optimizers, "experimental", None
			)
			adamw_optimizer = getattr(experimental_optimizers, "AdamW", None)
		if adamw_optimizer is None:
			raise RuntimeError(
				"AdamW is not available in this TensorFlow/Keras install. "
				"Run with --optimizer adam or install TensorFlow 2.10+."
			)
		optimizer_kwargs = {
			"learning_rate": learning_rate,
			"weight_decay": args.weight_decay,
			**clip_kwargs,
		}
		if "jit_compile" in inspect.signature(adamw_optimizer).parameters:
			optimizer_kwargs["jit_compile"] = False
		return adamw_optimizer(**optimizer_kwargs)
	return tf.keras.optimizers.Adam(learning_rate=learning_rate, **clip_kwargs)


def learning_rate_description(args):
	if args.lr_schedule == "constant":
		return f"lr={args.learning_rate:g}"
	total_steps = getattr(args, "total_train_steps", None)
	if total_steps is None:
		return f"lr={args.learning_rate:g}->schedule pending"
	step_offset = getattr(args, "lr_schedule_step_offset", 0)
	offset_description = f", step_offset={step_offset}" if step_offset else ""
	return (
		f"lr={args.learning_rate:g}->{args.min_learning_rate:g} "
		f"warmup-cosine(total_steps={total_steps}, "
		f"warmup_steps={warmup_steps_for_training(args)}{offset_description})"
	)


def optimizer_description(args):
	clip_description = ""
	if args.gradient_clipnorm > 0:
		clip_description = f", clipnorm={args.gradient_clipnorm:g}"
	if args.optimizer == "adamw":
		return (
			f"AdamW({learning_rate_description(args)}, "
			f"weight_decay={args.weight_decay:g}{clip_description})"
		)
	return f"Adam({learning_rate_description(args)}{clip_description})"


def loss_for_training(args):
	if args.loss == "huber":
		return tf.keras.losses.Huber(delta=args.huber_delta)
	return tf.keras.losses.MeanAbsoluteError()


def loss_description(args):
	if args.loss == "huber":
		return f"Huber(delta={args.huber_delta:g})"
	return "MAE"


class TrainSplitMae(tf.keras.callbacks.Callback):
	"""Log inference-mode MAE on a fixed training-split subsample.

	The metric Keras reports during training is computed with dropout
	active; this callback gives a train-split number directly comparable
	to val_mae.
	"""

	def __init__(self, dataset):
		super().__init__()
		self.dataset = dataset

	def on_epoch_end(self, epoch, logs=None):
		if logs is None:
			return
		results = self.model.evaluate(
			self.dataset, verbose=0, return_dict=True
		)
		logs["train_split_mae"] = float(results["mae"])


class TargetMaeStop(tf.keras.callbacks.Callback):
	def __init__(self, target_mae, require_validation):
		super().__init__()
		self.target_mae = target_mae
		self.require_validation = require_validation

	def on_epoch_end(self, epoch, logs=None):
		if self.target_mae <= 0:
			return
		logs = logs or {}
		train_metric = "train_split_mae" if "train_split_mae" in logs else "mae"
		train_mae = logs.get(train_metric)
		val_mae = logs.get("val_mae")
		if train_mae is None:
			return
		if train_mae > self.target_mae:
			return
		if self.require_validation and (
			val_mae is None or val_mae > self.target_mae
		):
			return

		metric_text = f"{train_metric}={train_mae:.6f}"
		if val_mae is not None:
			metric_text += f", val_mae={val_mae:.6f}"
		print(
			f"\nTarget MAE reached after epoch {epoch + 1}: {metric_text}",
			flush=True,
		)
		self.model.stop_training = True


def build_model(args):
	model = build_perspective_transformer_v3_model(
		d_model=args.transformer_d_model,
		heads=args.transformer_heads,
		layers=args.transformer_layers,
		ff_dim=args.transformer_ff_dim,
		dropout=args.transformer_dropout,
	)
	model.compile(
		optimizer=build_optimizer(args),
		loss=loss_for_training(args),
		metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
	)
	return model


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


def count_extra_rows(args):
	return {
		path: count_source_rows(path, args.max_rows)
		for path in args.extra_data_file
	}


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
	augment_mirror=False,
	train_only=False,
):
	with path.open(newline="") as file:
		reader = csv.DictReader(file)
		validate_csv_header(reader.fieldnames)
		for row_index, row in enumerate(reader):
			if row_index >= raw_rows:
				break

			fen = row["FEN"]
			if not train_only:
				if not selected_for_sample(fen, sample_fraction, sample_seed):
					continue
				if split_for_fen(fen, split_seed) != split:
					continue

			target = evaluation_to_pawns(
				row["Evaluation"],
				max_abs_pawns=max_abs_pawns,
			)
			target_array = np.array([target], dtype=np.float32)
			yield encode_fen_perspective_v3(fen), target_array
			if augment_mirror:
				yield encode_fen_perspective_v3(fen, mirror_files=True), target_array


def make_example_dataset(
	args,
	path,
	raw_rows,
	split,
	sample_fraction,
	augment_mirror=False,
	train_only=False,
):
	return tf.data.Dataset.from_generator(
		lambda: iter_examples(
			path=path,
			raw_rows=raw_rows,
			split=split,
			sample_fraction=sample_fraction,
			sample_seed=args.sample_seed,
			split_seed=args.split_seed,
			max_abs_pawns=args.max_abs_pawns,
			augment_mirror=augment_mirror,
			train_only=train_only,
		),
		output_signature=(
			tf.TensorSpec(shape=(PERSPECTIVE_V3_FEATURE_SIZE,), dtype=tf.float32),
			tf.TensorSpec(shape=(1,), dtype=tf.float32),
		),
	)


def make_train_dataset(args, raw_rows, sample_fraction, extra_row_counts):
	source_datasets = []
	source_weights = []
	mirror_factor = 2 if args.mirror_augmentation else 1

	def add_source(dataset, example_count, seed_value):
		if args.shuffle_buffer > 0:
			dataset = dataset.shuffle(
				args.shuffle_buffer,
				seed=int(hash_fraction(args.split_seed, seed_value) * (2**31 - 1)),
				reshuffle_each_iteration=True,
			)
		source_datasets.append(dataset.repeat())
		source_weights.append(float(example_count))

	add_source(
		make_example_dataset(
			args=args,
			path=args.data_file,
			raw_rows=raw_rows,
			split="train",
			sample_fraction=sample_fraction,
			augment_mirror=args.mirror_augmentation,
		),
		example_count=args.primary_train_rows * mirror_factor,
		seed_value="shuffle:primary",
	)

	for source_index, (path, extra_raw_rows) in enumerate(extra_row_counts.items()):
		add_source(
			make_example_dataset(
				args=args,
				path=path,
				raw_rows=extra_raw_rows,
				split="train",
				sample_fraction=1.0,
				augment_mirror=args.mirror_augmentation,
				train_only=True,
			),
			example_count=extra_raw_rows * mirror_factor,
			seed_value=f"shuffle:extra:{source_index}:{path.name}",
		)

	if len(source_datasets) == 1:
		dataset = source_datasets[0]
	else:
		weight_total = sum(source_weights)
		dataset = tf.data.Dataset.sample_from_datasets(
			source_datasets,
			weights=[weight / weight_total for weight in source_weights],
			seed=int(hash_fraction(args.split_seed, "source-mix") * (2**31 - 1)),
		)
	return dataset.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)


def make_dataset(args, raw_rows, split, sample_fraction, repeat=False):
	dataset = make_example_dataset(
		args=args,
		path=args.data_file,
		raw_rows=raw_rows,
		split=split,
		sample_fraction=sample_fraction,
	)
	if repeat:
		dataset = dataset.repeat()
	return dataset.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)


def make_cached_batch_fetcher(cache, max_abs_pawns):
	boards = cache.boards
	metadata = cache.metadata
	targets = cache.targets
	clip = float(max_abs_pawns) if max_abs_pawns < cache.max_abs_pawns else None

	def fetch(batch_indices, mirror_flags):
		batch_indices = np.asarray(batch_indices)
		batch_targets = targets[batch_indices].astype(np.float32)
		if clip is not None:
			batch_targets = np.clip(batch_targets, -clip, clip)
		features = unpack_features(
			boards[batch_indices],
			metadata[batch_indices],
			mirror_mask=np.asarray(mirror_flags, dtype=bool),
		)
		return features, batch_targets[:, np.newaxis]

	return fetch


def make_cached_dataset(
	cache,
	indices,
	args,
	shuffle=False,
	repeat=False,
	mirror=False,
	seed=0,
):
	fetch = make_cached_batch_fetcher(cache, args.max_abs_pawns)
	dataset = tf.data.Dataset.from_tensor_slices(np.asarray(indices, dtype=np.int64))
	if shuffle:
		dataset = dataset.shuffle(
			len(indices), seed=seed, reshuffle_each_iteration=True
		)
	if repeat:
		dataset = dataset.repeat()
	dataset = dataset.batch(args.batch_size)
	if mirror:
		dataset = dataset.map(
			lambda batch: (batch, tf.random.uniform(tf.shape(batch)) < 0.5)
		)
	else:
		dataset = dataset.map(
			lambda batch: (batch, tf.zeros_like(batch, dtype=tf.bool))
		)

	def load(batch_indices, mirror_flags):
		features, batch_targets = tf.numpy_function(
			fetch, [batch_indices, mirror_flags], (tf.float32, tf.float32)
		)
		features.set_shape((None, PERSPECTIVE_V3_FEATURE_SIZE))
		batch_targets.set_shape((None, 1))
		return features, batch_targets

	return dataset.map(load, num_parallel_calls=tf.data.AUTOTUNE)


def make_cached_train_dataset(args, cache, train_indices, extra_caches):
	sources = []
	weights = []

	def add_source(dataset, example_count):
		sources.append(dataset)
		weights.append(float(example_count))

	def seed_for(value):
		return int(hash_fraction(args.split_seed, value) * (2**31 - 1))

	add_source(
		make_cached_dataset(
			cache=cache,
			indices=train_indices,
			args=args,
			shuffle=args.shuffle_buffer > 0,
			repeat=True,
			mirror=args.mirror_augmentation,
			seed=seed_for("shuffle:primary"),
		),
		example_count=len(train_indices),
	)
	for source_index, (extra_cache, extra_indices) in enumerate(extra_caches):
		add_source(
			make_cached_dataset(
				cache=extra_cache,
				indices=extra_indices,
				args=args,
				shuffle=args.shuffle_buffer > 0,
				repeat=True,
				mirror=args.mirror_augmentation,
				seed=seed_for(f"shuffle:extra:{source_index}"),
			),
			example_count=len(extra_indices),
		)

	if len(sources) == 1:
		dataset = sources[0]
	else:
		# Sources are already batched, so mixing happens per batch here
		# rather than per example as on the CSV path.
		weight_total = sum(weights)
		dataset = tf.data.Dataset.sample_from_datasets(
			sources,
			weights=[weight / weight_total for weight in weights],
			seed=seed_for("source-mix"),
		)
	return dataset.prefetch(tf.data.AUTOTUNE)


def capped_steps(row_count, batch_size, requested_steps):
	steps = math.ceil(row_count / batch_size)
	if requested_steps is not None:
		steps = min(steps, requested_steps)
	return steps


def train_example_count(split_counts, extra_row_counts, mirror_augmentation):
	count = split_counts["train"] + sum(extra_row_counts.values())
	if mirror_augmentation:
		count *= 2
	return count


def can_use_default_split_counts(args):
	return (
		not args.scan_data
		and args.max_rows is None
		and args.data_file.resolve() == DEFAULT_DATA_FILE.resolve()
		and args.sample_size in DEFAULT_SPLIT_COUNTS_BY_SAMPLE
	)


def best_epoch_report(history):
	history_values = getattr(history, "history", {})
	monitor = "val_mae" if "val_mae" in history_values else "mae"
	values = history_values.get(monitor, [])
	if not values:
		return {}

	best_index = int(np.argmin(values))
	report = {
		"epoch": best_index + 1,
		"monitor": monitor,
		monitor: float(values[best_index]),
	}
	for name, metric_values in history_values.items():
		if best_index < len(metric_values):
			report[name] = float(metric_values[best_index])
	return report


def final_epoch_report(history):
	history_values = getattr(history, "history", {})
	report = {}
	for name, values in history_values.items():
		if values:
			report[name] = float(values[-1])
	return report


def empty_bucket_metric():
	return {"count": 0, "mae_sum": 0.0}


def add_bucket_error(buckets, name, error):
	bucket = buckets[name]
	bucket["count"] += 1
	bucket["mae_sum"] += float(error)


def finalize_bucket_metrics(buckets):
	return {
		name: {
			"count": int(values["count"]),
			"mae": (
				float(values["mae_sum"] / values["count"])
				if values["count"] > 0
				else None
			),
		}
		for name, values in buckets.items()
	}


def target_bucket_name(target):
	abs_target = abs(target)
	if abs_target < 1.0:
		return "abs_eval_lt_1"
	if abs_target < 3.0:
		return "abs_eval_1_to_3"
	if abs_target < 6.0:
		return "abs_eval_3_to_6"
	return "abs_eval_gt_6"


def evaluate_validation_buckets(model, args, raw_rows, sample_fraction, steps):
	if steps == 0:
		return {}

	buckets = {
		"abs_eval_lt_1": empty_bucket_metric(),
		"abs_eval_1_to_3": empty_bucket_metric(),
		"abs_eval_3_to_6": empty_bucket_metric(),
		"abs_eval_gt_6": empty_bucket_metric(),
		"mates": empty_bucket_metric(),
		"clipped": empty_bucket_metric(),
	}
	max_examples = steps * args.batch_size
	features_batch = []
	targets_batch = []
	flags_batch = []
	seen = 0

	def flush_batch():
		if not features_batch:
			return
		features = np.stack(features_batch).astype(np.float32)
		targets = np.asarray(targets_batch, dtype=np.float32)
		predictions = np.asarray(model.predict_on_batch(features)).reshape(-1)
		for prediction, target, flags in zip(predictions, targets, flags_batch):
			error = abs(float(prediction) - float(target))
			add_bucket_error(buckets, target_bucket_name(float(target)), error)
			if flags["mate"]:
				add_bucket_error(buckets, "mates", error)
			if flags["clipped"]:
				add_bucket_error(buckets, "clipped", error)
		features_batch.clear()
		targets_batch.clear()
		flags_batch.clear()

	with args.data_file.open(newline="") as file:
		reader = csv.DictReader(file)
		validate_csv_header(reader.fieldnames)
		for row_index, row in enumerate(reader):
			if row_index >= raw_rows or seen >= max_examples:
				break

			fen = row["FEN"]
			if not selected_for_sample(fen, sample_fraction, args.sample_seed):
				continue
			if split_for_fen(fen, args.split_seed) != "validation":
				continue

			target = evaluation_to_pawns(
				row["Evaluation"],
				max_abs_pawns=args.max_abs_pawns,
			)
			features_batch.append(encode_fen_perspective_v3(fen))
			targets_batch.append(target)
			flags_batch.append(
				{
					"mate": str(row["Evaluation"]).strip().startswith("#"),
					"clipped": abs(target) >= args.max_abs_pawns,
				}
			)
			seen += 1
			if len(features_batch) >= args.batch_size:
				flush_batch()

	flush_batch()
	return finalize_bucket_metrics(buckets)


def evaluate_validation_buckets_cached(model, cache, indices, args, steps):
	# Mate rows are not distinguishable from other clipped rows in the
	# cache, so only the clipped bucket is reported.
	if steps == 0 or len(indices) == 0:
		return {}

	buckets = {
		"abs_eval_lt_1": empty_bucket_metric(),
		"abs_eval_1_to_3": empty_bucket_metric(),
		"abs_eval_3_to_6": empty_bucket_metric(),
		"abs_eval_gt_6": empty_bucket_metric(),
		"clipped": empty_bucket_metric(),
	}
	clip = min(args.max_abs_pawns, cache.max_abs_pawns)
	max_examples = min(len(indices), steps * args.batch_size)
	for start in range(0, max_examples, args.batch_size):
		batch_indices = indices[start:start + args.batch_size]
		features = unpack_features(
			cache.boards[batch_indices],
			cache.metadata[batch_indices],
		)
		targets = cache.targets[batch_indices].astype(np.float32)
		targets = np.clip(targets, -clip, clip)
		predictions = np.asarray(model.predict_on_batch(features)).reshape(-1)
		for prediction, target in zip(predictions, targets):
			error = abs(float(prediction) - float(target))
			add_bucket_error(buckets, target_bucket_name(float(target)), error)
			if abs(float(target)) >= clip:
				add_bucket_error(buckets, "clipped", error)
	return finalize_bucket_metrics(buckets)


def make_generator_train_eval_dataset(args, raw_rows, sample_fraction):
	# The CSV path has no random access, so the first examples of the
	# training split are encoded once up front and kept in memory.
	count = args.train_split_eval_examples
	if count == 0:
		return None
	features = np.empty((count, PERSPECTIVE_V3_FEATURE_SIZE), dtype=np.float16)
	targets = np.empty((count, 1), dtype=np.float32)
	collected = 0
	for example_features, example_target in iter_examples(
		path=args.data_file,
		raw_rows=raw_rows,
		split="train",
		sample_fraction=sample_fraction,
		sample_seed=args.sample_seed,
		split_seed=args.split_seed,
		max_abs_pawns=args.max_abs_pawns,
	):
		features[collected] = example_features
		targets[collected] = example_target
		collected += 1
		if collected >= count:
			break
	if collected == 0:
		return None
	features = features[:collected]
	targets = targets[:collected]
	dataset = tf.data.Dataset.from_tensor_slices((features, targets))
	dataset = dataset.batch(args.batch_size)
	return dataset.map(
		lambda batch_features, batch_targets: (
			tf.cast(batch_features, tf.float32),
			batch_targets,
		)
	).prefetch(tf.data.AUTOTUNE)


def write_training_report(
	args,
	split_counts,
	extra_row_counts,
	history,
	test_results,
	validation_bucket_metrics,
	model_params,
	train_split_results=None,
):
	report = {
		"architecture": ARCHITECTURE,
		"data_file": str(args.data_file),
		"encoded_cache_dir": (
			str(args.encoded_cache_dir)
			if args.encoded_cache_dir is not None
			else None
		),
		"extra_data_files": [str(path) for path in args.extra_data_file],
		"extra_cache_dirs": [str(path) for path in args.extra_cache_dir],
		"train_split_eval_examples": int(args.train_split_eval_examples),
		"sample_size": args.sample_size,
		"split_counts": {name: int(value) for name, value in split_counts.items()},
		"extra_train_rows": {
			str(path): int(row_count)
			for path, row_count in extra_row_counts.items()
		},
		"input": (
			f"{PERSPECTIVE_V3_FEATURE_SIZE} values: "
			f"{PERSPECTIVE_V3_BOARD_FEATURE_SIZE} side-to-move perspective "
			f"piece/attack/tactical board values and "
			f"{PERSPECTIVE_V3_METADATA_SIZE} metadata values"
		),
		"target": {
			"units": "pawns",
			"clip": [-float(args.max_abs_pawns), float(args.max_abs_pawns)],
		},
		"loss": loss_description(args),
		"optimizer": optimizer_description(args),
		"batch_size": int(args.batch_size),
		"mixed_precision": args.mixed_precision_policy,
		"gradient_clipnorm": float(args.gradient_clipnorm),
		"mirror_augmentation": bool(args.mirror_augmentation),
		"epochs_requested": int(args.epochs),
		"initial_epoch": int(args.initial_epoch),
		"steps_per_epoch": int(args.train_steps_per_epoch),
		"total_train_steps": int(args.total_train_steps),
		"target_mae": float(args.target_mae),
		"early_stopping": {
			"patience": int(args.early_stopping_patience),
			"min_delta": float(args.early_stopping_min_delta),
		},
		"model_params": int(model_params),
		"best_epoch": best_epoch_report(history),
		"final_epoch": final_epoch_report(history),
		"best_weights_train_split_metrics": (
			{name: float(value) for name, value in train_split_results.items()}
			if train_split_results
			else None
		),
		"test_metrics": {
			name: float(value) for name, value in test_results.items()
		},
		"validation_bucket_metrics": validation_bucket_metrics,
		"model_out": str(args.model_out),
		"weights_out": str(args.weights_out),
		"initial_weights": (
			str(args.initial_weights) if args.initial_weights is not None else None
		),
		"log_out": str(args.log_out),
	}
	args.report_out.parent.mkdir(parents=True, exist_ok=True)
	with args.report_out.open("w", encoding="utf-8") as file:
		json.dump(report, file, indent=2)
		file.write("\n")


def prepare_cached_data(args):
	cache = EncodedCache(args.encoded_cache_dir, architecture=ARCHITECTURE)
	if cache.train_only:
		raise ValueError(
			"the primary encoded cache must contain train/validation/test "
			"splits; it was built with --train-only"
		)
	if args.split_seed != cache.split_seed:
		raise ValueError(
			f"cache split seed {cache.split_seed!r} does not match "
			f"--split-seed {args.split_seed!r}"
		)
	if args.max_abs_pawns > cache.max_abs_pawns:
		raise ValueError(
			f"cache targets are clipped to +/-{cache.max_abs_pawns:g} pawns; "
			f"--max-abs-pawns {args.max_abs_pawns:g} needs a rebuilt cache"
		)
	if args.sample_size > 0 and args.sample_seed != cache.sample_seed:
		raise ValueError(
			f"cache sample seed {cache.sample_seed!r} does not match "
			f"--sample-seed {args.sample_seed!r}"
		)

	sample_fraction = sample_fraction_for(cache.rows, args.sample_size)
	train_indices = cache.split_indices("train", sample_fraction)
	validation_indices = cache.split_indices("validation", sample_fraction)
	test_indices = cache.split_indices("test", sample_fraction)
	if len(train_indices) == 0:
		raise ValueError("deterministic sample contains no training rows")
	split_counts = {
		"train": len(train_indices),
		"validation": len(validation_indices),
		"test": len(test_indices),
	}

	extra_caches = []
	extra_row_counts = {}
	for path in args.extra_cache_dir:
		extra_cache = EncodedCache(path, architecture=ARCHITECTURE)
		if args.max_abs_pawns > extra_cache.max_abs_pawns:
			raise ValueError(
				f"extra cache {path} is clipped to "
				f"+/-{extra_cache.max_abs_pawns:g} pawns"
			)
		extra_indices = extra_cache.split_indices("train")
		extra_caches.append((extra_cache, extra_indices))
		extra_row_counts[path] = len(extra_indices)

	examples_per_epoch = len(train_indices) + sum(
		len(extra_indices) for _, extra_indices in extra_caches
	)

	validation_dataset = None
	if len(validation_indices) > 0:
		validation_dataset = make_cached_dataset(
			cache, validation_indices, args
		).prefetch(tf.data.AUTOTUNE)
	test_dataset = None
	if len(test_indices) > 0:
		test_dataset = make_cached_dataset(
			cache, test_indices, args
		).prefetch(tf.data.AUTOTUNE)

	train_eval_dataset = None
	if args.train_split_eval_examples > 0:
		rng = np.random.default_rng(
			int(hash_fraction(args.split_seed, "train-split-eval") * (2**31 - 1))
		)
		eval_count = min(args.train_split_eval_examples, len(train_indices))
		eval_indices = np.sort(
			rng.choice(train_indices, size=eval_count, replace=False)
		)
		train_eval_dataset = make_cached_dataset(
			cache, eval_indices, args
		).prefetch(tf.data.AUTOTUNE)

	def bucket_evaluator(model, steps):
		return evaluate_validation_buckets_cached(
			model=model,
			cache=cache,
			indices=validation_indices,
			args=args,
			steps=steps,
		)

	print(f"Encoded cache: {args.encoded_cache_dir} ({cache.rows} rows)")
	return SimpleNamespace(
		raw_rows=cache.rows,
		sample_fraction=sample_fraction,
		split_counts=split_counts,
		extra_row_counts=extra_row_counts,
		examples_per_epoch=examples_per_epoch,
		train_dataset=make_cached_train_dataset(
			args, cache, train_indices, extra_caches
		),
		validation_dataset=validation_dataset,
		test_dataset=test_dataset,
		train_eval_dataset=train_eval_dataset,
		bucket_evaluator=bucket_evaluator,
	)


def prepare_generator_data(args):
	if can_use_default_split_counts(args):
		print("Using known default source and split counts.", flush=True)
		raw_rows = DEFAULT_SOURCE_ROWS
		sample_fraction = sample_fraction_for(raw_rows, args.sample_size)
		split_counts = dict(DEFAULT_SPLIT_COUNTS_BY_SAMPLE[args.sample_size])
	else:
		print("Counting source rows...", flush=True)
		raw_rows = count_source_rows(args.data_file, args.max_rows)
		if raw_rows == 0:
			raise ValueError("input CSV contains no data rows")

		sample_fraction = sample_fraction_for(raw_rows, args.sample_size)
		print("Computing deterministic split counts...", flush=True)
		split_counts = count_split_rows(
			path=args.data_file,
			raw_rows=raw_rows,
			sample_fraction=sample_fraction,
			sample_seed=args.sample_seed,
			split_seed=args.split_seed,
		)
	if split_counts["train"] == 0:
		raise ValueError("deterministic sample contains no training rows")

	extra_row_counts = count_extra_rows(args)
	examples_per_epoch = train_example_count(
		split_counts=split_counts,
		extra_row_counts=extra_row_counts,
		mirror_augmentation=args.mirror_augmentation,
	)
	args.primary_train_rows = split_counts["train"]

	validation_dataset = None
	if split_counts["validation"] > 0:
		validation_dataset = make_dataset(
			args, raw_rows, "validation", sample_fraction
		)
	test_dataset = None
	if split_counts["test"] > 0:
		test_dataset = make_dataset(args, raw_rows, "test", sample_fraction)

	train_eval_dataset = None
	if args.train_split_eval_examples > 0 and not args.dry_run:
		print(
			f"Encoding {args.train_split_eval_examples} train-split rows "
			"for per-epoch evaluation...",
			flush=True,
		)
		train_eval_dataset = make_generator_train_eval_dataset(
			args, raw_rows, sample_fraction
		)

	def bucket_evaluator(model, steps):
		return evaluate_validation_buckets(
			model=model,
			args=args,
			raw_rows=raw_rows,
			sample_fraction=sample_fraction,
			steps=steps,
		)

	return SimpleNamespace(
		raw_rows=raw_rows,
		sample_fraction=sample_fraction,
		split_counts=split_counts,
		extra_row_counts=extra_row_counts,
		examples_per_epoch=examples_per_epoch,
		train_dataset=make_train_dataset(
			args,
			raw_rows,
			sample_fraction,
			extra_row_counts=extra_row_counts,
		),
		validation_dataset=validation_dataset,
		test_dataset=test_dataset,
		train_eval_dataset=train_eval_dataset,
		bucket_evaluator=bucket_evaluator,
	)


def main():
	args = parse_args()
	validate_args(args)
	configure_tensorflow(args)

	print("Preparing depth-8 position evaluator training.", flush=True)
	if args.encoded_cache_dir is not None:
		data = prepare_cached_data(args)
	else:
		data = prepare_generator_data(args)
	raw_rows = data.raw_rows
	sample_fraction = data.sample_fraction
	split_counts = data.split_counts
	extra_row_counts = data.extra_row_counts
	examples_per_epoch = data.examples_per_epoch

	steps_per_epoch = capped_steps(
		examples_per_epoch, args.batch_size, args.steps_per_epoch
	)
	validation_steps = capped_steps(
		split_counts["validation"], args.batch_size, args.validation_steps
	)
	test_steps = capped_steps(
		split_counts["test"], args.batch_size, args.test_steps
	)
	args.train_steps_per_epoch = steps_per_epoch
	args.total_train_steps = steps_per_epoch * args.epochs
	args.lr_schedule_step_offset = steps_per_epoch * args.initial_epoch

	print("Experiment: depth-8 position evaluator with pawn-scale MAE")
	print(f"Data file: {args.data_file}")
	if args.extra_data_file:
		print("Extra data policy: train-only")
		for path in args.extra_data_file:
			print(f"Extra data file: {path}")
	for path in args.extra_cache_dir:
		print(f"Extra cache: {path}")
	print(f"Architecture: {ARCHITECTURE}")
	print(
		"Transformer: "
		f"d_model={args.transformer_d_model}, "
		f"heads={args.transformer_heads}, "
		f"layers={args.transformer_layers}, "
		f"ff_dim={args.transformer_ff_dim}, "
		f"dropout={args.transformer_dropout:g}"
	)
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
	if extra_row_counts:
		print(
			"Train-only extra rows: "
			f"{sum(extra_row_counts.values())}"
		)
	if args.mirror_augmentation:
		print("Train augmentation: horizontal mirror")
	print(f"Training examples per epoch: {examples_per_epoch}")
	print(f"Target: clipped to [-{args.max_abs_pawns:g}, +{args.max_abs_pawns:g}] pawns")
	print(f"Loss: {loss_description(args)}")
	print("Metric: MAE in pawns")
	print(f"Optimizer: {optimizer_description(args)}")
	print(f"Batch size: {args.batch_size}")
	print(f"Mixed precision: {args.mixed_precision_policy}")
	if args.train_split_eval_examples > 0:
		print(
			"Train-split eval: "
			f"{args.train_split_eval_examples} fixed rows, inference mode"
		)
	if args.target_mae > 0:
		print(f"Target stop: train and validation MAE <= {args.target_mae:g}")
	if args.early_stopping_patience > 0:
		print(
			"Early stopping: "
			f"patience={args.early_stopping_patience}, "
			f"min_delta={args.early_stopping_min_delta:g}"
		)
	print(
		"Steps: "
		f"{steps_per_epoch} train, "
		f"{validation_steps} validation, "
		f"{test_steps} test"
	)
	print(f"Model checkpoint: {args.model_out}")
	print(f"Weights checkpoint: {args.weights_out}")
	if args.initial_weights is not None:
		print(f"Initial weights: {args.initial_weights}")
	if args.initial_epoch > 0:
		print(
			f"Resuming at epoch {args.initial_epoch + 1}/{args.epochs} "
			f"(LR schedule step offset {args.lr_schedule_step_offset})"
		)
		if args.initial_epoch >= args.epochs:
			print(
				"All requested epochs are already in the history CSV; "
				"skipping training and running final evaluation only."
			)
	print(f"Training report: {args.report_out}")
	print(f"Training log: {args.log_out}")
	if validation_steps == 0:
		print("Validation is disabled because this split has no rows.")

	train_dataset = data.train_dataset
	validation_dataset = data.validation_dataset if validation_steps > 0 else None
	test_dataset = data.test_dataset if test_steps > 0 else None

	model = build_model(args)
	if args.initial_weights is not None:
		model.load_weights(args.initial_weights)
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
	args.log_out.parent.mkdir(parents=True, exist_ok=True)

	checkpoint_monitor = "val_mae" if validation_steps > 0 else "mae"

	# On resume, seed the checkpoint's best-so-far from the history CSV so a
	# worse first resumed epoch cannot overwrite the better saved weights.
	checkpoint_kwargs = {}
	if args.initial_epoch > 0:
		prior_best = best_metric_from_history(args.log_out, checkpoint_monitor)
		if prior_best is not None and "initial_value_threshold" in inspect.signature(
			tf.keras.callbacks.ModelCheckpoint.__init__
		).parameters:
			checkpoint_kwargs["initial_value_threshold"] = prior_best

	callbacks = [
		tf.keras.callbacks.TerminateOnNaN(),
	]
	if data.train_eval_dataset is not None:
		callbacks.append(TrainSplitMae(data.train_eval_dataset))
	callbacks += [
		tf.keras.callbacks.ModelCheckpoint(
			filepath=str(args.weights_out),
			monitor=checkpoint_monitor,
			mode="min",
			save_best_only=True,
			save_weights_only=True,
			**checkpoint_kwargs,
		),
		tf.keras.callbacks.CSVLogger(
			str(args.log_out), append=args.initial_epoch > 0
		),
	]
	if args.early_stopping_patience > 0:
		callbacks.append(
			tf.keras.callbacks.EarlyStopping(
				monitor=checkpoint_monitor,
				mode="min",
				patience=args.early_stopping_patience,
				min_delta=args.early_stopping_min_delta,
				restore_best_weights=True,
			)
		)
	if args.target_mae > 0:
		callbacks.append(
			TargetMaeStop(
				target_mae=args.target_mae,
				require_validation=validation_steps > 0,
			)
		)

	fit_kwargs = {
		"x": train_dataset,
		"epochs": args.epochs,
		"initial_epoch": args.initial_epoch,
		"steps_per_epoch": steps_per_epoch,
		"callbacks": callbacks,
	}
	if validation_dataset is not None:
		fit_kwargs["validation_data"] = validation_dataset
		fit_kwargs["validation_steps"] = validation_steps

	history = model.fit(**fit_kwargs)

	model.load_weights(args.weights_out)
	model.save(str(args.model_out))
	if test_steps > 0:
		results = model.evaluate(
			test_dataset,
			steps=test_steps,
			return_dict=True,
		)
	else:
		results = {}
	train_split_results = None
	if data.train_eval_dataset is not None:
		train_split_results = model.evaluate(
			data.train_eval_dataset,
			verbose=0,
			return_dict=True,
		)
	validation_bucket_metrics = data.bucket_evaluator(model, validation_steps)
	write_training_report(
		args=args,
		split_counts=split_counts,
		extra_row_counts=extra_row_counts,
		history=history,
		test_results=results,
		validation_bucket_metrics=validation_bucket_metrics,
		model_params=model.count_params(),
		train_split_results=train_split_results,
	)
	print(f"Saved best evaluator model to {args.model_out}")
	print(f"Saved best weights to {args.weights_out}")
	print(f"Wrote training report to {args.report_out}")
	if train_split_results:
		print(
			"Best-weights train-split metrics: "
			+ ", ".join(
				f"{name}={value:.6f}"
				for name, value in train_split_results.items()
			)
		)
	print(
		"Test metrics: "
		+ ", ".join(f"{name}={value:.6f}" for name, value in results.items())
	)


if __name__ == "__main__":
	main()
