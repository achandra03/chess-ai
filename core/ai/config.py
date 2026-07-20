"""Training configuration for the perspective-transformer-v3 evaluator.

There is no command line: edit TrainConfig's defaults to change a run.
A None step cap means "use the whole split". The last five fields, from
mixed_precision_policy onward, are filled in by the trainer at runtime and
are not meant to be set by hand.

Must stay TensorFlow-free so the light modules can import it.
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path

from encoded_cache import MANIFEST_NAME as CACHE_MANIFEST_NAME


ROOT_DIR = Path(__file__).resolve().parents[2]
AI_DIR = Path(__file__).resolve().parent
ARCHITECTURE = "perspective-transformer-v3"
DEFAULT_DATA_FILE = ROOT_DIR / "data" / "chessData_depth8.csv"
MODEL_STEM = "position_evaluator_perspective_transformer_v3_mae"
DEFAULT_MODEL_OUT = AI_DIR / f"{MODEL_STEM}.keras"
DEFAULT_WEIGHTS_OUT = AI_DIR / f"{MODEL_STEM}.weights.h5"


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


@dataclass
class TrainConfig:
	data_file: Path = DEFAULT_DATA_FILE
	extra_data_file: list = field(default_factory=list)
	encoded_cache_dir: Path = None
	no_encoded_cache: bool = False
	extra_cache_dir: list = field(default_factory=list)

	sample_size: int = 0
	scan_data: bool = False
	max_rows: int = None
	sample_seed: str = "sabatelli-dataset4-sample-v1"
	split_seed: str = "sabatelli-dataset4-split-v1"
	max_abs_pawns: float = 10.0
	mirror_augmentation: bool = True
	shuffle_buffer: int = 100_000

	transformer_d_model: int = 384
	transformer_heads: int = 8
	transformer_layers: int = 6
	transformer_ff_dim: int = 1536
	transformer_dropout: float = 0.05

	epochs: int = 50
	batch_size: int = 512
	loss: str = "huber"
	huber_delta: float = 0.75
	optimizer: str = "adamw"
	weight_decay: float = 1e-4
	gradient_clipnorm: float = 1.0
	learning_rate: float = 0.001
	min_learning_rate: float = 0.000001
	lr_schedule: str = "warmup-cosine"
	warmup_ratio: float = 0.05
	warmup_epochs: float = 2.0
	warmup_steps: int = None
	mixed_precision: str = "auto"

	target_mae: float = 0.5
	early_stopping_patience: int = 18
	early_stopping_min_delta: float = 0.001

	steps_per_epoch: int = None
	validation_steps: int = None
	test_steps: int = None
	train_split_eval_examples: int = 100_000

	model_out: Path = DEFAULT_MODEL_OUT
	weights_out: Path = DEFAULT_WEIGHTS_OUT
	log_out: Path = None

	resume: bool = False
	initial_weights: Path = None
	initial_epoch: int = 0

	dry_run: bool = False

	mixed_precision_policy: str = "float32"
	primary_train_rows: int = 0
	train_steps_per_epoch: int = 0
	total_train_steps: int = 0
	lr_schedule_step_offset: int = 0

	def __post_init__(self):
		self.data_file = Path(self.data_file)
		self.model_out = Path(self.model_out)
		self.weights_out = Path(self.weights_out)
		self.extra_data_file = list(
			dict.fromkeys(Path(path).resolve() for path in self.extra_data_file)
		)
		self.extra_cache_dir = [Path(path) for path in self.extra_cache_dir]
		if self.log_out is None:
			self.log_out = self.model_out.with_suffix(".history.csv")

		# The checkpoint holds the best epoch's weights, not the last one's.
		if self.resume and self.weights_out.is_file():
			self.initial_weights = self.weights_out
			self.initial_epoch = completed_epochs_from_history(self.log_out)

		if self.no_encoded_cache:
			self.encoded_cache_dir = None
		elif self.encoded_cache_dir is None:
			candidate = default_cache_dir_for(self.data_file)
			if (candidate / CACHE_MANIFEST_NAME).is_file():
				self.encoded_cache_dir = candidate

	def validate(self):
		"""Check the things that stay wrong until a human intervenes.

		Values are code constants now, so this covers missing files and
		cross-field invariants rather than every numeric bound.
		"""
		if not self.data_file.is_file():
			raise FileNotFoundError(self.data_file)
		for path in self.extra_data_file:
			if not path.is_file():
				raise FileNotFoundError(path)

		if self.encoded_cache_dir is not None:
			if not (self.encoded_cache_dir / CACHE_MANIFEST_NAME).is_file():
				raise FileNotFoundError(
					f"{self.encoded_cache_dir / CACHE_MANIFEST_NAME}; "
					"build the cache with encode_dataset.py"
				)
			if self.extra_data_file:
				raise ValueError(
					"CSV extra data cannot be mixed with an encoded cache; "
					"encode extras with encode_dataset.py --train-only and set "
					"extra_cache_dir"
				)
			for extra_cache_dir in self.extra_cache_dir:
				if not (extra_cache_dir / CACHE_MANIFEST_NAME).is_file():
					raise FileNotFoundError(extra_cache_dir / CACHE_MANIFEST_NAME)
		elif self.extra_cache_dir:
			raise ValueError("extra_cache_dir requires an encoded cache")

		if self.initial_weights is not None and not Path(self.initial_weights).is_file():
			raise FileNotFoundError(self.initial_weights)
		if self.transformer_d_model % self.transformer_heads != 0:
			raise ValueError(
				"transformer_d_model must be divisible by transformer_heads"
			)
		if self.min_learning_rate > self.learning_rate:
			raise ValueError("min_learning_rate cannot exceed learning_rate")
		if not str(self.weights_out).endswith(".weights.h5"):
			raise ValueError("weights_out must end with .weights.h5")
