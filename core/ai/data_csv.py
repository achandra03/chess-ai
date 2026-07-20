"""Stream training examples straight from the source CSV.

Used when no encoded cache is available. Encoding FENs in Python is slow,
so prefer building a cache with encode_dataset.py.
"""

import csv
from types import SimpleNamespace

import numpy as np
import tensorflow as tf

from board_features import (
	PERSPECTIVE_V3_FEATURE_SIZE,
	encode_fen_perspective_v3,
	evaluation_to_pawns,
	hash_fraction,
	sample_fraction_for,
	selected_for_sample,
	split_for_fen,
)
from config import DEFAULT_DATA_FILE
from metrics import (
	add_bucket_error,
	finalize_bucket_metrics,
	new_target_buckets,
	target_bucket_name,
)


DEFAULT_SOURCE_ROWS = 12_958_035
DEFAULT_SPLIT_COUNTS_BY_SAMPLE = {
	3_000_000: {
		"train": 2_402_392,
		"validation": 300_013,
		"test": 300_099,
	},
}


def validate_csv_header(fieldnames):
	if fieldnames is None:
		raise ValueError("input CSV has no header")
	if "FEN" not in fieldnames or "Evaluation" not in fieldnames:
		raise ValueError("input CSV must contain FEN and Evaluation columns")


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
				row["Evaluation"], max_abs_pawns=max_abs_pawns
			)
			target_array = np.array([target], dtype=np.float32)
			yield encode_fen_perspective_v3(fen), target_array
			if augment_mirror:
				yield encode_fen_perspective_v3(fen, mirror_files=True), target_array


def make_example_dataset(
	config,
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
			sample_seed=config.sample_seed,
			split_seed=config.split_seed,
			max_abs_pawns=config.max_abs_pawns,
			augment_mirror=augment_mirror,
			train_only=train_only,
		),
		output_signature=(
			tf.TensorSpec(shape=(PERSPECTIVE_V3_FEATURE_SIZE,), dtype=tf.float32),
			tf.TensorSpec(shape=(1,), dtype=tf.float32),
		),
	)


def make_train_dataset(config, raw_rows, sample_fraction, extra_row_counts):
	source_datasets = []
	source_weights = []
	mirror_factor = 2 if config.mirror_augmentation else 1

	def seed_for(value):
		return int(hash_fraction(config.split_seed, value) * (2**31 - 1))

	def add_source(dataset, example_count, seed_value):
		if config.shuffle_buffer > 0:
			dataset = dataset.shuffle(
				config.shuffle_buffer,
				seed=seed_for(seed_value),
				reshuffle_each_iteration=True,
			)
		source_datasets.append(dataset.repeat())
		source_weights.append(float(example_count))

	add_source(
		make_example_dataset(
			config=config,
			path=config.data_file,
			raw_rows=raw_rows,
			split="train",
			sample_fraction=sample_fraction,
			augment_mirror=config.mirror_augmentation,
		),
		example_count=config.primary_train_rows * mirror_factor,
		seed_value="shuffle:primary",
	)

	for source_index, (path, extra_raw_rows) in enumerate(extra_row_counts.items()):
		add_source(
			make_example_dataset(
				config=config,
				path=path,
				raw_rows=extra_raw_rows,
				split="train",
				sample_fraction=1.0,
				augment_mirror=config.mirror_augmentation,
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
			seed=seed_for("source-mix"),
		)
	return dataset.batch(config.batch_size).prefetch(tf.data.AUTOTUNE)


def make_split_dataset(config, raw_rows, split, sample_fraction):
	dataset = make_example_dataset(
		config=config,
		path=config.data_file,
		raw_rows=raw_rows,
		split=split,
		sample_fraction=sample_fraction,
	)
	return dataset.batch(config.batch_size).prefetch(tf.data.AUTOTUNE)


def make_train_eval_dataset(config, raw_rows, sample_fraction):
	# The CSV path has no random access, so the first examples of the
	# training split are encoded once up front and kept in memory.
	count = config.train_split_eval_examples
	if count == 0:
		return None
	features = np.empty((count, PERSPECTIVE_V3_FEATURE_SIZE), dtype=np.float16)
	targets = np.empty((count, 1), dtype=np.float32)
	collected = 0
	for example_features, example_target in iter_examples(
		path=config.data_file,
		raw_rows=raw_rows,
		split="train",
		sample_fraction=sample_fraction,
		sample_seed=config.sample_seed,
		split_seed=config.split_seed,
		max_abs_pawns=config.max_abs_pawns,
	):
		features[collected] = example_features
		targets[collected] = example_target
		collected += 1
		if collected >= count:
			break
	if collected == 0:
		return None
	dataset = tf.data.Dataset.from_tensor_slices(
		(features[:collected], targets[:collected])
	)
	dataset = dataset.batch(config.batch_size)
	return dataset.map(
		lambda batch_features, batch_targets: (
			tf.cast(batch_features, tf.float32),
			batch_targets,
		)
	).prefetch(tf.data.AUTOTUNE)


def evaluate_validation_buckets(model, config, raw_rows, sample_fraction, steps):
	if steps == 0:
		return {}

	buckets = new_target_buckets(("mates", "clipped"))
	max_examples = steps * config.batch_size
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

	with config.data_file.open(newline="") as file:
		reader = csv.DictReader(file)
		validate_csv_header(reader.fieldnames)
		for row_index, row in enumerate(reader):
			if row_index >= raw_rows or seen >= max_examples:
				break

			fen = row["FEN"]
			if not selected_for_sample(fen, sample_fraction, config.sample_seed):
				continue
			if split_for_fen(fen, config.split_seed) != "validation":
				continue

			target = evaluation_to_pawns(
				row["Evaluation"], max_abs_pawns=config.max_abs_pawns
			)
			features_batch.append(encode_fen_perspective_v3(fen))
			targets_batch.append(target)
			flags_batch.append(
				{
					"mate": str(row["Evaluation"]).strip().startswith("#"),
					"clipped": abs(target) >= config.max_abs_pawns,
				}
			)
			seen += 1
			if len(features_batch) >= config.batch_size:
				flush_batch()

	flush_batch()
	return finalize_bucket_metrics(buckets)


def can_use_default_split_counts(config):
	return (
		not config.scan_data
		and config.max_rows is None
		and config.data_file.resolve() == DEFAULT_DATA_FILE.resolve()
		and config.sample_size in DEFAULT_SPLIT_COUNTS_BY_SAMPLE
	)


def train_example_count(split_counts, extra_row_counts, mirror_augmentation):
	count = split_counts["train"] + sum(extra_row_counts.values())
	if mirror_augmentation:
		count *= 2
	return count


def prepare_generator_data(config):
	if can_use_default_split_counts(config):
		print("Using known default source and split counts.", flush=True)
		raw_rows = DEFAULT_SOURCE_ROWS
		sample_fraction = sample_fraction_for(raw_rows, config.sample_size)
		split_counts = dict(DEFAULT_SPLIT_COUNTS_BY_SAMPLE[config.sample_size])
	else:
		print("Counting source rows...", flush=True)
		raw_rows = count_source_rows(config.data_file, config.max_rows)
		if raw_rows == 0:
			raise ValueError("input CSV contains no data rows")

		sample_fraction = sample_fraction_for(raw_rows, config.sample_size)
		print("Computing deterministic split counts...", flush=True)
		split_counts = count_split_rows(
			path=config.data_file,
			raw_rows=raw_rows,
			sample_fraction=sample_fraction,
			sample_seed=config.sample_seed,
			split_seed=config.split_seed,
		)
	if split_counts["train"] == 0:
		raise ValueError("deterministic sample contains no training rows")

	extra_row_counts = {
		path: count_source_rows(path, config.max_rows)
		for path in config.extra_data_file
	}
	config.primary_train_rows = split_counts["train"]

	validation_dataset = None
	if split_counts["validation"] > 0:
		validation_dataset = make_split_dataset(
			config, raw_rows, "validation", sample_fraction
		)
	test_dataset = None
	if split_counts["test"] > 0:
		test_dataset = make_split_dataset(config, raw_rows, "test", sample_fraction)

	train_eval_dataset = None
	if config.train_split_eval_examples > 0 and not config.dry_run:
		print(
			f"Encoding {config.train_split_eval_examples} train-split rows "
			"for per-epoch evaluation...",
			flush=True,
		)
		train_eval_dataset = make_train_eval_dataset(
			config, raw_rows, sample_fraction
		)

	def bucket_evaluator(model, steps):
		return evaluate_validation_buckets(
			model=model,
			config=config,
			raw_rows=raw_rows,
			sample_fraction=sample_fraction,
			steps=steps,
		)

	return SimpleNamespace(
		raw_rows=raw_rows,
		sample_fraction=sample_fraction,
		split_counts=split_counts,
		extra_row_counts=extra_row_counts,
		examples_per_epoch=train_example_count(
			split_counts=split_counts,
			extra_row_counts=extra_row_counts,
			mirror_augmentation=config.mirror_augmentation,
		),
		train_dataset=make_train_dataset(
			config, raw_rows, sample_fraction, extra_row_counts=extra_row_counts
		),
		validation_dataset=validation_dataset,
		test_dataset=test_dataset,
		train_eval_dataset=train_eval_dataset,
		bucket_evaluator=bucket_evaluator,
	)
