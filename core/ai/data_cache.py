"""Stream training examples from a pre-encoded cache built by encode_dataset.py."""

from types import SimpleNamespace

import numpy as np
import tensorflow as tf

from board_features import (
	PERSPECTIVE_V3_FEATURE_SIZE,
	hash_fraction,
	sample_fraction_for,
)
from config import ARCHITECTURE
from encoded_cache import EncodedCache, unpack_features
from metrics import (
	add_bucket_error,
	finalize_bucket_metrics,
	new_target_buckets,
	target_bucket_name,
)


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
	config,
	shuffle=False,
	repeat=False,
	mirror=False,
	seed=0,
):
	fetch = make_cached_batch_fetcher(cache, config.max_abs_pawns)
	dataset = tf.data.Dataset.from_tensor_slices(np.asarray(indices, dtype=np.int64))
	if shuffle:
		dataset = dataset.shuffle(
			len(indices), seed=seed, reshuffle_each_iteration=True
		)
	if repeat:
		dataset = dataset.repeat()
	dataset = dataset.batch(config.batch_size)
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


def make_cached_train_dataset(config, cache, train_indices, extra_caches):
	sources = []
	weights = []

	def seed_for(value):
		return int(hash_fraction(config.split_seed, value) * (2**31 - 1))

	def add_source(dataset, example_count):
		sources.append(dataset)
		weights.append(float(example_count))

	add_source(
		make_cached_dataset(
			cache=cache,
			indices=train_indices,
			config=config,
			shuffle=config.shuffle_buffer > 0,
			repeat=True,
			mirror=config.mirror_augmentation,
			seed=seed_for("shuffle:primary"),
		),
		example_count=len(train_indices),
	)
	for source_index, (extra_cache, extra_indices) in enumerate(extra_caches):
		add_source(
			make_cached_dataset(
				cache=extra_cache,
				indices=extra_indices,
				config=config,
				shuffle=config.shuffle_buffer > 0,
				repeat=True,
				mirror=config.mirror_augmentation,
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


def evaluate_validation_buckets_cached(model, cache, indices, config, steps):
	# Mate rows are not distinguishable from other clipped rows in the
	# cache, so only the clipped bucket is reported.
	if steps == 0 or len(indices) == 0:
		return {}

	buckets = new_target_buckets(("clipped",))
	clip = min(config.max_abs_pawns, cache.max_abs_pawns)
	max_examples = min(len(indices), steps * config.batch_size)
	for start in range(0, max_examples, config.batch_size):
		batch_indices = indices[start:start + config.batch_size]
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


def _check_cache_compatible(config, cache):
	if cache.train_only:
		raise ValueError(
			"the primary encoded cache must contain train/validation/test "
			"splits; it was built with --train-only"
		)
	if config.split_seed != cache.split_seed:
		raise ValueError(
			f"cache split seed {cache.split_seed!r} does not match "
			f"split_seed {config.split_seed!r}"
		)
	if config.max_abs_pawns > cache.max_abs_pawns:
		raise ValueError(
			f"cache targets are clipped to +/-{cache.max_abs_pawns:g} pawns; "
			f"max_abs_pawns {config.max_abs_pawns:g} needs a rebuilt cache"
		)
	if config.sample_size > 0 and config.sample_seed != cache.sample_seed:
		raise ValueError(
			f"cache sample seed {cache.sample_seed!r} does not match "
			f"sample_seed {config.sample_seed!r}"
		)


def prepare_cached_data(config):
	cache = EncodedCache(config.encoded_cache_dir, architecture=ARCHITECTURE)
	_check_cache_compatible(config, cache)

	sample_fraction = sample_fraction_for(cache.rows, config.sample_size)
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
	for path in config.extra_cache_dir:
		extra_cache = EncodedCache(path, architecture=ARCHITECTURE)
		if config.max_abs_pawns > extra_cache.max_abs_pawns:
			raise ValueError(
				f"extra cache {path} is clipped to "
				f"+/-{extra_cache.max_abs_pawns:g} pawns"
			)
		extra_indices = extra_cache.split_indices("train")
		extra_caches.append((extra_cache, extra_indices))
		extra_row_counts[path] = len(extra_indices)

	validation_dataset = None
	if len(validation_indices) > 0:
		validation_dataset = make_cached_dataset(
			cache, validation_indices, config
		).prefetch(tf.data.AUTOTUNE)
	test_dataset = None
	if len(test_indices) > 0:
		test_dataset = make_cached_dataset(cache, test_indices, config).prefetch(
			tf.data.AUTOTUNE
		)

	train_eval_dataset = None
	if config.train_split_eval_examples > 0:
		rng = np.random.default_rng(
			int(hash_fraction(config.split_seed, "train-split-eval") * (2**31 - 1))
		)
		eval_count = min(config.train_split_eval_examples, len(train_indices))
		eval_indices = np.sort(
			rng.choice(train_indices, size=eval_count, replace=False)
		)
		train_eval_dataset = make_cached_dataset(
			cache, eval_indices, config
		).prefetch(tf.data.AUTOTUNE)

	def bucket_evaluator(model, steps):
		return evaluate_validation_buckets_cached(
			model=model,
			cache=cache,
			indices=validation_indices,
			config=config,
			steps=steps,
		)

	print(f"Encoded cache: {config.encoded_cache_dir} ({cache.rows} rows)")
	return SimpleNamespace(
		raw_rows=cache.rows,
		sample_fraction=sample_fraction,
		split_counts=split_counts,
		extra_row_counts=extra_row_counts,
		examples_per_epoch=len(train_indices)
		+ sum(len(extra_indices) for _, extra_indices in extra_caches),
		train_dataset=make_cached_train_dataset(
			config, cache, train_indices, extra_caches
		),
		validation_dataset=validation_dataset,
		test_dataset=test_dataset,
		train_eval_dataset=train_eval_dataset,
		bucket_evaluator=bucket_evaluator,
	)
