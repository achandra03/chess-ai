"""Train the perspective-transformer-v3 position evaluator.

Run with `python train.py` from the repo root. Settings live in
config.py; edit TrainConfig's defaults to change a run.
"""

import math
import os
from types import SimpleNamespace

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

from callbacks import build_callbacks
from config import TrainConfig
from data_cache import prepare_cached_data
from data_csv import prepare_generator_data
from model import build_model_from_config
from optimizer import (
	build_optimizer,
	loss_description,
	loss_for_training,
	optimizer_description,
)


def configure_tensorflow(config):
	gpus = tf.config.list_physical_devices("GPU")
	for gpu in gpus:
		try:
			tf.config.experimental.set_memory_growth(gpu, True)
		except RuntimeError:
			pass
	use_mixed_precision = config.mixed_precision == "on" or (
		config.mixed_precision == "auto" and bool(gpus)
	)
	policy = "mixed_float16" if use_mixed_precision else "float32"
	tf.keras.mixed_precision.set_global_policy(policy)
	config.mixed_precision_policy = policy


def build_model(config):
	model = build_model_from_config(config)
	model.compile(
		optimizer=build_optimizer(config),
		loss=loss_for_training(config),
		metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
	)
	return model


def capped_steps(row_count, batch_size, requested_steps):
	steps = math.ceil(row_count / batch_size)
	if requested_steps is not None:
		steps = min(steps, requested_steps)
	return steps


def resolve_steps(config, data):
	"""Per-epoch step counts, also recording them on the config.

	The LR schedule needs the totals, so this has to run before the
	optimizer is built.
	"""
	steps = SimpleNamespace(
		train=capped_steps(
			data.examples_per_epoch, config.batch_size, config.steps_per_epoch
		),
		validation=capped_steps(
			data.split_counts["validation"],
			config.batch_size,
			config.validation_steps,
		),
		test=capped_steps(
			data.split_counts["test"], config.batch_size, config.test_steps
		),
	)
	config.train_steps_per_epoch = steps.train
	config.total_train_steps = steps.train * config.epochs
	config.lr_schedule_step_offset = steps.train * config.initial_epoch
	return steps


def print_run_summary(config, data, steps):
	"""Echo what this run is about to do.

	Per-epoch metrics go to the history CSV; this is only the up-front
	sanity check before a long run.
	"""
	source = config.encoded_cache_dir or config.data_file
	print(f"Source: {source}")
	print(
		f"Split: {data.split_counts['train']} train, "
		f"{data.split_counts['validation']} validation, "
		f"{data.split_counts['test']} test"
	)
	print(f"Examples per epoch: {data.examples_per_epoch}")
	print(
		f"Transformer: d_model={config.transformer_d_model}, "
		f"heads={config.transformer_heads}, "
		f"layers={config.transformer_layers}, "
		f"ff_dim={config.transformer_ff_dim}, "
		f"dropout={config.transformer_dropout:g}"
	)
	print(f"Loss: {loss_description(config)} (metric: MAE in pawns)")
	print(f"Optimizer: {optimizer_description(config)}")
	print(
		f"Batch size: {config.batch_size}, "
		f"mixed precision: {config.mixed_precision_policy}"
	)
	print(
		f"Steps: {steps.train} train, {steps.validation} validation, "
		f"{steps.test} test"
	)
	if config.initial_epoch > 0:
		print(f"Resuming at epoch {config.initial_epoch + 1}/{config.epochs}")
	print(f"Weights: {config.weights_out}")
	print(f"History: {config.log_out}")


def print_bucket_metrics(bucket_metrics):
	if not bucket_metrics:
		return
	print("Validation MAE by target magnitude:")
	for name, values in bucket_metrics.items():
		mae = values["mae"]
		mae_text = "n/a" if mae is None else f"{mae:.6f}"
		print(f"  {name:<18} count={values['count']:<8} mae={mae_text}")


def run_dry_run(model, train_dataset):
	for features, targets in train_dataset.take(1):
		print(f"features: {features.shape} {features.dtype}")
		print(f"targets: {targets.shape} {targets.dtype}")
		print(
			f"target range: {targets.numpy().min():.5f} "
			f"to {targets.numpy().max():.5f}"
		)
	print(f"Model parameters: {model.count_params()}")


def fit_model(model, config, data, steps):
	fit_kwargs = {
		"x": data.train_dataset,
		"epochs": config.epochs,
		"initial_epoch": config.initial_epoch,
		"steps_per_epoch": steps.train,
		"callbacks": build_callbacks(config, data.train_eval_dataset, steps.validation),
	}
	if steps.validation > 0 and data.validation_dataset is not None:
		fit_kwargs["validation_data"] = data.validation_dataset
		fit_kwargs["validation_steps"] = steps.validation
	return model.fit(**fit_kwargs)


def evaluate_final(model, config, data, steps):
	model.load_weights(config.weights_out)
	model.save(str(config.model_out))

	test_results = {}
	if steps.test > 0 and data.test_dataset is not None:
		test_results = model.evaluate(
			data.test_dataset, steps=steps.test, return_dict=True
		)
	train_split_results = None
	if data.train_eval_dataset is not None:
		train_split_results = model.evaluate(
			data.train_eval_dataset, verbose=0, return_dict=True
		)

	print(f"Saved best evaluator model to {config.model_out}")
	print(f"Saved best weights to {config.weights_out}")
	print(f"Model parameters: {model.count_params()}")
	if train_split_results:
		print(
			"Best-weights train-split metrics: "
			+ ", ".join(
				f"{name}={value:.6f}"
				for name, value in train_split_results.items()
			)
		)
	if test_results:
		print(
			"Test metrics: "
			+ ", ".join(
				f"{name}={value:.6f}" for name, value in test_results.items()
			)
		)
	print_bucket_metrics(data.bucket_evaluator(model, steps.validation))


def main():
	config = TrainConfig()
	config.validate()
	configure_tensorflow(config)

	print("Preparing depth-8 position evaluator training.", flush=True)
	if config.encoded_cache_dir is not None:
		data = prepare_cached_data(config)
	else:
		data = prepare_generator_data(config)

	steps = resolve_steps(config, data)
	print_run_summary(config, data, steps)

	model = build_model(config)
	if config.initial_weights is not None:
		model.load_weights(config.initial_weights)
	if config.dry_run:
		run_dry_run(model, data.train_dataset)
		return

	for path in (config.model_out, config.weights_out, config.log_out):
		path.parent.mkdir(parents=True, exist_ok=True)

	fit_model(model, config, data, steps)
	evaluate_final(model, config, data, steps)


if __name__ == "__main__":
	main()
