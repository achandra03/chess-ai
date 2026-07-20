"""Keras callbacks for the evaluator training run."""

import inspect

import tensorflow as tf

from config import best_metric_from_history


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
		results = self.model.evaluate(self.dataset, verbose=0, return_dict=True)
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
		if self.require_validation and (val_mae is None or val_mae > self.target_mae):
			return

		metric_text = f"{train_metric}={train_mae:.6f}"
		if val_mae is not None:
			metric_text += f", val_mae={val_mae:.6f}"
		print(
			f"\nTarget MAE reached after epoch {epoch + 1}: {metric_text}",
			flush=True,
		)
		self.model.stop_training = True


def _checkpoint_kwargs(config, monitor):
	# On resume, seed the checkpoint's best-so-far from the history CSV so a
	# worse first resumed epoch cannot overwrite the better saved weights.
	if config.initial_epoch <= 0:
		return {}
	prior_best = best_metric_from_history(config.log_out, monitor)
	if prior_best is None:
		return {}
	supported = inspect.signature(
		tf.keras.callbacks.ModelCheckpoint.__init__
	).parameters
	if "initial_value_threshold" not in supported:
		return {}
	return {"initial_value_threshold": prior_best}


def build_callbacks(config, train_eval_dataset, validation_steps):
	monitor = "val_mae" if validation_steps > 0 else "mae"

	callbacks = [tf.keras.callbacks.TerminateOnNaN()]
	if train_eval_dataset is not None:
		callbacks.append(TrainSplitMae(train_eval_dataset))
	callbacks += [
		tf.keras.callbacks.ModelCheckpoint(
			filepath=str(config.weights_out),
			monitor=monitor,
			mode="min",
			save_best_only=True,
			save_weights_only=True,
			**_checkpoint_kwargs(config, monitor),
		),
		tf.keras.callbacks.CSVLogger(
			str(config.log_out), append=config.initial_epoch > 0
		),
	]
	if config.early_stopping_patience > 0:
		callbacks.append(
			tf.keras.callbacks.EarlyStopping(
				monitor=monitor,
				mode="min",
				patience=config.early_stopping_patience,
				min_delta=config.early_stopping_min_delta,
				restore_best_weights=True,
			)
		)
	if config.target_mae > 0:
		callbacks.append(
			TargetMaeStop(
				target_mae=config.target_mae,
				require_validation=validation_steps > 0,
			)
		)
	return callbacks
