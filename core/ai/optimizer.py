"""Learning-rate schedule, optimizer and loss used to compile the model."""

import inspect
import math

import tensorflow as tf


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
			step = tf.cast(step, tf.float32) + tf.cast(self.step_offset, tf.float32)
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


def warmup_steps_for_training(config):
	total_steps = config.total_train_steps
	steps_per_epoch = config.train_steps_per_epoch
	if total_steps < 1:
		return 0
	if config.warmup_steps is not None:
		return min(config.warmup_steps, max(total_steps - 1, 0))
	if config.warmup_epochs is not None and steps_per_epoch:
		return min(
			int(round(config.warmup_epochs * steps_per_epoch)),
			max(total_steps - 1, 0),
		)
	return min(int(round(config.warmup_ratio * total_steps)), max(total_steps - 1, 0))


def learning_rate_for_optimizer(config):
	if config.lr_schedule == "constant":
		return config.learning_rate
	if config.total_train_steps < 1:
		return config.learning_rate
	return WarmupCosineDecay(
		initial_learning_rate=config.learning_rate,
		decay_steps=config.total_train_steps,
		warmup_steps=warmup_steps_for_training(config),
		min_learning_rate=config.min_learning_rate,
		step_offset=config.lr_schedule_step_offset,
	)


def build_optimizer(config):
	learning_rate = learning_rate_for_optimizer(config)
	clip_kwargs = {}
	if config.gradient_clipnorm > 0:
		clip_kwargs["clipnorm"] = config.gradient_clipnorm
	if config.optimizer == "adamw":
		adamw_optimizer = getattr(tf.keras.optimizers, "AdamW", None)
		if adamw_optimizer is None:
			experimental_optimizers = getattr(
				tf.keras.optimizers, "experimental", None
			)
			adamw_optimizer = getattr(experimental_optimizers, "AdamW", None)
		if adamw_optimizer is None:
			raise RuntimeError(
				"AdamW is not available in this TensorFlow/Keras install. "
				"Set optimizer to 'adam' or install TensorFlow 2.10+."
			)
		optimizer_kwargs = {
			"learning_rate": learning_rate,
			"weight_decay": config.weight_decay,
			**clip_kwargs,
		}
		if "jit_compile" in inspect.signature(adamw_optimizer).parameters:
			optimizer_kwargs["jit_compile"] = False
		return adamw_optimizer(**optimizer_kwargs)
	return tf.keras.optimizers.Adam(learning_rate=learning_rate, **clip_kwargs)


def loss_for_training(config):
	if config.loss == "huber":
		return tf.keras.losses.Huber(delta=config.huber_delta)
	return tf.keras.losses.MeanAbsoluteError()


def learning_rate_description(config):
	if config.lr_schedule == "constant":
		return f"lr={config.learning_rate:g}"
	if config.total_train_steps < 1:
		return f"lr={config.learning_rate:g}->schedule pending"
	offset_description = ""
	if config.lr_schedule_step_offset:
		offset_description = f", step_offset={config.lr_schedule_step_offset}"
	return (
		f"lr={config.learning_rate:g}->{config.min_learning_rate:g} "
		f"warmup-cosine(total_steps={config.total_train_steps}, "
		f"warmup_steps={warmup_steps_for_training(config)}{offset_description})"
	)


def optimizer_description(config):
	clip_description = ""
	if config.gradient_clipnorm > 0:
		clip_description = f", clipnorm={config.gradient_clipnorm:g}"
	if config.optimizer == "adamw":
		return (
			f"AdamW({learning_rate_description(config)}, "
			f"weight_decay={config.weight_decay:g}{clip_description})"
		)
	return f"Adam({learning_rate_description(config)}{clip_description})"


def loss_description(config):
	if config.loss == "huber":
		return f"Huber(delta={config.huber_delta:g})"
	return "MAE"
