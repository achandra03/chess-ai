"""Validation error buckets, shared by both data paths."""


def empty_bucket_metric():
	return {"count": 0, "mae_sum": 0.0}


def new_target_buckets(extra_names=()):
	buckets = {
		"abs_eval_lt_1": empty_bucket_metric(),
		"abs_eval_1_to_3": empty_bucket_metric(),
		"abs_eval_3_to_6": empty_bucket_metric(),
		"abs_eval_gt_6": empty_bucket_metric(),
	}
	for name in extra_names:
		buckets[name] = empty_bucket_metric()
	return buckets


def add_bucket_error(buckets, name, error):
	bucket = buckets[name]
	bucket["count"] += 1
	bucket["mae_sum"] += float(error)


def target_bucket_name(target):
	abs_target = abs(target)
	if abs_target < 1.0:
		return "abs_eval_lt_1"
	if abs_target < 3.0:
		return "abs_eval_1_to_3"
	if abs_target < 6.0:
		return "abs_eval_3_to_6"
	return "abs_eval_gt_6"


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
