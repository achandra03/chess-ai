"""On-disk cache of pre-encoded perspective-v2 positions.

Layout (all little-endian, row-aligned across files):
  boards.u8.bin    (rows, 192)  uint8    board planes bit-packed (1536 bits)
  metadata.f16.bin (rows, 24)   float16  scalar metadata
  targets.f32.bin  (rows,)      float32  side-to-move pawns, clipped
  splits.u8.bin    (rows,)      uint8    0=train, 1=validation, 2=test
  sample.f32.bin   (rows,)      float32  deterministic sample hash fraction
  manifest.json

The horizontal mirror of a position is a fixed permutation of the feature
vector, so only the unmirrored encoding is stored and augmentation is applied
at load time. This module must stay TensorFlow-free: encoder worker processes
import it.
"""

import json
from pathlib import Path

import numpy as np

from board_features import (
	ATTACK_BOARD_FEATURE_SIZE,
	BOARD_SIZE,
	PERSPECTIVE_METADATA_SIZE,
	PERSPECTIVE_V2_FEATURE_SIZE,
	PERSPECTIVE_V2_METADATA_SIZE,
)


SCHEMA_VERSION = 1
CACHE_ARCHITECTURE = "perspective-transformer-v2"
BOARD_BITS = ATTACK_BOARD_FEATURE_SIZE
BOARD_PACKED_BYTES = BOARD_BITS // 8
METADATA_SIZE = PERSPECTIVE_V2_METADATA_SIZE
FEATURE_SIZE = PERSPECTIVE_V2_FEATURE_SIZE
SQUARE_PLANE_COUNT = BOARD_BITS // (BOARD_SIZE * BOARD_SIZE)
SPLIT_CODES = {"train": 0, "validation": 1, "test": 2}

MANIFEST_NAME = "manifest.json"
CACHE_FILES = {
	"boards": ("boards.u8.bin", np.uint8, (BOARD_PACKED_BYTES,)),
	"metadata": ("metadata.f16.bin", np.float16, (METADATA_SIZE,)),
	"targets": ("targets.f32.bin", np.float32, ()),
	"splits": ("splits.u8.bin", np.uint8, ()),
	"sample": ("sample.f32.bin", np.float32, ()),
}


def build_mirror_permutation():
	"""Index permutation mapping an encoded position to its file-mirrored twin.

	mirrored_features = features[permutation]
	"""
	permutation = np.arange(FEATURE_SIZE, dtype=np.int64)
	board_indices = np.arange(BOARD_BITS, dtype=np.int64).reshape(
		BOARD_SIZE, BOARD_SIZE, SQUARE_PLANE_COUNT
	)
	permutation[:BOARD_BITS] = board_indices[:, ::-1, :].reshape(-1)
	# Mirroring swaps kingside/queenside castling flags for both players
	# (metadata slots 2..5); checks and the extra scalars are unaffected.
	base = BOARD_BITS
	permutation[base + 2] = base + 3
	permutation[base + 3] = base + 2
	permutation[base + 4] = base + 5
	permutation[base + 5] = base + 4
	return permutation


MIRROR_PERMUTATION = build_mirror_permutation()


def pack_features(features):
	"""Split (n, FEATURE_SIZE) float32 features into packed boards + f16 metadata."""
	features = np.asarray(features, dtype=np.float32)
	if features.ndim == 1:
		features = features[np.newaxis, :]
	if features.shape[1] != FEATURE_SIZE:
		raise ValueError(f"expected {FEATURE_SIZE} features, got {features.shape[1]}")
	boards = np.packbits(features[:, :BOARD_BITS] > 0.5, axis=1)
	metadata = features[:, BOARD_BITS:].astype(np.float16)
	return boards, metadata


def unpack_features(boards, metadata, mirror_mask=None):
	"""Rebuild (n, FEATURE_SIZE) float32 features from packed rows.

	mirror_mask optionally selects rows to replace with their horizontal
	mirror.
	"""
	boards = np.ascontiguousarray(boards, dtype=np.uint8)
	features = np.empty((boards.shape[0], FEATURE_SIZE), dtype=np.float32)
	features[:, :BOARD_BITS] = np.unpackbits(boards, axis=1)
	features[:, BOARD_BITS:] = metadata.astype(np.float32)
	if mirror_mask is not None:
		mirror_mask = np.asarray(mirror_mask, dtype=bool)
		if mirror_mask.any():
			features[mirror_mask] = features[mirror_mask][:, MIRROR_PERMUTATION]
	return features


class CacheWriter:
	"""Sequentially writes encoded rows into preallocated cache memmaps."""

	def __init__(self, directory, rows):
		self.directory = Path(directory)
		self.directory.mkdir(parents=True, exist_ok=True)
		self.rows = int(rows)
		self.cursor = 0
		self.arrays = {}
		for key, (file_name, dtype, row_shape) in CACHE_FILES.items():
			self.arrays[key] = np.memmap(
				self.directory / file_name,
				dtype=dtype,
				mode="w+",
				shape=(self.rows,) + row_shape,
			)

	def append(self, boards, metadata, targets, splits, sample):
		count = boards.shape[0]
		if self.cursor + count > self.rows:
			raise ValueError("cache writer overflow")
		stop = self.cursor + count
		self.arrays["boards"][self.cursor:stop] = boards
		self.arrays["metadata"][self.cursor:stop] = metadata
		self.arrays["targets"][self.cursor:stop] = targets
		self.arrays["splits"][self.cursor:stop] = splits
		self.arrays["sample"][self.cursor:stop] = sample
		self.cursor = stop

	def finalize(self, manifest):
		if self.cursor != self.rows:
			raise ValueError(
				f"cache writer wrote {self.cursor} of {self.rows} rows"
			)
		for array in self.arrays.values():
			array.flush()
		manifest = dict(manifest)
		manifest["schema_version"] = SCHEMA_VERSION
		manifest["architecture"] = CACHE_ARCHITECTURE
		manifest["rows"] = self.rows
		with (self.directory / MANIFEST_NAME).open("w", encoding="utf-8") as file:
			json.dump(manifest, file, indent=2)
			file.write("\n")


class EncodedCache:
	"""Read-only view of an encoded dataset directory."""

	def __init__(self, directory):
		self.directory = Path(directory)
		manifest_path = self.directory / MANIFEST_NAME
		if not manifest_path.is_file():
			raise FileNotFoundError(
				f"no cache manifest at {manifest_path}; "
				"build the cache with encode_dataset.py"
			)
		with manifest_path.open(encoding="utf-8") as file:
			self.manifest = json.load(file)
		if self.manifest.get("schema_version") != SCHEMA_VERSION:
			raise ValueError(
				f"cache schema {self.manifest.get('schema_version')} does not "
				f"match expected {SCHEMA_VERSION}: {self.directory}"
			)
		if self.manifest.get("architecture") != CACHE_ARCHITECTURE:
			raise ValueError(
				f"cache was built for {self.manifest.get('architecture')}, "
				f"not {CACHE_ARCHITECTURE}: {self.directory}"
			)

		self.rows = int(self.manifest["rows"])
		self.max_abs_pawns = float(self.manifest["max_abs_pawns"])
		self.split_seed = self.manifest["split_seed"]
		self.sample_seed = self.manifest["sample_seed"]
		self.train_only = bool(self.manifest.get("train_only", False))
		for key, (file_name, dtype, row_shape) in CACHE_FILES.items():
			setattr(
				self,
				key,
				np.memmap(
					self.directory / file_name,
					dtype=dtype,
					mode="r",
					shape=(self.rows,) + row_shape,
				),
			)

	def split_indices(self, split, sample_fraction=1.0):
		mask = np.asarray(self.splits) == SPLIT_CODES[split]
		if sample_fraction < 1.0:
			mask &= np.asarray(self.sample) < sample_fraction
		return np.flatnonzero(mask).astype(np.int64)
