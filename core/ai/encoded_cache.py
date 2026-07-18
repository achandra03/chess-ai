"""On-disk cache of pre-encoded perspective-v3 positions.

Layout (all little-endian, row-aligned across files):
  boards.u8.bin    (rows, 224)  uint8    board planes bit-packed (1792 bits)
  metadata.f16.bin (rows, 32)   float16  scalar metadata
  targets.f32.bin  (rows,)      float32  side-to-move pawns, clipped
  splits.u8.bin    (rows,)      uint8    0=train, 1=validation, 2=test
  sample.f32.bin   (rows,)      float32  deterministic sample hash fraction
  manifest.json

Only the unmirrored encoding is stored; the horizontal mirror is a fixed
index permutation applied at load time. Must stay TensorFlow-free:
encoder worker processes import it.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from board_features import (
	BOARD_SIZE,
	PERSPECTIVE_V3_BOARD_FEATURE_SIZE,
	PERSPECTIVE_V3_METADATA_SIZE,
)


SCHEMA_VERSION = 1
CACHE_ARCHITECTURE_V3 = "perspective-transformer-v3"
SPLIT_CODES = {"train": 0, "validation": 1, "test": 2}
MANIFEST_NAME = "manifest.json"


def _build_mirror_permutation(board_bits, metadata_size):
	permutation = np.arange(board_bits + metadata_size, dtype=np.int64)
	board_indices = np.arange(board_bits, dtype=np.int64).reshape(
		BOARD_SIZE, BOARD_SIZE, board_bits // (BOARD_SIZE * BOARD_SIZE)
	)
	permutation[:board_bits] = board_indices[:, ::-1, :].reshape(-1)
	# Mirroring swaps kingside/queenside castling flags for both players
	# (metadata slots 2..5); every other scalar is file-mirror invariant.
	base = board_bits
	permutation[base + 2] = base + 3
	permutation[base + 3] = base + 2
	permutation[base + 4] = base + 5
	permutation[base + 5] = base + 4
	return permutation


def _make_spec(board_bits, metadata_size):
	if board_bits % 8 != 0:
		raise ValueError(f"board bits {board_bits} not byte-aligned")
	return SimpleNamespace(
		board_bits=board_bits,
		board_packed_bytes=board_bits // 8,
		metadata_size=metadata_size,
		feature_size=board_bits + metadata_size,
		square_plane_count=board_bits // (BOARD_SIZE * BOARD_SIZE),
		mirror_permutation=_build_mirror_permutation(board_bits, metadata_size),
	)


CACHE_SPECS = {
	CACHE_ARCHITECTURE_V3: _make_spec(
		PERSPECTIVE_V3_BOARD_FEATURE_SIZE, PERSPECTIVE_V3_METADATA_SIZE
	),
}


def cache_files(spec):
	return {
		"boards": ("boards.u8.bin", np.uint8, (spec.board_packed_bytes,)),
		"metadata": ("metadata.f16.bin", np.float16, (spec.metadata_size,)),
		"targets": ("targets.f32.bin", np.float32, ()),
		"splits": ("splits.u8.bin", np.uint8, ()),
		"sample": ("sample.f32.bin", np.float32, ()),
	}


def pack_features(features, architecture=CACHE_ARCHITECTURE_V3):
	spec = CACHE_SPECS[architecture]
	features = np.asarray(features, dtype=np.float32)
	if features.ndim == 1:
		features = features[np.newaxis, :]
	if features.shape[1] != spec.feature_size:
		raise ValueError(
			f"expected {spec.feature_size} features, got {features.shape[1]}"
		)
	boards = np.packbits(features[:, :spec.board_bits] > 0.5, axis=1)
	metadata = features[:, spec.board_bits:].astype(np.float16)
	return boards, metadata


def unpack_features(boards, metadata, mirror_mask=None, architecture=CACHE_ARCHITECTURE_V3):
	spec = CACHE_SPECS[architecture]
	boards = np.ascontiguousarray(boards, dtype=np.uint8)
	features = np.empty((boards.shape[0], spec.feature_size), dtype=np.float32)
	features[:, :spec.board_bits] = np.unpackbits(boards, axis=1)
	features[:, spec.board_bits:] = metadata.astype(np.float32)
	if mirror_mask is not None:
		mirror_mask = np.asarray(mirror_mask, dtype=bool)
		if mirror_mask.any():
			features[mirror_mask] = features[mirror_mask][
				:, spec.mirror_permutation
			]
	return features


class CacheWriter:
	"""Sequentially writes encoded rows into preallocated cache memmaps."""

	def __init__(self, directory, rows, architecture=CACHE_ARCHITECTURE_V3):
		self.directory = Path(directory)
		self.directory.mkdir(parents=True, exist_ok=True)
		self.rows = int(rows)
		self.architecture = architecture
		self.spec = CACHE_SPECS[architecture]
		self.cursor = 0
		self.arrays = {}
		for key, (file_name, dtype, row_shape) in cache_files(self.spec).items():
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
		manifest["architecture"] = self.architecture
		manifest["rows"] = self.rows
		with (self.directory / MANIFEST_NAME).open("w", encoding="utf-8") as file:
			json.dump(manifest, file, indent=2)
			file.write("\n")


class EncodedCache:
	"""Read-only view of an encoded dataset directory."""

	def __init__(self, directory, architecture=None):
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
		self.architecture = self.manifest.get("architecture")
		if self.architecture not in CACHE_SPECS:
			raise ValueError(
				f"cache was built for {self.architecture}, which is not a "
				f"known cache architecture: {self.directory}"
			)
		if architecture is not None and self.architecture != architecture:
			raise ValueError(
				f"cache was built for {self.architecture}, "
				f"not {architecture}: {self.directory}"
			)
		self.spec = CACHE_SPECS[self.architecture]

		self.rows = int(self.manifest["rows"])
		self.max_abs_pawns = float(self.manifest["max_abs_pawns"])
		self.split_seed = self.manifest["split_seed"]
		self.sample_seed = self.manifest["sample_seed"]
		self.train_only = bool(self.manifest.get("train_only", False))
		for key, (file_name, dtype, row_shape) in cache_files(self.spec).items():
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
