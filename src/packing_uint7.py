#!/usr/bin/env python3
"""packing_uint7.py: bit-exact 7-bit storage formats for digital filography.

These two formats replace uniform_uint6 and adjusted_uint6:

    uniform_uint7    all 8 values of a thread at 7 bits          56 bits = 7.0 bytes
    adjusted_uint7   x1 y1 x2 y2 at 7 bits, r g b a as RGBA4444   44 bits = 5.5 bytes

A thread is [x1, y1, x2, y2, r, g, b, a], every value normalized to [0, 1].
The quantizer follows the convention of the existing uint8 path:
code = round(value * (2**bits - 1)), decoded value = code / (2**bits - 1).

Threads are written as one continuous bit stream (most significant bit first),
so adjusted_uint7 really costs 5.5 bytes per thread and not 6. The stream is
padded with zero bits to a whole number of bytes at the very end. For an even
thread count the size is exactly count * 5.5 bytes. For an odd count it is one
byte more.

Public API (same shape as the existing packing module):
    UINT7_FORMATS
    bytes_per_thread(tier) -> int | float
    quantize_threads(threads, tier) -> (decoded float32 array (N, 8), packed byte count)
    pack_threads(threads, tier) -> bytes
    unpack_threads(blob, count, tier) -> float32 array (N, 8)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

COORD_FIELDS = 4
COLOUR_FIELDS = 4
FIELDS_PER_THREAD = COORD_FIELDS + COLOUR_FIELDS
# Values may drift slightly outside [0, 1] through float rounding. Anything
# further out than this is a bug upstream and is rejected.
RANGE_TOLERANCE = 1e-3


@dataclass(frozen=True)
class BitLayout:
    """Bit widths of the four coordinates and of the four colour/alpha values."""

    coord_bits: int
    colour_bits: int

    @property
    def bits_per_thread(self) -> int:
        return COORD_FIELDS * self.coord_bits + COLOUR_FIELDS * self.colour_bits


LAYOUTS: dict[str, BitLayout] = {
    "uniform_uint7": BitLayout(coord_bits=7, colour_bits=7),
    "adjusted_uint7": BitLayout(coord_bits=7, colour_bits=4),
}
UINT7_FORMATS: tuple[str, ...] = tuple(LAYOUTS)


def _layout(tier: str) -> BitLayout:
    try:
        return LAYOUTS[tier]
    except KeyError:
        raise ValueError(f"unknown 7-bit format {tier!r}, expected one of {UINT7_FORMATS}") from None


def bytes_per_thread(tier: str) -> int | float:
    """Average storage per thread in bytes: 7 for uniform_uint7, 5.5 for adjusted_uint7.

    Whole-byte formats return an int so CSV output matches the other formats.
    """
    bits = _layout(tier).bits_per_thread
    return bits // 8 if bits % 8 == 0 else bits / 8


def packed_size(count: int, tier: str) -> int:
    """Exact size in bytes of `count` packed threads (ceil to a whole byte)."""
    if count < 0:
        raise ValueError("count must not be negative")
    return (count * _layout(tier).bits_per_thread + 7) // 8


def _validated(threads: np.ndarray) -> np.ndarray:
    arr = np.asarray(threads, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != FIELDS_PER_THREAD:
        raise ValueError(f"threads must have shape (N, {FIELDS_PER_THREAD}), got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("threads contain NaN or infinite values")
    if arr.size and (arr.min() < -RANGE_TOLERANCE or arr.max() > 1.0 + RANGE_TOLERANCE):
        raise ValueError("thread values must be normalized to [0, 1]")
    return np.clip(arr, 0.0, 1.0)


def _to_bits(codes: np.ndarray, bits: int) -> np.ndarray:
    """Integer codes of shape (...) to bit arrays of shape (..., bits), MSB first."""
    shifts = np.arange(bits - 1, -1, -1, dtype=np.uint16)
    return ((codes.astype(np.uint16)[..., None] >> shifts) & 1).astype(np.uint8)


def _from_bits(bit_array: np.ndarray) -> np.ndarray:
    """Bit arrays of shape (..., bits), MSB first, back to integer codes of shape (...)."""
    bits = bit_array.shape[-1]
    weights = 1 << np.arange(bits - 1, -1, -1, dtype=np.int64)
    return (bit_array.astype(np.int64) * weights).sum(axis=-1)


def pack_threads(threads: np.ndarray, tier: str) -> bytes:
    """Quantize `threads` and write them as a continuous bit stream."""
    layout = _layout(tier)
    arr = _validated(threads)
    n = arr.shape[0]
    coord_levels = (1 << layout.coord_bits) - 1
    colour_levels = (1 << layout.colour_bits) - 1
    coord_codes = np.rint(arr[:, :COORD_FIELDS] * coord_levels)
    colour_codes = np.rint(arr[:, COORD_FIELDS:] * colour_levels)
    stream = np.concatenate(
        [
            _to_bits(coord_codes, layout.coord_bits).reshape(n, COORD_FIELDS * layout.coord_bits),
            _to_bits(colour_codes, layout.colour_bits).reshape(n, COLOUR_FIELDS * layout.colour_bits),
        ],
        axis=1,
    )
    return np.packbits(stream.reshape(-1)).tobytes()


def unpack_threads(blob: bytes, count: int, tier: str) -> np.ndarray:
    """Read `count` threads from a bit stream written by pack_threads."""
    layout = _layout(tier)
    expected = packed_size(count, tier)
    if len(blob) != expected:
        raise ValueError(f"{tier}: expected {expected} bytes for {count} threads, got {len(blob)}")
    per_thread = layout.bits_per_thread
    stream = np.unpackbits(np.frombuffer(blob, dtype=np.uint8), count=count * per_thread)
    stream = stream.reshape(count, per_thread)
    split = COORD_FIELDS * layout.coord_bits
    coord_bits = stream[:, :split].reshape(count, COORD_FIELDS, layout.coord_bits)
    colour_bits = stream[:, split:].reshape(count, COLOUR_FIELDS, layout.colour_bits)
    coord_levels = (1 << layout.coord_bits) - 1
    colour_levels = (1 << layout.colour_bits) - 1
    out = np.empty((count, FIELDS_PER_THREAD), dtype=np.float32)
    out[:, :COORD_FIELDS] = _from_bits(coord_bits) / coord_levels
    out[:, COORD_FIELDS:] = _from_bits(colour_bits) / colour_levels
    return out


def quantize_threads(threads: np.ndarray, tier: str) -> tuple[np.ndarray, int]:
    """Real round trip: floats -> packed bytes -> floats.

    Returns the decoded threads (what a receiver would draw) and the number of
    bytes the packed matrix occupies.
    """
    arr = np.asarray(threads)
    blob = pack_threads(arr, tier)
    return unpack_threads(blob, arr.shape[0], tier), len(blob)