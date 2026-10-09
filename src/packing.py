#!/usr/bin/env python3
"""packing.py: byte-exact packed storage formats for digital filography threads.

A thread is 8 values in [0, 1]:  x1 y1 x2 y2 | r g b a

Two families are compared, each at three precisions:

    Uniform   all 8 values use the same precision
    Adjusted  the 4 coordinates use the chosen precision,
              colour + alpha use RGBA4444 (4 bits per channel, 16 bits total)

    precision   bits per value   notes
    fp16        16               IEEE half float, little endian
    uint8        8               round(v * 255)
    uint6        6               round(v * 63), bit-packed (no wasted bits)

Bytes per thread (every record is a whole number of bytes, so a thread
never straddles another thread's bytes):

    uniform_fp16   8 x 16 bit            = 128 bit = 16 B
    uniform_uint8  8 x  8 bit            =  64 bit =  8 B
    uniform_uint6  8 x  6 bit            =  48 bit =  6 B
    adjusted_fp16  4 x 16 bit + 16 bit   =  80 bit = 10 B
    adjusted_uint8 4 x  8 bit + 16 bit   =  48 bit =  6 B
    adjusted_uint6 4 x  6 bit + 16 bit   =  40 bit =  5 B

Layout of one record: coordinates first (x1 y1 x2 y2), then colour. Integer
codes are written most significant bit first. RGBA4444 is one 16 bit word
r<<12 | g<<8 | b<<4 | a.

Run `python src/packing.py` to print the size table and check that every
format survives a pack -> bytes -> unpack round trip.

Requires: numpy
"""

from __future__ import annotations

import sys
from typing import Union

import numpy as np

FAMILIES = ("uniform", "adjusted")
PRECISIONS = ("fp16", "uint8", "uint6")
FORMATS = tuple(f"{fam}_{prec}" for fam in FAMILIES for prec in PRECISIONS)

PRECISION_BITS = {"fp16": 16, "uint8": 8, "uint6": 6}
COLOR_BITS = 4                      # RGBA4444 in the adjusted family
N_COORD = 4                         # x1 y1 x2 y2
N_COLOR = 4                         # r g b a

PackedLike = Union[np.ndarray, bytes, bytearray, memoryview]


# -----------------------------------------------------------------------------
# Format bookkeeping
# -----------------------------------------------------------------------------
def parse_format(name: str) -> tuple[str, str]:
    """Split 'adjusted_uint6' into ('adjusted', 'uint6'). Raises on unknown names."""
    family, _, precision = name.partition("_")
    if family not in FAMILIES or precision not in PRECISIONS:
        raise ValueError(f"Unknown format {name!r}. Choose from: {', '.join(FORMATS)}")
    return family, precision


def bits_per_thread(name: str) -> int:
    family, precision = parse_format(name)
    bits = PRECISION_BITS[precision]
    if family == "uniform":
        return (N_COORD + N_COLOR) * bits
    return N_COORD * bits + N_COLOR * COLOR_BITS


def bytes_per_thread(name: str) -> int:
    bits = bits_per_thread(name)
    if bits % 8:
        raise ValueError(f"{name} needs {bits} bits per thread, which is not a whole number of bytes")
    return bits // 8


# -----------------------------------------------------------------------------
# Integer codes <-> bits
# -----------------------------------------------------------------------------
def _to_codes(values: np.ndarray, bits: int) -> np.ndarray:
    """Quantize [0, 1] floats to unsigned integer codes with 2**bits - 1 steps."""
    levels = (1 << bits) - 1
    return np.clip(np.rint(values.astype(np.float64) * levels), 0, levels).astype(np.uint32)


def _from_codes(codes: np.ndarray, bits: int) -> np.ndarray:
    levels = (1 << bits) - 1
    return (codes.astype(np.float32) / np.float32(levels)).astype(np.float32)


def _pack_codes(codes: np.ndarray, bits: int) -> np.ndarray:
    """(N, k) integer codes -> (N, k * bits / 8) bytes, most significant bit first."""
    n, k = codes.shape
    total = k * bits
    if total % 8:
        raise ValueError(f"{k} values x {bits} bits = {total} bits is not a whole number of bytes")
    shifts = np.arange(bits - 1, -1, -1, dtype=np.uint32)
    bit_matrix = ((codes[:, :, None] >> shifts) & 1).astype(np.uint8)
    return np.packbits(bit_matrix.reshape(n, total), axis=1)


def _unpack_codes(buf: np.ndarray, k: int, bits: int) -> np.ndarray:
    """(N, k * bits / 8) bytes -> (N, k) integer codes."""
    n = buf.shape[0]
    bit_matrix = np.unpackbits(buf, axis=1).reshape(n, k, bits).astype(np.uint32)
    weights = np.uint32(1) << np.arange(bits - 1, -1, -1, dtype=np.uint32)
    return (bit_matrix * weights).sum(axis=2, dtype=np.uint32)


def _pack_values(values: np.ndarray, precision: str) -> np.ndarray:
    """(N, k) floats in [0, 1] -> (N, bytes) for one block at the given precision."""
    n, k = values.shape
    if precision == "fp16":
        half = np.ascontiguousarray(values.astype("<f2"))
        return half.view(np.uint8).reshape(n, k * 2)
    return _pack_codes(_to_codes(values, PRECISION_BITS[precision]), PRECISION_BITS[precision])


def _unpack_values(buf: np.ndarray, k: int, precision: str) -> np.ndarray:
    n = buf.shape[0]
    if precision == "fp16":
        half = np.ascontiguousarray(buf).view("<f2").reshape(n, k)
        return half.astype(np.float32)
    bits = PRECISION_BITS[precision]
    return _from_codes(_unpack_codes(buf, k, bits), bits)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def pack_threads(threads: np.ndarray, fmt: str) -> np.ndarray:
    """Pack an (N, 8) float matrix into an (N, bytes_per_thread(fmt)) uint8 matrix.

    `packed.tobytes()` is the exact content of a file or network message that
    stores the threads in this format.
    """
    family, precision = parse_format(fmt)
    threads = np.asarray(threads)
    if threads.ndim != 2 or threads.shape[1] != 8:
        raise ValueError(f"threads must have shape (N, 8), got {threads.shape}")

    coords = threads[:, :N_COORD]
    colour = threads[:, N_COORD:]
    if family == "uniform":
        packed = _pack_values(threads, precision)
    else:
        packed = np.concatenate(
            [_pack_values(coords, precision), _pack_codes(_to_codes(colour, COLOR_BITS), COLOR_BITS)],
            axis=1,
        )

    expected = bytes_per_thread(fmt)
    if packed.shape != (len(threads), expected):
        raise AssertionError(f"{fmt}: packed shape {packed.shape}, expected ({len(threads)}, {expected})")
    return packed


def unpack_threads(packed: PackedLike, fmt: str) -> np.ndarray:
    """Inverse of pack_threads: bytes -> (N, 8) float32 matrix in [0, 1]."""
    family, precision = parse_format(fmt)
    width = bytes_per_thread(fmt)
    buf = np.frombuffer(packed, dtype=np.uint8) if not isinstance(packed, np.ndarray) else packed
    if buf.dtype != np.uint8:
        raise ValueError("packed data must be uint8")
    if buf.size % width:
        raise ValueError(f"{fmt}: {buf.size} bytes is not a multiple of {width} bytes per thread")
    buf = np.ascontiguousarray(buf.reshape(-1, width))

    if family == "uniform":
        return _unpack_values(buf, N_COORD + N_COLOR, precision)

    coord_bytes = width - (N_COLOR * COLOR_BITS) // 8
    coords = _unpack_values(buf[:, :coord_bytes], N_COORD, precision)
    colour = _from_codes(_unpack_codes(buf[:, coord_bytes:], N_COLOR, COLOR_BITS), COLOR_BITS)
    return np.concatenate([coords, colour], axis=1).astype(np.float32)


def quantize_threads(threads: np.ndarray, fmt: str) -> tuple[np.ndarray, int]:
    """Pack, then unpack. Returns (threads as the decoder sees them, packed size in bytes).

    The threads that go into the renderer have really been through the packed
    byte representation, so the storage size reported with them is not an estimate.
    """
    packed = pack_threads(threads, fmt)
    return unpack_threads(packed, fmt), int(packed.nbytes)


# -----------------------------------------------------------------------------
# Self test: python src/packing.py
# -----------------------------------------------------------------------------
def _max_error_bound(fmt: str) -> float:
    """Largest expected |decoded - original| for values in [0, 1]."""
    family, precision = parse_format(fmt)
    coord = 2.0 ** -11 if precision == "fp16" else 0.5 / ((1 << PRECISION_BITS[precision]) - 1)
    if family == "uniform":
        return coord
    return max(coord, 0.5 / ((1 << COLOR_BITS) - 1))


def main() -> int:
    rng = np.random.default_rng(0)
    threads = rng.random((10000, 8), dtype=np.float32)
    threads[:, 7] = 1.0                       # the benchmark draws opaque threads

    print(f"{'format':<16}{'bits/thread':>12}{'bytes/thread':>14}{'KB @ N=10000':>14}{'max error':>12}{'bound':>10}")
    ok = True
    for fmt in FORMATS:
        packed = pack_threads(threads, fmt)
        raw = packed.tobytes()
        decoded = unpack_threads(raw, fmt)                 # decode from real bytes, not the array
        err = float(np.abs(decoded - threads).max())
        bound = _max_error_bound(fmt)
        size_ok = len(raw) == 10000 * bytes_per_thread(fmt)
        again = pack_threads(decoded, fmt).tobytes() == raw   # decoded values re-encode to the same bytes
        good = size_ok and again and err <= bound * 1.0001
        ok &= good
        print(
            f"{fmt:<16}{bits_per_thread(fmt):>12}{bytes_per_thread(fmt):>14}"
            f"{len(raw) / 1000.0:>14.1f}{err:>12.5f}{bound:>10.5f}  {'OK' if good else 'FAIL'}"
        )

    empty = np.zeros((0, 8), dtype=np.float32)
    for fmt in FORMATS:
        ok &= unpack_threads(pack_threads(empty, fmt), fmt).shape == (0, 8)

    print("\nAll formats round-trip through packed bytes." if ok else "\nSELF TEST FAILED.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
