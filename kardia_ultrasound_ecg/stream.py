"""Block-wise reader for whole-night WAV files (memory-mapped). Downstream stages are stateful and consume 4-s blocks."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from scipy.io import wavfile

@dataclass
class Block:
    """One block: `x` float64 normalised to +/-1, `t0` time of the first sample (s), `fs` sample rate."""

    x: np.ndarray
    t0: float
    fs: int
    index: int
    is_last: bool

class WavStream:
    """Memory-mapped whole-night WAV; `blocks(block_s)` yields consecutive blocks."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.fs, data = wavfile.read(str(self.path), mmap=True)
        if data.ndim > 1:
            data = data[:, 0]
        self._data = data
        info = np.iinfo(data.dtype) if np.issubdtype(data.dtype, np.integer) else None
        self._scale = float(max(abs(info.min), info.max)) if info else 1.0
        self.n = len(data)

    @property
    def duration_s(self) -> float:
        return self.n / self.fs

    def blocks(self, block_s: float = 4.0) -> Iterator[Block]:
        n_blk = int(round(block_s * self.fs))
        k = 0
        for start in range(0, self.n, n_blk):
            stop = min(start + n_blk, self.n)
            x = np.asarray(self._data[start:stop], dtype=np.float64) / self._scale
            yield Block(x=x, t0=start / self.fs, fs=self.fs, index=k, is_last=stop >= self.n)
            k += 1
