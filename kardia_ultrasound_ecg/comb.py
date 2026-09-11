"""Comb-interference canceller: detects equally spaced narrow spectral lines (120.17-Hz spacing, mains-harmonic comb
coupled into some recordings) and removes them with cascaded 3-Hz notch filters whose state persists across blocks."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks

COMB_SPACING = 120.17
BAND = (17500.0, 22500.0)
NFFT = 2 ** 15

def detect_lines(x: np.ndarray, fs: int, carrier_hz: float | None) -> list[float]:
    """Detect equally spaced narrow peaks on >= 30 s of audio. Returns the line frequencies (Hz)."""
    n = len(x) // NFFT
    if n < 10:
        return []
    f = np.fft.rfftfreq(NFFT, 1 / fs); m = (f >= BAND[0]) & (f <= BAND[1])
    win = np.hanning(NFFT)
    P = np.median([np.abs(np.fft.rfft(x[i * NFFT:(i + 1) * NFFT] * win))[m] ** 2 for i in range(n)], axis=0)
    db = 10 * np.log10(P + 1e-20); base = np.percentile(db, 30)
    pk, _ = find_peaks(db, prominence=6)
    out = []
    for i in pk:
        fr = float(f[m][i])
        if any(abs(fr - e) < 60 for e in (18000.0, 19500.0, 20000.0, 21000.0)):
            continue
        if carrier_hz is not None and abs(fr - carrier_hz) < 40:
            continue
        if abs(fr / COMB_SPACING - round(fr / COMB_SPACING)) * COMB_SPACING < 4 and db[i] - base > 8:

            if 0 < i < len(db) - 1:
                y0, y1, y2 = db[i - 1], db[i], db[i + 1]; den = y0 - 2 * y1 + y2
                if den != 0:
                    fr += 0.5 * (y0 - y2) / den * (f[1] - f[0])
            out.append(fr)
    return out if len(out) >= 5 else []

@dataclass
class CombCanceller:
    """Cascaded narrow notches (one biquad per line, ~3 Hz -3 dB bandwidth) with state carried across blocks; lines are detected on the first 60 s and re-estimated every 10 min (detected lines are kept for the night)."""
    fs: int
    lines: list = field(default_factory=list)
    notch_bw_hz: float = 3.0
    _sos: np.ndarray | None = None
    _zi: np.ndarray | None = None
    _probe: list = field(default_factory=list)
    _probe_len: int = 0
    _t_last_detect: float = -1e9
    redetect_s: float = 600.0
    active: bool = False

    def _build(self, carrier_hz: float | None):
        from scipy.signal import iirnotch, tf2sos
        use = [f for f in self.lines if carrier_hz is None or abs(f - carrier_hz) >= 40]
        if not use:
            self._sos, self._zi, self.active = None, None, False; return
        sos = np.vstack([tf2sos(*iirnotch(f, f / self.notch_bw_hz, fs=self.fs)) for f in use])
        if self._sos is not None and self._sos.shape == sos.shape:
            self._sos = sos
        else:
            self._sos, self._zi = sos, np.zeros((sos.shape[0], 2))
        self.active = True

    def step(self, x: np.ndarray, t0: float, carrier_hz: float | None) -> np.ndarray:
        from scipy.signal import sosfilt
        if (t0 - self._t_last_detect) >= self.redetect_s:
            self._probe.append(x); self._probe_len += len(x)
            if self._probe_len >= 60 * self.fs:
                found = detect_lines(np.concatenate(self._probe), self.fs, carrier_hz)
                self._probe, self._probe_len = [], 0; self._t_last_detect = t0
                if found:
                    new = [f for f in found if not any(abs(f - g) < 3 for g in self.lines)]
                    if new or not self.active:
                        self.lines = sorted(self.lines + new); self._build(carrier_hz)
        if not self.active or self._sos is None:
            return x
        y, self._zi = sosfilt(self._sos, x, zi=self._zi)
        return y
