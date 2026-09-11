"""Carrier presence and tracking (stateful across the night).

Every 0.25 s the 18.5-19.3 kHz band is analysed. Carrier presence is decided from device-independent features:
(1) in-band prominence of the 1-s averaged spectrum (peak vs. in-band median excluding +/-60 Hz) with 18/15 dB hysteresis,
(2) frequency continuity (global peak within +/-150 Hz for 1 s). While locked the carrier frequency is smoothed with an EMA
and searched within +/-120 Hz. The legacy out-of-band SNR is still reported for diagnostics."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import os as _os

import numpy as np

LOCK_PROM_DB = float(_os.environ.get("CT_LOCK_PROM", "15.0"))
RELOCK_PROM_DB = float(_os.environ.get("CT_RELOCK_PROM", "18.0"))
HOLD_HOPS = int(_os.environ.get("CT_HOLD_HOPS", "4"))
CONT_HZ = 150.0
AVG_HOPS = 4

CARRIER_BAND = (18500.0, 19300.0)
NOISE_BANDS = ((18350.0, 18480.0), (19320.0, 19440.0), (21300.0, 22500.0))

@dataclass
class CarrierEstimate:
    t: float
    f_raw: float
    f_smooth: float
    snr_db: float
    locked: bool
    prom_db: float = 0.0

@dataclass
class CarrierTracker:
    fs: int
    hop_s: float = 0.25
    tau_s: float = 2.0
    lock_db: float = 20.0
    relock_db: float = 25.0
    search_hz: float = 120.0
    f_smooth: float | None = None
    locked: bool = False
    _buf: np.ndarray = field(default_factory=lambda: np.zeros(0))
    _t_buf: float = 0.0
    _win: np.ndarray | None = None
    _masks: tuple | None = None
    _pbuf: deque = field(default_factory=lambda: deque(maxlen=AVG_HOPS))
    _hold: int = 0
    _bad: int = 0
    _fg_prev: float | None = None

    def _prepare(self):
        n = int(round(self.hop_s * self.fs))
        self._win = np.hanning(n)
        fr = np.fft.rfftfreq(n, 1 / self.fs)
        cb = (fr >= CARRIER_BAND[0]) & (fr <= CARRIER_BAND[1])
        nb = np.zeros_like(cb)
        for lo, hi in NOISE_BANDS:
            nb |= (fr >= lo) & (fr <= hi)
        self._masks = (fr, cb, nb)

    def step(self, x: np.ndarray, t0: float) -> list[CarrierEstimate]:
        """Feed one block; return the estimates for every hop completed within it (timestamps at hop centres)."""
        if self._win is None:
            self._prepare()
        n = len(self._win)
        if len(self._buf) == 0:
            self._t_buf = t0
        self._buf = np.concatenate([self._buf, x])
        out = []
        fr, cb, nb = self._masks
        alpha = 1.0 - np.exp(-self.hop_s / self.tau_s)
        while len(self._buf) >= n:
            seg = self._buf[:n]; self._buf = self._buf[n:]
            tc = self._t_buf + 0.5 * self.hop_s; self._t_buf += self.hop_s
            p = np.abs(np.fft.rfft(seg * self._win)) ** 2
            pc = p[cb]; fc_band = fr[cb]

            self._pbuf.append(pc); pav = np.mean(np.stack(self._pbuf), axis=0)
            ig = int(np.argmax(pav)); fg = float(fc_band[ig])
            excl = np.abs(fc_band - fg) < 60.0
            med = float(np.median(pav[~excl])) if (~excl).any() else float(np.median(pav))
            prom = float(10 * np.log10((pav[ig] + 1e-30) / (med + 1e-30)))
            cont_prev = self._fg_prev is None or abs(fg - self._fg_prev) <= CONT_HZ
            self._fg_prev = fg
            if self.locked and self.f_smooth is not None:
                near = np.abs(fc_band - self.f_smooth) <= self.search_hz
                i = int(np.argmax(np.where(near, pc, -1.0)))
            else:
                i = int(np.argmax(np.where(np.abs(fc_band - fg) <= 60.0, pc, -1.0)))
            f_raw = float(fc_band[i])
            snr = float(10 * np.log10(pc[i] + 1e-30) - np.median(10 * np.log10(p[nb] + 1e-30)))

            if 0 < i < len(pc) - 1:
                y0, y1, y2 = np.log(pc[i - 1] + 1e-30), np.log(pc[i] + 1e-30), np.log(pc[i + 1] + 1e-30)
                den = y0 - 2 * y1 + y2
                if den != 0:
                    f_raw += float(0.5 * (y0 - y2) / den) * (fr[1] - fr[0])
            if self.locked:
                ok = prom >= LOCK_PROM_DB and abs(fg - self.f_smooth) <= CONT_HZ
                if ok:
                    self._bad = 0
                    self.f_smooth = self.f_smooth + alpha * (f_raw - self.f_smooth)
                else:
                    self._bad += 1
                    if self._bad >= HOLD_HOPS:
                        self.locked = False; self._hold = 0; self._bad = 0
            else:
                if prom >= RELOCK_PROM_DB and cont_prev:
                    self._hold += 1
                else:
                    self._hold = 0
                if self._hold >= HOLD_HOPS:
                    self.locked = True; self.f_smooth = f_raw
                if self.f_smooth is None:
                    self.f_smooth = 0.5 * (CARRIER_BAND[0] + CARRIER_BAND[1])
            out.append(CarrierEstimate(tc, f_raw, float(self.f_smooth), snr, self.locked, prom))
        return out
