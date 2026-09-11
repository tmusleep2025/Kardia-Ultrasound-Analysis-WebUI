"""Quadrature FM demodulation of the KardiaMobile carrier into an ECG-like waveform at 400 Hz (stateful across blocks):
phase-continuous mixing at the tracked carrier -> 300-Hz low-pass -> instantaneous frequency deviation (limited to +/-400 Hz)
-> anti-alias + decimation -> 0.5-40 Hz band. Also emits 15-40 Hz / 60-150 Hz RMS diagnostics."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import os as _os
from scipy.signal import bessel, butter, lfilter, lfilter_zi, medfilt

AA_TYPE = _os.environ.get("DEMOD_AA", "butter8")
ELP_TYPE = _os.environ.get("DEMOD_ELP", "butter4")

def _design_lp(kind: str, fc: float, fs: float):
    fam, order = (bessel if kind.startswith("bessel") else butter), int(kind[-1])
    return fam(order, fc / (fs / 2), btype="low", norm="phase") if fam is bessel else fam(order, fc / (fs / 2), btype="low")

FS_OUT = 400
DECIM = 120

@dataclass
class QuadDemod:
    fs: int
    lp_hz: float = 300.0
    hp_hz: float = 0.5
    ecg_lp_hz: float = 40.0
    dev_limit_hz: float = 400.0
    polarity: float = 1.0
    _phi: float = 0.0
    _last_phase: float | None = None
    _decim_phase: int = 0
    _med_tail: np.ndarray = field(default_factory=lambda: np.zeros(0))
    _filters: dict = field(default_factory=dict)

    def _prepare(self):
        assert self.fs % FS_OUT == 0, "sample rate must be an integer multiple of 400 Hz"
        b, a = butter(4, self.lp_hz / (self.fs / 2), btype="low")
        self._filters["lp"] = (b, a, lfilter_zi(b, a) * 0.0, lfilter_zi(b, a) * 0.0)
        ba, aa = _design_lp(AA_TYPE, 150.0, self.fs)
        self._filters["aa"] = (ba, aa, lfilter_zi(ba, aa) * 0.0)
        b2, a2 = butter(2, self.hp_hz / (FS_OUT / 2), btype="high")
        self._filters["hp"] = (b2, a2, lfilter_zi(b2, a2) * 0.0)
        b3, a3 = _design_lp(ELP_TYPE, self.ecg_lp_hz, FS_OUT)
        self._filters["elp"] = (b3, a3, lfilter_zi(b3, a3) * 0.0)

        bn, an = butter(2, [60.0 / (FS_OUT / 2), 150.0 / (FS_OUT / 2)], btype="band"); self._filters["nz"] = (bn, an, lfilter_zi(bn, an) * 0.0)
        bs, as_ = butter(2, [15.0 / (FS_OUT / 2), 40.0 / (FS_OUT / 2)], btype="band"); self._filters["sg"] = (bs, as_, lfilter_zi(bs, as_) * 0.0)

    def step(self, x: np.ndarray, t0: float, fc_of_t):
        """`fc_of_t(t_array) -> f_c`; returns (ecg_like @ 400 Hz, timestamps, quality dict).

        quality: `q_t` (timestamps every 0.1 s), `sig_rms` (15-40 Hz RMS), `noise_rms` (60-150 Hz RMS)."""
        if not self._filters:
            self._prepare()
        n = len(x)
        t = t0 + np.arange(n) / self.fs
        fc = fc_of_t(t)

        dphi = 2 * np.pi * fc / self.fs
        phi = self._phi + np.cumsum(dphi)
        self._phi = float(phi[-1] % (2 * np.pi))
        z = x * np.exp(-1j * phi)

        b, a, zr, zi = self._filters["lp"]
        re, zr = lfilter(b, a, z.real, zi=zr); im, zi = lfilter(b, a, z.imag, zi=zi)
        self._filters["lp"] = (b, a, zr, zi)

        ph = np.unwrap(np.angle(re + 1j * im))
        if self._last_phase is not None:
            ph += (self._last_phase - ph[0]) - np.round((self._last_phase - ph[0]) / (2 * np.pi)) * 2 * np.pi
            dev = np.diff(np.concatenate([[self._last_phase], ph]))
        else:
            dev = np.diff(np.concatenate([[ph[0]], ph]))
        self._last_phase = float(ph[-1])
        df = dev * self.fs / (2 * np.pi)
        np.clip(df, -self.dev_limit_hz, self.dev_limit_hz, out=df)

        h, one, zaa = self._filters["aa"]
        y, zaa = lfilter(h, one, df, zi=zaa); self._filters["aa"] = (h, one, zaa)
        idx = np.arange(self._decim_phase, n, DECIM)
        ecg = y[idx] * self.polarity

        med_in = np.concatenate([self._med_tail, ecg]); med = medfilt(med_in, kernel_size=5)
        ecg = med[len(self._med_tail):] if len(self._med_tail) else med
        self._med_tail = med_in[-2:].copy()
        t_out = t[idx]
        self._decim_phase = int((idx[-1] + DECIM - n) if len(idx) else (self._decim_phase - n) % DECIM)

        bn, an, zn = self._filters["nz"]; nz, zn = lfilter(bn, an, ecg, zi=zn); self._filters["nz"] = (bn, an, zn)
        bs, as_, zs = self._filters["sg"]; sg, zs = lfilter(bs, as_, ecg, zi=zs); self._filters["sg"] = (bs, as_, zs)
        hop = FS_OUT // 10; nq = len(ecg) // hop
        q_t = t_out[:nq * hop:hop] if nq else np.zeros(0)
        noise_rms = np.sqrt(np.mean(nz[:nq * hop].reshape(nq, hop) ** 2, axis=1)) if nq else np.zeros(0)
        sig_rms = np.sqrt(np.mean(sg[:nq * hop].reshape(nq, hop) ** 2, axis=1)) if nq else np.zeros(0)

        b2, a2, z2 = self._filters["hp"]; ecg, z2 = lfilter(b2, a2, ecg, zi=z2); self._filters["hp"] = (b2, a2, z2)
        b3, a3, z3 = self._filters["elp"]; ecg, z3 = lfilter(b3, a3, ecg, zi=z3); self._filters["elp"] = (b3, a3, z3)
        return ecg, t_out, {"q_t": q_t, "sig_rms": sig_rms, "noise_rms": noise_rms}
