"""Pan-Tompkins QRS detector (Pan & Tompkins, 1985) adapted to the demodulated Kardia waveform: 15-40 Hz band-pass,
squaring, 50-ms moving-window integration, adaptive SPKI/NPKI thresholds, search-back, 3-s dropout re-learning and an
amplitude-and-rhythm contest for candidates closer than 0.45 s. All state persists across blocks."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi

FS = 400
BP_BAND = (15.0, 40.0)
MWI_S = 0.050
REFRACTORY_S = 0.200
import os as _os
MIN_RR_S = float(_os.environ.get("PT_MIN_RR_S", "0.450"))
TWAVE_S = 0.360
LEARN_S = 2.0
RR_MISSED_FACTOR = 1.66
THR_FRAC = float(_os.environ.get("PT_THR_FRAC", "0.25"))
SHORT_RR = float(_os.environ.get("PT_SHORT_RR", "0"))
DROPOUT_S = 3.0
BP_SEARCH_S = 0.150

@dataclass
class Beat:
    t: float
    t_mwi: float
    mwi_peak: float
    bp_peak: float
    thr_i1: float
    slope: float
    searchback: bool
    rr: float | None
    rr_avg: float | None = None
    level_ratio: float = 0.0
    rr_cv: float = 0.0

@dataclass
class PanTompkins:
    fs: int = FS
    reacq_s: float = 3.0
    _f: dict = field(default_factory=dict)

    spki: float = 0.0; npki: float = 0.0; spkf: float = 0.0; npkf: float = 0.0
    thr_i1: float = 0.0; thr_i2: float = 0.0; thr_f1: float = 0.0; thr_f2: float = 0.0
    learned: bool = False
    n_dropout: int = 0
    rejects: list = field(default_factory=list)
    _learn_buf: list = field(default_factory=list)

    rr1: deque = field(default_factory=lambda: deque(maxlen=8))
    rr2: deque = field(default_factory=lambda: deque(maxlen=8))
    last_qrs_t: float | None = None
    last_qrs_slope: float = 0.0
    beats: list = field(default_factory=list)

    _cands: list = field(default_factory=list)

    _tail_mwi: np.ndarray = field(default_factory=lambda: np.zeros(0))
    _tail_bp: np.ndarray = field(default_factory=lambda: np.zeros(0))
    _tail_slope: np.ndarray = field(default_factory=lambda: np.zeros(0))
    _tail_t0: float = 0.0
    _unlock_since: float | None = None
    _n_total: int = 0

    def _prepare(self):
        b, a = butter(2, [BP_BAND[0] / (self.fs / 2), BP_BAND[1] / (self.fs / 2)], btype="band")
        self._f["bp"] = (b, a, lfilter_zi(b, a) * 0.0)

        self._f["d"] = (np.array([1.0]), np.array([1.0]), np.zeros(0))
        n = int(round(MWI_S * self.fs))
        self._f["mwi"] = (np.ones(n) / n, np.array([1.0]), np.zeros(n - 1))

    def _run(self, key, x):
        b, a, z = self._f[key]
        y, z = lfilter(b, a, x, zi=z); self._f[key] = (b, a, z)
        return y

    def _update_thr(self):
        self.thr_i1 = self.npki + THR_FRAC * (self.spki - self.npki); self.thr_i2 = 0.5 * self.thr_i1
        self.thr_f1 = self.npkf + THR_FRAC * (self.spkf - self.npkf); self.thr_f2 = 0.5 * self.thr_f1

    def _learn(self, mwi, bp):

        self.spki = float(np.percentile(mwi, 99)); self.npki = float(np.median(mwi))
        self.spkf = float(np.percentile(np.abs(bp), 99)); self.npkf = float(np.median(np.abs(bp)))
        self._update_thr(); self.learned = True

    def _rr_limits(self):
        if len(self.rr2) >= 2:
            avg = float(np.mean(self.rr2))
        elif len(self.rr1) >= 2:
            avg = float(np.mean(self.rr1))
        else:
            return None
        return 0.92 * avg, 1.16 * avg, RR_MISSED_FACTOR * avg, avg

    def _accept(self, t_r, t_mwi, peak, bp_peak, slope, searchback):
        rr = None if self.last_qrs_t is None else t_r - self.last_qrs_t
        lim = self._rr_limits()
        if rr is not None:
            self.rr1.append(rr)
            if lim is None or lim[0] <= rr <= lim[1]:
                self.rr2.append(rr)
            elif not searchback and rr > lim[1]:

                self.thr_i1 *= 0.5; self.thr_i2 *= 0.5; self.thr_f1 *= 0.5; self.thr_f2 *= 0.5
        pk_i = min(peak, 4.0 * self.spki) if self.spki > 0 else peak
        pk_f = min(bp_peak, 4.0 * self.spkf) if self.spkf > 0 else bp_peak
        if searchback:
            self.spki = 0.25 * pk_i + 0.75 * self.spki; self.spkf = 0.25 * pk_f + 0.75 * self.spkf
        else:
            self.spki = 0.125 * pk_i + 0.875 * self.spki; self.spkf = 0.125 * pk_f + 0.875 * self.spkf
        self._update_thr()
        self.last_qrs_t = t_r; self.last_qrs_slope = slope
        lim2 = self._rr_limits()
        lr = self.spki / self.npki if self.npki > 0 else 0.0
        cv = float(np.std(self.rr1) / np.mean(self.rr1)) if len(self.rr1) >= 4 else 0.0
        self.beats.append(Beat(t_r, t_mwi, peak, bp_peak, self.thr_i1, slope, searchback, rr, lim2[3] if lim2 else None, lr, cv))

    def _noise(self, peak, bp_peak):
        peak = min(peak, 4.0 * self.spki) if self.spki > 0 else peak
        bp_peak = min(bp_peak, 4.0 * self.spkf) if self.spkf > 0 else bp_peak
        self.npki = 0.125 * peak + 0.875 * self.npki; self.npkf = 0.125 * bp_peak + 0.875 * self.npkf
        self._update_thr()

    def step(self, ecg: np.ndarray, t0: float, locked: bool, stages: dict | None = None, allow: np.ndarray | None = None) -> None:
        """Feed one block of the ECG-like signal (400 Hz). `locked`: carrier lock state for this block (re-learn after >= reacq_s unlocked).
        `allow`: boolean mask of the same length; where False no detection and no threshold update."""
        if not self._f:
            self._prepare()
        if allow is None:
            allow = np.ones(len(ecg), dtype=bool)
        bp = self._run("bp", ecg)
        d = self._run("d", bp)
        sq = d * d
        mwi = self._run("mwi", sq)
        if stages is not None:
            stages.setdefault("bp", []).append(bp.astype(np.float32)); stages.setdefault("deriv", []).append(d.astype(np.float32))
            stages.setdefault("sq", []).append(sq.astype(np.float32)); stages.setdefault("mwi", []).append(mwi.astype(np.float32))
            stages.setdefault("thr_i1", []).append(np.full(len(mwi), self.thr_i1, dtype=np.float32))
            stages.setdefault("spki", []).append(np.full(len(mwi), self.spki, dtype=np.float32))
            stages.setdefault("npki", []).append(np.full(len(mwi), self.npki, dtype=np.float32))
            stages.setdefault("thr_f1", []).append(np.full(len(mwi), self.thr_f1, dtype=np.float32))
            stages.setdefault("learned", []).append(np.full(len(mwi), int(self.learned and locked), dtype=np.int8))

        if not locked:
            if self._unlock_since is None:
                self._unlock_since = t0
        else:
            if self._unlock_since is not None and (t0 - self._unlock_since) >= self.reacq_s:
                self.learned = False; self._learn_buf = []
            self._unlock_since = None
        if not self.learned:
            self._learn_buf.append((mwi, bp))
            if sum(len(m) for m, _ in self._learn_buf) >= LEARN_S * self.fs:
                self._learn(np.concatenate([m for m, _ in self._learn_buf]), np.concatenate([b for _, b in self._learn_buf]))
                self._learn_buf = []
            if not self.learned:
                self._push_tail(mwi, bp, np.abs(d), t0); return
        if not locked:
            self._push_tail(mwi, bp, np.abs(d), t0); return

        n_tail = len(self._tail_mwi)
        M = np.concatenate([self._tail_mwi, mwi]); B = np.concatenate([self._tail_bp, bp]); S = np.concatenate([self._tail_slope, np.abs(d)])
        A = np.concatenate([np.ones(n_tail, dtype=bool), allow])
        t_base = self._tail_t0 if n_tail else t0

        lo = max(n_tail - 1, 1); hi = len(M) - 1
        seg = M[lo - 1:hi + 1]
        pk = np.where((seg[1:-1] > seg[:-2]) & (seg[1:-1] >= seg[2:]))[0] + lo
        win = int(BP_SEARCH_S * self.fs)
        for i in pk:
            if not A[i]:
                if M[i] > self.thr_i1: self.rejects.append((t_base + i / self.fs, 6))
                continue
            peak = float(M[i]); t_mwi = t_base + i / self.fs
            j0 = max(0, i - win); j = j0 + int(np.argmax(np.abs(B[j0:i + 1]))); t_r = t_base + j / self.fs
            bp_peak = float(abs(B[j])); slope = float(np.max(S[j0:i + 1]))
            self._cands.append((t_mwi, t_r, peak, bp_peak, slope))

            lim = self._rr_limits()
            if lim is not None and self.last_qrs_t is not None and (t_mwi - self.last_qrs_t) > lim[2]:
                cands = [c for c in self._cands if self.last_qrs_t + REFRACTORY_S < c[1] < t_mwi - REFRACTORY_S / 2 and c[2] > self.thr_i2 and c[3] > self.thr_f2]
                if cands:
                    c = max(cands, key=lambda c: c[2])
                    self._accept(c[1], c[0], c[2], c[3], c[4], True)
            is_qrs = peak > self.thr_i1 and bp_peak > self.thr_f1
            if self.last_qrs_t is not None and (t_r - self.last_qrs_t) < REFRACTORY_S and not is_qrs:
                continue

            if is_qrs and MIN_RR_S > 0 and self.last_qrs_t is not None and (t_r - self.last_qrs_t) < MIN_RR_S and self.beats:
                prev = self.beats[-1]; lim = self._rr_limits()

                win_new = peak > prev.mwi_peak
                if lim is not None and prev.rr is not None:
                    rr_new = prev.rr + (t_r - prev.t)
                    win_new = win_new and abs(rr_new - lim[3]) < abs(prev.rr - lim[3])
                if win_new:
                    self.beats.pop(); self.last_qrs_t = self.beats[-1].t if self.beats else None
                    if self.rr1: self.rr1.pop()
                    if self.rr2 and prev.rr is not None and abs(self.rr2[-1] - prev.rr) < 1e-9: self.rr2.pop()
                    self.rejects.append((prev.t, 2))
                else:
                    self.rejects.append((t_r, 2)); continue
            elif peak > self.thr_i1 and not is_qrs:
                self.rejects.append((t_r, 5))
            if is_qrs and self.last_qrs_t is not None:
                rr_now = t_r - self.last_qrs_t
                lim = self._rr_limits()
                if rr_now < TWAVE_S and (slope < 0.5 * self.last_qrs_slope or peak < 0.5 * self.spki):
                    is_qrs = False; self.rejects.append((t_r, 3))
                elif SHORT_RR > 0 and lim is not None and rr_now < SHORT_RR * lim[3] and peak < SHORT_RR * self.spki:
                    is_qrs = False; self.rejects.append((t_r, 4))
            if is_qrs:
                self._accept(t_r, t_mwi, peak, bp_peak, slope, False)
            else:
                self._noise(peak, bp_peak)

        t_now = t_base + (len(M) - 1) / self.fs

        if self.last_qrs_t is not None and (t_now - self.last_qrs_t) > DROPOUT_S:
            self.learned = False; self._learn_buf = []; self.n_dropout += 1
            self.last_qrs_t = None; self._cands = []
        self._cands = [c for c in self._cands if c[0] > t_now - 3.0]
        self._push_tail(mwi, bp, np.abs(d), t0)

    def _push_tail(self, mwi, bp, slope, t0):
        keep = int(BP_SEARCH_S * self.fs) + 2
        M = np.concatenate([self._tail_mwi, mwi]); B = np.concatenate([self._tail_bp, bp]); S = np.concatenate([self._tail_slope, slope])
        t_base = self._tail_t0 if len(self._tail_mwi) else t0
        if len(M) > keep:
            self._tail_t0 = t_base + (len(M) - keep) / self.fs
            self._tail_mwi, self._tail_bp, self._tail_slope = M[-keep:], B[-keep:], S[-keep:]
        else:
            self._tail_t0 = t_base; self._tail_mwi, self._tail_bp, self._tail_slope = M, B, S
