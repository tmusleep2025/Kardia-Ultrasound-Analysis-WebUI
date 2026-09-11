"""Whole-night pipeline: comb cancellation -> carrier tracking -> quadrature demodulation -> Pan-Tompkins -> per-beat
confidence and emission gate -> R-peak table (CSV). Environment variables PT_* / CT_* / DEMOD_* override the frozen
parameters (defaults = version 0.5.0)."""
from __future__ import annotations

import csv
import os
import json
from pathlib import Path

import numpy as np

from .carrier import CarrierTracker
from .comb import CombCanceller
from .demod import FS_OUT, QuadDemod
from .pt import PanTompkins
from .stream import WavStream

BLOCK_S = 4.0
HR_CONF = 0.6
EMIT_TPL = float(os.environ.get("PT_EMIT_TPL", "0.4"))
EMIT_LR = float(os.environ.get("PT_EMIT_LR", "10.0"))
EMIT_SNR = float(os.environ.get("PT_EMIT_SNR", "0"))
EMIT_SNR_REL = float(os.environ.get("PT_EMIT_SNR_REL", "0"))
EMIT_PROM = float(os.environ.get("PT_EMIT_PROM", "25.0"))
EMIT_Q = float(os.environ.get("PT_EMIT_Q", "0"))
SPARSE_WIN_S = 10.0; SPARSE_MIN = int(os.environ.get("PT_SPARSE_MIN", "4"))
Q_GATE = 0.0
GATE_WIN_S = 15.0
GATE_LR_MIN = 3.0

TPL_HALF_S = 0.100

def _conf_components(snr_db, thr_ratio, rr, rr_ref, tpl_corr):
    c_snr = float(np.clip((snr_db - 20.0) / 30.0, 0, 1))
    c_thr = float(np.clip((thr_ratio - 1.0) / 2.0, 0, 1))
    if rr is None or rr_ref is None:
        c_rr = 0.5
    else:
        c_rr = 1.0 if 0.85 * rr_ref <= rr <= 1.2 * rr_ref else (0.5 if 0.5 * rr_ref <= rr <= 1.6 * rr_ref else 0.0)
    c_tpl = float(np.clip(tpl_corr, 0, 1)) if tpl_corr is not None else 0.5
    return 0.3 * c_snr + 0.2 * c_thr + 0.2 * c_rr + 0.3 * c_tpl, (c_snr, c_thr, c_rr, c_tpl)

def run_night(wav: str | Path, name: str, out_dir: str | Path, dev_dir: str | Path | None = None,
              polarity: float = 1.0, save_stages: bool = False) -> dict:
    wav = Path(wav); out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    st = WavStream(wav)
    tracker = CarrierTracker(st.fs)
    comb = CombCanceller(st.fs)
    demod = QuadDemod(st.fs, polarity=polarity)
    pt = PanTompkins()
    est_t, est_f, est_snr, est_lock, est_prom = [], [], [], [], []
    q_t, q_sig, q_noise = [], [], []
    ecg_chunks, t_chunks = [], []
    stages: dict | None = {} if save_stages else None
    for blk in st.blocks(BLOCK_S):
        x_blk = comb.step(blk.x, blk.t0, tracker.f_smooth if tracker.locked else None)
        ests = tracker.step(x_blk, blk.t0)
        for e in ests:
            est_t.append(e.t); est_f.append(e.f_smooth); est_snr.append(e.snr_db); est_lock.append(e.locked); est_prom.append(e.prom_db)
        if len(est_t) >= 2:
            fc_of_t = lambda t, T=np.array(est_t), F=np.array(est_f): np.interp(t, T, F)
        else:
            f0 = est_f[-1] if est_f else tracker.f_smooth or 18950.0
            fc_of_t = lambda t, f0=f0: np.full_like(t, f0)
        ecg, t_out, qd = demod.step(x_blk, blk.t0, fc_of_t)
        q_t.append(qd["q_t"]); q_sig.append(qd["sig_rms"]); q_noise.append(qd["noise_rms"])
        locked_now = bool(np.mean(est_lock[-max(1, int(BLOCK_S / tracker.hop_s)):]) >= 0.5) if est_lock else False
        q = qd["sig_rms"] / (qd["noise_rms"] + 1e-9)
        allow = np.repeat(q >= Q_GATE, FS_OUT // 10)[: len(ecg)] if len(q) else np.ones(len(ecg), dtype=bool)
        if len(allow) < len(ecg):
            allow = np.concatenate([allow, np.ones(len(ecg) - len(allow), dtype=bool)])
        pt.step(ecg, float(t_out[0]) if len(t_out) else blk.t0, locked_now, stages, allow=allow)
        if save_stages or True:
            ecg_chunks.append(ecg.astype(np.float32)); t_chunks.append(t_out[0] if len(t_out) else blk.t0)
    ecg_all = np.concatenate(ecg_chunks) if ecg_chunks else np.zeros(0, dtype=np.float32)
    t_ecg0 = float(t_chunks[0]) if t_chunks else 0.0
    T = np.array(est_t); SNR = np.array(est_snr); PROM = np.array(est_prom)
    QT = np.concatenate(q_t) if q_t else np.zeros(0); QR = (np.concatenate(q_sig) / (np.concatenate(q_noise) + 1e-9)) if q_t else np.zeros(0)

    beats = pt.beats
    import os as _os
    if len(beats) >= 3 and _os.environ.get("PT_CLEANUP", "0") == "1":
        keep = np.ones(len(beats), dtype=bool)
        bt = np.array([b.t for b in beats]); amp = np.array([b.mwi_peak for b in beats])
        for i in range(1, len(beats) - 1):
            if not keep[i - 1]:
                continue
            ref = beats[i].rr_avg
            if ref is None:
                continue
            rr_a, rr_b = bt[i] - bt[i - 1], bt[i + 1] - bt[i]
            if rr_a < 0.6 * ref and 0.8 * ref <= rr_a + rr_b <= 1.2 * ref:
                j = i if amp[i] < amp[i - 1] else i - 1
                keep[j] = False
        beats = [b for b, k in zip(beats, keep) if k]
        for k in range(1, len(beats)):
            beats[k].rr = beats[k].t - beats[k - 1].t
        pt.beats = beats

    beats = pt.beats
    half = int(TPL_HALF_S * FS_OUT)
    segs = []
    for b in beats:
        i = int(round((b.t - t_ecg0) * FS_OUT))
        seg = ecg_all[i - half:i + half] if i - half >= 0 and i + half <= len(ecg_all) else None
        segs.append((seg - seg.mean()) if (seg is not None and len(seg) == 2 * half) else None)

    tpl = None; n_tpl = 0; shift = int(0.012 * FS_OUT)
    rows = []
    for b, s in zip(beats, segs):
        snr_b = float(np.interp(b.t, T, SNR)) if len(T) else 0.0
        prom_b = float(np.interp(b.t, T, PROM)) if len(T) else 0.0
        if len(QT):
            i0, i1 = np.searchsorted(QT, b.t - 0.5), np.searchsorted(QT, b.t + 0.5, side="right")
            q_b = float(np.median(QR[i0:i1])) if i1 > i0 else 0.0
        else:
            q_b = 0.0
        corr = None
        if s is not None and tpl is not None:
            best = -1.0
            for k in range(-shift, shift + 1):
                x = s[shift + k: len(s) - shift + k]; y = tpl[shift: len(tpl) - shift]
                den = np.linalg.norm(x) * np.linalg.norm(y)
                best = max(best, float(np.dot(x, y) / den) if den > 0 else 0.0)
            corr = best
        if s is not None and b.level_ratio > 10:
            if tpl is None:
                tpl = s.copy()
            else:
                a_ = 0.2 if n_tpl < 20 else 0.05
                tpl = (1 - a_) * tpl + a_ * s
            n_tpl += 1
        conf, comp = _conf_components(snr_b, b.mwi_peak / b.thr_i1 if b.thr_i1 > 0 else 0.0, b.rr, b.rr_avg, corr)
        inst = 60.0 / b.rr if (b.rr is not None and 0.3 <= b.rr <= 2.0 and conf >= HR_CONF) else 0.0
        rows.append((b.t, b.mwi_peak, inst, 0.0, round(conf, 3), round(snr_b, 1), int(b.searchback), *[round(c, 2) for c in comp], round(b.level_ratio, 2), round(b.rr_cv, 3), round(prom_b, 1), round(q_b, 2)))

    if rows and EMIT_TPL > 0:
        snr_floor = max(EMIT_SNR, (float(np.median([r[5] for r in rows])) - EMIT_SNR_REL) if EMIT_SNR_REL > 0 else -1e9)
        rows = [r for r in rows if (r[10] >= EMIT_TPL or r[11] >= EMIT_LR) and r[5] >= snr_floor and r[13] >= EMIT_PROM and r[14] >= EMIT_Q]

    if rows and SPARSE_MIN > 0:
        bt_ = np.array([r[0] for r in rows]); lo_ = np.searchsorted(bt_, bt_ - SPARSE_WIN_S / 2); hi_ = np.searchsorted(bt_, bt_ + SPARSE_WIN_S / 2, side="right")
        rows = [r for r, n_ in zip(rows, hi_ - lo_) if n_ >= SPARSE_MIN]

    if rows:
        bt = np.array([r[0] for r in rows]); lrat = np.array([r[11] for r in rows]); rr_all = np.diff(bt, prepend=np.nan)
        lo = np.searchsorted(bt, bt - GATE_WIN_S); hi = np.searchsorted(bt, bt + GATE_WIN_S, side="right")
        gate = np.zeros(len(rows), dtype=bool)
        for i in range(len(rows)):
            if np.isnan(rr_all[i]):
                continue
            w_rr = rr_all[lo[i] + 1:hi[i]]; w_rr = w_rr[(w_rr >= 0.3) & (w_rr <= 2.0)]
            if len(w_rr) < 5:
                continue
            m = float(np.median(w_rr)); lr_med = float(np.median(lrat[lo[i]:hi[i]]))
            gate[i] = (0.6 * m <= rr_all[i] <= 1.4 * m) and (lr_med > GATE_LR_MIN)
        rows = [(r[0], r[1], r[2] if gate[i] else 0.0, *r[3:], int(gate[i])) for i, r in enumerate(rows)]
    stem = wav.stem
    p_csv = out_dir / f"{stem}_{name}_rpeaks.csv"
    with open(p_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["R Peak Times", "R Peak Features", "Instant HR (BPM)", "Avg HR (30s) (BPM)", "confidence", "snr_db", "searchback", "c_snr", "c_thr", "c_rr", "c_tpl", "level_ratio", "rr_cv", "prom_db", "q_ratio", "gate"])
        w.writerows(rows)
    summary = {"stem": stem, "name": name, "n_beats": len(rows), "n_valid": int(sum(r[2] > 0 for r in rows)),
               "n_searchback": int(sum(r[6] for r in rows)), "n_dropout": int(pt.n_dropout), "locked_frac": float(np.mean(est_lock)) if est_lock else 0.0,
               "snr_median": float(np.median(SNR)) if len(SNR) else None, "duration_s": st.duration_s,
               "fc_median": float(np.median(est_f)) if est_f else None,
               "comb_lines": len(comb.lines), "comb_active": bool(comb.active)}
    if dev_dir is not None:
        dev_dir = Path(dev_dir); dev_dir.mkdir(parents=True, exist_ok=True)
        arrays = {"ecg": ecg_all, "t0": np.array([t_ecg0]), "fs": np.array([FS_OUT]),
                  "q_t": np.concatenate(q_t).astype(np.float32), "q_sig": np.concatenate(q_sig).astype(np.float32), "q_noise": np.concatenate(q_noise).astype(np.float32),
                  "est_t": T.astype(np.float32), "est_f": np.array(est_f, dtype=np.float32), "est_snr": SNR.astype(np.float32), "est_prom": PROM.astype(np.float32),
                  "est_lock": np.array(est_lock, dtype=np.int8)}
        if stages:
            for k, v in stages.items():
                arrays[k] = np.concatenate(v)
            arrays["rej_t"] = np.array([r[0] for r in pt.rejects], dtype=np.float64); arrays["rej_code"] = np.array([r[1] for r in pt.rejects], dtype=np.int8)
        np.savez_compressed(dev_dir / f"{stem}_{name}.npz", **arrays)
        (dev_dir / f"{stem}_{name}.json").write_text(json.dumps(summary, indent=1))
    return summary
