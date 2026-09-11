#!/usr/bin/env python
"""Command-line entry point: WAV -> R-peak table and 30-s heart-rate / HRV summary.

Usage::
    python -m kardia_ultrasound_ecg.analyze recording.wav --out results/
    kardia-ecg recording.wav --out results/          # after `pip install .`

Outputs (in --out; <name> is set by --name, default "v050"):
    <stem>_<name>_rpeaks.csv   one row per emitted beat: time (s), instantaneous HR (bpm, 0 if confidence < 0.6), confidence and diagnostics
    <stem>_<name>_30s.csv      per 30-s epoch: n beats, mean HR (bpm), RMSSD (ms), SDNN (ms); epochs with fewer than 3 confident beats or HR < 40 bpm are left empty
    <stem>_<name>_summary.json run summary (duration, beats, carrier lock fraction, median carrier-to-noise ratio, comb lines detected)
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .pipeline import HR_CONF, run_night

EPOCH_S = 30.0


def summarise_30s(rpeaks_csv: Path, duration_s: float, out_csv: Path) -> None:
    rows = list(csv.DictReader(open(rpeaks_csv, newline="")))
    t = np.array([float(r["R Peak Times"]) for r in rows]); conf = np.array([float(r["confidence"]) for r in rows])
    n_ep = int(np.ceil(duration_s / EPOCH_S)) if duration_s else 0
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["epoch", "start_s", "n_beats", "n_confident", "mean_hr_bpm", "rmssd_ms", "sdnn_ms"])
        for k in range(n_ep):
            s0 = k * EPOCH_S; m = (t >= s0) & (t < s0 + EPOCH_S); mc = m & (conf >= HR_CONF)
            tc = t[mc]; rr = np.diff(tc) * 1000.0
            rr = rr[(rr >= 300) & (rr <= 2000)]
            hr = 60000.0 / rr.mean() if len(rr) >= 2 else None
            ok = hr is not None and hr >= 40 and len(tc) >= 3
            w.writerow([k, f"{s0:.0f}", int(m.sum()), int(mc.sum()), f"{hr:.2f}" if ok else "", f"{np.sqrt(np.mean(np.diff(rr)**2)):.2f}" if ok and len(rr) >= 3 else "", f"{rr.std(ddof=1):.2f}" if ok and len(rr) >= 3 else ""])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("wav", type=Path); ap.add_argument("--out", type=Path, default=Path("."))
    ap.add_argument("--name", default="v050", help="variant tag used in the output file names")
    ap.add_argument("--polarity", type=float, default=1.0, help="+1 or -1: sign of the ECG-like waveform (does not affect detection)")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    summary = run_night(a.wav, a.name, a.out, dev_dir=None, polarity=a.polarity, save_stages=False)
    rp = a.out / f"{a.wav.stem}_{a.name}_rpeaks.csv"
    summarise_30s(rp, float(summary.get("duration_s") or 0.0), a.out / f"{a.wav.stem}_{a.name}_30s.csv")
    (a.out / f"{a.wav.stem}_{a.name}_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"{a.wav.name}: {summary['n_beats']} beats, lock {summary['locked_frac']:.2f}, carrier SNR median {summary['snr_median']:.0f} dB -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
