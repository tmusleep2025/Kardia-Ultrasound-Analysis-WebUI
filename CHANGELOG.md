# Changelog

## 0.5.0 — 2026-09-09

First release of the continuous-state pipeline. Compared with v0.3.0 (30-s segment-wise processing, Hilbert instantaneous
frequency, wavelet denoising, dual fixed-floor thresholds):

- **Whole-night streaming**: filters, carrier tracker, detector thresholds, RR history and QRS template persist across blocks;
  the 30-s segment boundaries no longer exist (the boundary miss-rate excess of the segment-wise design disappears).
- **Carrier presence decided from device-independent features** (in-band prominence + frequency continuity) instead of an
  out-of-band carrier-to-noise ratio; pure noise is no longer processed into beats and the same thresholds work on iPhone and Android.
- **Explicit ECG-like waveform** by quadrature FM demodulation before detection.
- **Pan–Tompkins detector** (15–40 Hz, 50-ms integration) with search-back, dropout re-learning and an amplitude-and-rhythm
  candidate contest.
- **Per-beat confidence and emission gate** (template correlation / level ratio, in-band prominence ≥ 25 dB, sparse-beat rule).
- **Comb-interference cancellation** for 120.17-Hz-spaced mains-harmonic lines.
- New CLI `kardia-ecg` producing R-peak, 30-s HR/HRV and summary files.

## 0.3.0 — 2025

Segment-wise WebUI algorithm (Zenodo 10.5281/zenodo.14886145).
