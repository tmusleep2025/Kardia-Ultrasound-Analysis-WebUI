# Kardia ultrasound single-lead ECG — continuous-state analysis pipeline (v0.5.0)

Recovers an ECG-like waveform and R-peak times from a **smartphone audio recording** of the AliveCor KardiaMobile
ultrasonic carrier (18.5–19.3 kHz, frequency-modulated by the body-surface voltage), and reports 30-s heart rate (HR)
and time-domain heart-rate variability (HRV). It replaces the 30-s segment-wise algorithm released as v0.3.0
([Zenodo 10.5281/zenodo.14886145](https://doi.org/10.5281/zenodo.14886145)).

The pipeline was validated against polysomnography (PSG) ECG in 58 adults during overnight sleep (development set
21 participants, hold-out set 37; iPhone and Android recordings). Summary of the hold-out (iPhone) results:
30-s HR bias 0.20 bpm, SD of difference 2.64 bpm, 94.6% of epochs within ±5 bpm, beat-level sensitivity 0.85 and PPV 0.94
(150-ms window, ANSI/AAMI EC57). Absolute 30-s HRV is overestimated (timestamp jitter) and should be read as a trend.

> Research software. Not a medical device; no diagnostic claims.

## Install

```bash
pip install git+https://github.com/tmusleep2025/Kardia-Ultrasound-Analysis-WebUI@v0.5.0
# or from a local checkout (Python ≥ 3.10; numpy, scipy):
pip install .
# or without installing:
pip install -r requirements.txt
```

Reference environment: the validation outputs were produced with Python 3.12.3, numpy 1.26.4 and scipy 1.15.2. Use these
versions to reproduce them bit for bit; newer versions run the same algorithm but are not guaranteed to be bit-identical.

## Use

```bash
kardia-ecg recording.wav --out results/
# equivalent
python -m kardia_ultrasound_ecg.analyze recording.wav --out results/
```

Input: mono WAV, 48 kHz (the phone's microphone recording; any sample rate that is a multiple of 400 Hz works).
Outputs in `--out`:

| file | content |
|---|---|
| `<stem>_v050_rpeaks.csv` | one row per emitted beat: time (s), instantaneous HR (bpm; 0 when confidence < 0.6), confidence and diagnostics (carrier SNR, in-band prominence, template correlation, level ratio) |
| `<stem>_v050_30s.csv` | per 30-s epoch: beats, confident beats, mean HR (bpm), RMSSD (ms), SDNN (ms); empty when fewer than 3 confident beats or HR < 40 bpm |
| `<stem>_v050_summary.json` | duration, beat counts, carrier lock fraction, median carrier-to-noise ratio, comb lines detected |

Python API:

```python
from kardia_ultrasound_ecg import run_night
summary = run_night("recording.wav", "v050", "results/")
```

## Algorithm (whole night as one stream; every state persists across 4-s blocks)

1. **Comb-interference cancellation** — equally spaced narrow lines (120.17-Hz spacing, mains-harmonic comb coupled into
   some recordings) are detected and removed with cascaded 3-Hz notch filters, retained for the night.
2. **Carrier presence and tracking** — every 0.25 s the 18.5–19.3 kHz band is analysed; the carrier is deemed present from
   *device-independent* features: in-band prominence of the 1-s averaged spectrum (≥ 18 dB to acquire, ≥ 15 dB to keep) and
   frequency continuity (global peak within ±150 Hz for 1 s). While locked the frequency is smoothed (EMA, 2 s) and searched
   within ±120 Hz; when lock is lost, detection is suspended.
3. **Quadrature FM demodulation** — phase-continuous mixing at the tracked carrier → 300-Hz low-pass → instantaneous
   frequency deviation (limited to ±400 Hz) → decimation to 400 Hz → 0.5–40 Hz. The result is an ECG-like waveform.
4. **Pan–Tompkins QRS detection** (Pan & Tompkins, 1985) on the demodulated waveform: 15–40-Hz band-pass, squaring, 50-ms
   moving-window integration, adaptive SPKI/NPKI thresholds, search-back, 3-s dropout re-learning, and an
   amplitude-and-rhythm contest for candidates closer than 0.45 s.
5. **Per-beat confidence and emission gate** — carrier-to-noise ratio, threshold ratio, RR consistency and correlation with a
   continuously adapted QRS template; a beat is emitted only if template correlation ≥ 0.4 or level ratio ≥ 10, the in-band
   prominence at that moment is ≥ 25 dB, and at least 4 beats fall within ±5 s. 30-s HR uses beats with confidence ≥ 0.6.

Parameters can be overridden with environment variables (`PT_*`, `CT_*`, `DEMOD_*`; see the module constants). The defaults
are the frozen v0.5.0 values.

## Citation

If you use this software, please cite the validation paper (in preparation) and the software itself:
Li C-Y, Tani J. *Kardia Ultrasound Analysis WebUI*, version 0.5.0 (2026). Zenodo. https://doi.org/10.5281/zenodo.14886144
— the concept DOI, which always resolves to the latest version. Machine-readable metadata is in `CITATION.cff`.

## License

MIT (see `LICENSE`). The original WebUI project (v0.3.0) is © 2025 Ching-Yu Li and Jowy Tani, MIT.

---

## Legacy WebUI (v0.3.0)

The `Flask/` directory keeps the v0.3.0 web application (30-s segment-wise algorithm) unchanged for existing users; it does
**not** use the v0.5.0 pipeline above. A ready-to-run Windows build (`app_v0.3.0.exe`) is attached to the
[v0.3.0 release](https://github.com/tmusleep2025/Kardia-Ultrasound-Analysis-WebUI/releases/tag/v0.3.0), and the full WebUI
instructions are in the [v0.3.0 README](https://github.com/tmusleep2025/Kardia-Ultrasound-Analysis-WebUI/tree/v0.3.0#readme)
([中文](README-CN.md)). v0.3.0 was developed for the study *Feasibility and Validation of a Cost-Effective Continuous Remote
Cardiac Monitoring in Clinical Practice and Home-based Application* by Dr. Jowy Tani (Sleep Center, Taipei Medical University
Wanfang Hospital) and is archived at [Zenodo 10.5281/zenodo.14886145](https://doi.org/10.5281/zenodo.14886145).

A 10-minute test recording is provided at [`Flask/audio_files/example_audio_10min.wav`](Flask/audio_files/example_audio_10min.wav);
it can also be analysed with the v0.5.0 command line:

```bash
kardia-ecg Flask/audio_files/example_audio_10min.wav --out results/
```

## Contact

If you have any questions or suggestions, please contact us or open a
[GitHub issue](https://github.com/tmusleep2025/Kardia-Ultrasound-Analysis-WebUI/issues):

- [jowytani@tmu.edu.tw](mailto:jowytani@tmu.edu.tw)
- [tmusleep2025@gmail.com](mailto:tmusleep2025@gmail.com)
