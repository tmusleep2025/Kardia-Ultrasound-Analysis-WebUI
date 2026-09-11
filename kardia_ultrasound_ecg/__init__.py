"""Kardia ultrasound single-lead ECG: continuous-state demodulation and QRS detection (version 0.5.0).

Recovers the ECG-like waveform and R-peak times from a smartphone recording (48 kHz WAV) of the KardiaMobile
18.5-19.3 kHz frequency-modulated ultrasonic carrier. See README.md for the algorithm and validation summary.
"""
from .pipeline import run_night  # noqa: F401

__version__ = "0.5.0"
