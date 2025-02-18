# signal_processing.py
import numpy as np
import pywt
import dtcwt
from scipy.signal import hilbert, resample

def wavelet_denoise(signal, wavelet='db4', level=9, remove_levels=None):
    """
    使用小波分解與重建對訊號進行降噪
    """
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    if remove_levels is None:
        remove_levels = [0, 9]  # 預設去除的層級
    for i in remove_levels:
        if i < len(coeffs):
            coeffs[i] = np.zeros_like(coeffs[i])
    return pywt.waverec(coeffs, wavelet)

def dtcwt_denoise(signal, nlevels=11, remove_levels=None):
    """
    使用 DTCWT 對訊號進行降噪
    """
    if len(signal) % 2 != 0:
        signal = np.pad(signal, (0, 1), mode='edge')
    if remove_levels is None:
        remove_levels = [0, 9, 10]
    transform = dtcwt.Transform1d()
    result = transform.forward(signal, nlevels=nlevels)
    highpasses = list(result.highpasses)
    for i in remove_levels:
        if i < len(highpasses):
            highpasses[i] = np.zeros_like(highpasses[i])
    pyramid = dtcwt.Pyramid(lowpass=result.lowpass, highpasses=highpasses)
    denoised_signal = transform.inverse(pyramid)
    return denoised_signal

def frequency_tracking(signal, fs):
    """
    利用 Hilbert transform 估算訊號瞬時頻率
    """
    analytic_signal = hilbert(signal)
    filtered_phase = np.unwrap(np.angle(analytic_signal))
    filtered_frequency = np.gradient(filtered_phase) * (fs / (2.0 * np.pi))
    return filtered_frequency

def resample_xHz(original_signal, original_fs, target_fs=200):
    """
    將原始訊號重採樣至 target_fs (Hz)
    """
    resampled_signal = resample(original_signal, int(round(len(original_signal) * target_fs / original_fs)))
    return resampled_signal, target_fs

def double_slope_preprocessing(signal, fs):
    """
    對輸入訊號進行「雙斜率」預處理
    """
    n = len(signal)
    processed = np.zeros(n)
    deriv = np.gradient(signal) * fs
    offset_start = int(np.floor(0.015 * fs))
    offset_end = int(np.floor(0.040 * fs))
    for i in range(offset_end, n - offset_end):
        left_window = deriv[i - offset_end : i - offset_start + 1]
        right_window = deriv[i + offset_start : i + offset_end + 1]
        if left_window.size == 0 or right_window.size == 0:
            continue
        left_max = np.max(left_window)
        left_min = np.min(left_window)
        right_max = np.max(right_window)
        right_min = np.min(right_window)
        candidate1 = left_max - right_min
        candidate2 = right_max - left_min
        processed[i] = max(candidate1, candidate2)
    return processed

def moving_window_integration(signal, window_size=17):
    """
    進行滑動窗口積分，回傳與原訊號長度相同的積分結果
    """
    window = np.ones(window_size)
    integrated_signal = np.convolve(signal, window, mode='same')
    return integrated_signal
