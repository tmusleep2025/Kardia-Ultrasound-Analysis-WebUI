# ECGAnalyzer.py
import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter_zi, firwin, lfilter, butter, filtfilt, medfilt
from tqdm import tqdm
from collections import deque

import signal_processing as sp
import stats_analysis as sa

class ECGAnalyzer:
    """
    處理並分析音訊中 ECG 訊號的類別。
    """
    def __init__(self, file_path):
        self.file_path = file_path
        
        # 初始化資料與 QRS 偵測相關變數
        self.resampled_freq_list = []
        self.time_list = []
        self.r_peak_records = []
        
        # QRS 峰值偵測的暫存變數
        self.r_peak_times = []
        self.r_peak_features = []
        self.resampled_freq_features = []
        self.inst_hr = 0
        self.avg_hr = 0
        self.current_segment_features = []
        self.current_segment_times = []
        self.error_segment_times = []
        self.peak_buffer = []
        self.recent_8_QRS_peaks = deque(maxlen=8)
        self.recent_8_QRS_time = deque(maxlen=8)
        self.above_threshold = False

        # 動態門檻參數
        self.threshold_low_lim = 1250
        self.threshold_high_lim = 2500
        self.threshold_low = self.threshold_low_lim 
        self.threshold_high = self.threshold_high_lim
        self.no_detected = 0
        self.th_state = 0
        self.temp_th_state = 0

        # 滑動窗口 (待在 process_audio 中根據重採樣頻率初始化)
        self.sliding_window = None
        self.sliding_window_STD = 100
        self.sliding_window_VAR = 10000

        # 目前處理到的區塊編號與總區塊數
        self.current_chunk = 0
        self.total_chunks = 0

    def process_audio(self):
        # ---------------------------
        # 1. 讀取與預處理音訊資料
        # ---------------------------
        sample_rate, audio_data = wavfile.read(self.file_path, mmap=True)
        if audio_data.ndim > 1:
            audio_data = audio_data[:, 0]
        info = np.iinfo(audio_data.dtype)
        scale_factor = float(max(abs(info.min), info.max))

        chunk_duration = 30.0  # 每個區塊 30 秒
        chunk_samples = int(sample_rate * chunk_duration)
        extra = int(0.05 * chunk_samples)
        resampled_target_fs = 360 
        max_seconds = 30
        max_samples = max_seconds * resampled_target_fs

        # 根據重採樣頻率初始化滑動窗口 (視窗長度 5 秒)
        self.sliding_window = deque(maxlen=int(resampled_target_fs * 5))

        # ---------------------------
        # 2. 濾波器參數與延遲時間計算
        # ---------------------------
        freq_min = 18500
        freq_max = 19300
        b, a = butter(8, [freq_min / (sample_rate / 2), freq_max / (sample_rate / 2)], btype='band')
        zi = lfilter_zi(b, a)

        taps = firwin(41, [15, 35], pass_zero=False, fs=resampled_target_fs)
        normalized_cutoff = 5 / (0.5 * resampled_target_fs)
        process_DS_b, process_DS_a = butter(2, normalized_cutoff, btype='low', analog=False)

        shift_time_offset_1 = 20 / resampled_target_fs  
        shift_time_offset_2 = 1 / resampled_target_fs   
        shift_total_time = shift_time_offset_1 + shift_time_offset_2

        # ---------------------------
        # 3. 計算區塊索引 (依據音訊長度)
        # ---------------------------
        chunk_starts = list(range(0, len(audio_data) - chunk_samples + 1, chunk_samples))
        self.total_chunks = len(chunk_starts)

        # ---------------------------
        # 處理每個區塊
        # ---------------------------
        for idx, i in enumerate(tqdm(chunk_starts, total=self.total_chunks, desc="Processing...", unit="chunk")):
            # 更新目前處理到的區塊編號 (從 1 開始)
            self.current_chunk = idx + 1

            # 根據區塊位置處理 extra 部分
            if idx == 0:
                start_idx = i
                end_idx = i + chunk_samples + extra
            elif idx == self.total_chunks - 1:
                start_idx = i - extra
                end_idx = i + chunk_samples
            else:
                start_idx = i - extra
                end_idx = i + chunk_samples + extra

            chunk = audio_data[start_idx:end_idx].astype(float) / scale_factor
            
            # (a) 帶通濾波
            Audio_filtered_signal, zi = lfilter(b, a, chunk, zi=zi)
            estimated_freq_full = sp.frequency_tracking(Audio_filtered_signal, sample_rate)

            # 根據區塊位置切除 extra 部分
            if idx == 0:
                estimated_freq = estimated_freq_full[:-extra] if extra > 0 else estimated_freq_full
                Audio_filtered_signal = Audio_filtered_signal[:-extra] if extra > 0 else Audio_filtered_signal
            elif idx == self.total_chunks - 1:
                estimated_freq = estimated_freq_full[extra:] if extra > 0 else estimated_freq_full
                Audio_filtered_signal = Audio_filtered_signal[extra:] if extra > 0 else Audio_filtered_signal
            else:
                estimated_freq = estimated_freq_full[extra:-extra] if extra > 0 else estimated_freq_full
                Audio_filtered_signal = Audio_filtered_signal[extra:-extra] if extra > 0 else Audio_filtered_signal

            # (b) 重採樣
            resampled_freq, resampled_fs = sp.resample_xHz(estimated_freq, sample_rate, target_fs=resampled_target_fs)
            current_time_offset = i / sample_rate
            time_chunk = np.arange(len(resampled_freq)) / resampled_fs + current_time_offset

            # (c) 中值濾波
            resampled_freq = medfilt(resampled_freq, kernel_size=9)
            self.resampled_freq_list.extend(resampled_freq)
            self.time_list.extend(time_chunk)
            
            # (d) DTCWT 去雜訊 與 Wavelet 校正基線
            dtcwt_denoise_results = sp.dtcwt_denoise(resampled_freq)
            wavelet_denoise_results = sp.wavelet_denoise(dtcwt_denoise_results)
            filtered_signal = lfilter(taps, 1.0, wavelet_denoise_results)

            # (e) 雙斜率前處理與低通濾波
            double_slope_signal = sp.double_slope_preprocessing(filtered_signal, resampled_fs)
            filtered_signal = filtfilt(process_DS_b, process_DS_a, double_slope_signal)
            integrated_signal = sp.moving_window_integration(filtered_signal, window_size=3)
            
            # (f) 動態門檻與 QRS 峰值偵測
            self._detect_qrs_peaks(integrated_signal, time_chunk, dtcwt_denoise_results,
                                     shift_total_time, resampled_fs)

            # 保留最近 max_samples 個點
            if len(self.resampled_freq_list) > max_samples:
                self.resampled_freq_list = self.resampled_freq_list[-max_samples:]
                self.time_list = self.time_list[-max_samples:]
        
        # 使用 r_peak_records 計算 HRV 統計數據（例如 30 秒視窗）
        hrv_stats = sa.analyze_r_peak_records_30s(self.r_peak_records, self.total_chunks)
        import csv
        output_file = f"./output.csv"
        with open(output_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["R Peak Times", "R Peak Features", "Instant HR (BPM)", "Avg HR (30s) (BPM)"])
            for record in self.r_peak_records:
                writer.writerow(record)
        print(f"CSV 檔案已保存至 {output_file}")
        return hrv_stats

    def _detect_qrs_peaks(self, integrated_signal, time_chunk, dtcwt_denoise_results,
                          shift_total_time, resampled_fs):
        """
        將動態門檻與 QRS 峰值偵測邏輯抽出成一個獨立方法，
        並直接使用 self 的成員變數進行狀態管理。
        """
        for idx_inner, (value, time_val, raw_data) in enumerate(zip(integrated_signal, time_chunk - shift_total_time, dtcwt_denoise_results)):
            self.sliding_window.append(raw_data)
            if idx_inner % resampled_fs == 0 and idx_inner != 0:
                self.sliding_window_STD = np.nanstd(list(self.sliding_window))
                self.sliding_window_VAR = np.nanvar(list(self.sliding_window))
            if self.sliding_window_STD < 75 and self.sliding_window_VAR < 1250:
                self.peak_buffer.append(value)
                if len(self.peak_buffer) > 8:
                    self.peak_buffer.pop(0)
                if len(self.recent_8_QRS_peaks) > 2:
                    recent_8_QRS_peaks_avg = np.nanmean(self.recent_8_QRS_peaks)
                else:
                    recent_8_QRS_peaks_avg = self.threshold_high_lim

                if value > self.threshold_high:
                    self.threshold_high = recent_8_QRS_peaks_avg * 0.75 * 0.025 + self.threshold_high * 0.975
                    self.threshold_low = recent_8_QRS_peaks_avg * 0.4 * 0.025 + self.threshold_low * 0.975
                    self.th_state = 2
                elif value > self.threshold_low:
                    self.threshold_high = recent_8_QRS_peaks_avg * 0.75 * 0.005 + self.threshold_high * 0.995
                    self.threshold_low = recent_8_QRS_peaks_avg * 0.25 * 0.005 + self.threshold_low * 0.995
                    self.th_state = 1

                if self.threshold_low > value:
                    self.th_state = 0
                    self.no_detected += 1
                    if self.no_detected > (resampled_fs * 3):
                        self.no_detected = 0
                        self.threshold_low = self.threshold_low_lim
                        self.threshold_high = self.threshold_high_lim
                else:
                    self.no_detected = 0

                if self.threshold_high_lim > self.threshold_high:
                    self.threshold_high = self.threshold_high_lim
                if self.threshold_low_lim > self.threshold_low:
                    self.threshold_low = self.threshold_low_lim

                if self.th_state > 0:
                    self.current_segment_features.append(value)
                    self.current_segment_times.append(time_val)
                    if self.th_state == 2:
                        self.temp_th_state = 2
                    elif self.temp_th_state != 2:
                        self.temp_th_state = 1
                elif self.current_segment_features:
                    max_idx = np.argmax(self.current_segment_features)
                    candidate_time = self.current_segment_times[max_idx]
                    candidate_features = self.current_segment_features[max_idx]
                    time_array = np.array(self.time_list)
                    index = np.argmin(np.abs(time_array - candidate_time))

                    valid_candidate = True
                    recent_8_avg_rr = 1
                    if len(self.recent_8_QRS_time) == 8:
                        current_rr_interval = candidate_time - self.recent_8_QRS_time[-1]
                        recent_8_rr_intervals = np.diff(self.recent_8_QRS_time)
                        filtered_rr_intervals = [rr for rr in recent_8_rr_intervals if rr <= 1.5]
                        recent_8_avg_rr = np.nanmean(filtered_rr_intervals) if filtered_rr_intervals else 0
                        if self.recent_8_QRS_time:
                            current_rr_interval = candidate_time - self.recent_8_QRS_time[-1]
                        else:
                            current_rr_interval = 0
                        if len(self.recent_8_QRS_time) > 1:
                            recent_8_rr_intervals = np.diff(self.recent_8_QRS_time)
                        else:
                            recent_8_rr_intervals = np.array([])
                        if recent_8_rr_intervals.size > 0:
                            filtered_rr_intervals = [rr for rr in recent_8_rr_intervals if rr <= 1.5]
                        else:
                            filtered_rr_intervals = []
                        if filtered_rr_intervals:
                            recent_8_avg_rr = np.nanmean(filtered_rr_intervals)
                        else:
                            recent_8_avg_rr = 0
                        if recent_8_rr_intervals.size > 0:
                            recent_8_median_rr = np.nanmedian(recent_8_rr_intervals)
                        else:
                            recent_8_median_rr = 0
                        seg_time_diff = round(self.current_segment_times[-1] - self.current_segment_times[0], 4) if self.current_segment_times else 0
                        err_time_diff = round(self.error_segment_times[-1] - self.error_segment_times[0], 4) if self.error_segment_times else 0
                        std_sliding = round(np.nanstd(list(self.sliding_window)), 4) if self.sliding_window else 0
                        var_sliding = round(np.nanvar(list(self.sliding_window)), 4) if self.sliding_window else 0
                        mean_sliding = round(np.nanmean(list(self.sliding_window)), 4) if self.sliding_window else 0

                        if (recent_8_QRS_peaks_avg * 2 < candidate_features):
                            valid_candidate = False
                        elif std_sliding > 75 and var_sliding > 750:
                            valid_candidate = False
                        elif self.error_segment_times and (self.error_segment_times[-1] - self.error_segment_times[0] < 0.2):
                            valid_candidate = False
                        elif current_rr_interval > recent_8_avg_rr * 2:
                            valid_candidate = True
                        elif self.temp_th_state == 2:
                            valid_candidate = True
                        elif (recent_8_avg_rr * 0.6 < current_rr_interval) and (current_rr_interval < recent_8_avg_rr * 1.4) and (current_rr_interval > 0.24):
                            valid_candidate = True
                        else:
                            valid_candidate = False
                    if valid_candidate:
                        if len(self.recent_8_QRS_time) == 8 and current_rr_interval < 0.4:
                            if candidate_features > self.r_peak_features[-1]:
                                self.r_peak_times[-1] = candidate_time
                                self.r_peak_features[-1] = candidate_features
                                if len(self.recent_8_QRS_peaks) < 1:
                                    self.recent_8_QRS_peaks.append(candidate_features)
                                    self.recent_8_QRS_time.append(candidate_time)
                                else:
                                    self.recent_8_QRS_peaks[-1] = candidate_features
                                    self.recent_8_QRS_time[-1] = candidate_time
                                self.resampled_freq_features[-1] = self.resampled_freq_list[index]
                                if len(self.r_peak_times) >= 2:
                                    rr_interval = self.r_peak_times[-1] - self.r_peak_times[-2]
                                    self.inst_hr = 60.0 / rr_interval if (recent_8_avg_rr * 0.6 < rr_interval < recent_8_avg_rr * 1.4) else 0.0
                                else:
                                    self.inst_hr = 0.0
                                self.avg_hr = 0.0 if not recent_8_avg_rr else 60 / recent_8_avg_rr
                                self.r_peak_records[-1] = (candidate_time, candidate_features, self.inst_hr, self.avg_hr)
                        else:
                            self.r_peak_times.append(candidate_time)
                            self.r_peak_features.append(candidate_features)
                            self.recent_8_QRS_peaks.append(candidate_features)
                            self.recent_8_QRS_time.append(candidate_time)
                            self.resampled_freq_features.append(self.resampled_freq_list[index])
                            if len(self.r_peak_times) > 2:
                                rr_interval = self.r_peak_times[-1] - self.r_peak_times[-2]
                                if len(self.recent_8_QRS_time) == 8:
                                    self.inst_hr = 60.0 / rr_interval if (recent_8_avg_rr * 0.6 < rr_interval < recent_8_avg_rr * 1.4) else 0.0
                                else:
                                    self.inst_hr = 60.0 / rr_interval if 0.4 < rr_interval < 1.5 else 0.0
                            else:
                                self.inst_hr = 0.0
                            self.avg_hr = 0.0
                            self.r_peak_records.append((candidate_time, candidate_features, self.inst_hr, self.avg_hr))
                    self.current_segment_features = []
                    self.current_segment_times = []
                    self.error_segment_times = []
                    self.temp_th_state = 0
                else:
                    self.error_segment_times.append(time_val)
