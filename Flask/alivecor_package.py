import numpy as np
from scipy.io import wavfile
from scipy.signal import stft, butter, filtfilt, spectrogram, hilbert, resample
import matplotlib.pyplot as plt
import pywt
import pandas as pd

class ButterworthFilter:
    """
    A class to design and apply Butterworth filters.
    用於設計並套用 Butterworth 濾波器的類別。
    """
    def __init__(self, filter_type, cutoff, fs, order=5, lowcut=None, highcut=None):
        """
        Initialize filter parameters and design the filter.
        初始化濾波器參數並設計濾波器。
        """
        self.filter_type = filter_type
        self.cutoff = cutoff
        self.fs = fs
        self.order = order
        self.lowcut = lowcut
        self.highcut = highcut
        self.b, self.a = self.design_filter()

    def design_filter(self):
        """
        Design the Butterworth filter based on the specified parameters.
        根據指定參數設計 Butterworth 濾波器。
        """
        nyquist = 0.5 * self.fs
        if self.filter_type == 'low':
            normalized_cutoff = self.cutoff / nyquist
            return butter(self.order, normalized_cutoff, btype='low', analog=False)
        elif self.filter_type == 'high':
            normalized_cutoff = self.cutoff / nyquist
            return butter(self.order, normalized_cutoff, btype='high', analog=False)
        elif self.filter_type == 'band':
            low = self.lowcut / nyquist
            high = self.highcut / nyquist
            return butter(self.order, [low, high], btype='band', analog=False)
        else:
            raise ValueError("Invalid filter type. Choose 'low', 'high', or 'band'.")

    def apply(self, data):
        """
        Apply the designed filter to the input data.
        將設計好的濾波器套用到輸入資料上。
        """
        return filtfilt(self.b, self.a, data)

class ECGAnalyzer:
    """
    A class to handle ECG-related signal processing and analysis from audio data.
    處理並分析音訊中 ECG 訊號的類別。
    """
    def __init__(self, file_path, refractory_period=0.5):
        """
        Initialize with audio file path and refractory period.
        使用音訊檔案路徑及 refractory period 初始化。
        """
        self.file_path = file_path
        self.refractory_period = refractory_period
        self.ecg_stats = {}
        self.plot_data = None

    @staticmethod
    def detect_r_peaks(times, values, refractory_period=0.5, FS=0):
        """
        Detect R peaks using precomputed features.
        使用已計算的特徵值來偵測 R 波。
        """
        threshold_1 = np.nanmax(values[:int(FS * 5)]) * 0.25
        threshold_2 = np.nanmax(values[:int(FS * 5)]) * 0.75
        threshold_1_lim = np.nanmax(values[:int(FS * 5)]) * 0.1
        threshold_2_lim = np.nanmax(values[:int(FS * 5)]) * 0.4

        dynamic_thresholds_1 = []
        dynamic_thresholds_2 = []
        r_peak_times = []
        r_peak_features = []
        current_segment_features = []
        current_segment_times = []
        peak_buffer = []
        above_threshold = False

        for time, value in zip(times, values):
            peak_buf_avg = np.mean(peak_buffer) if peak_buffer else 0

            if not np.isnan(peak_buf_avg):
                if value > threshold_2:
                    threshold_2 = peak_buf_avg * 0.75
                elif threshold_1 < value < threshold_2:
                    threshold_2 -= np.abs(value - peak_buf_avg) / 2
                elif threshold_2_lim > threshold_2:
                    threshold_2 = threshold_2_lim

                if value > threshold_2:
                    threshold_1 = peak_buf_avg * 0.4
                elif threshold_1 < value < threshold_2:
                    threshold_1 = value * 0.4
                elif threshold_1_lim > threshold_1:
                    threshold_1 = threshold_1_lim

            peak_buffer.append(value)
            if len(peak_buffer) > 8:
                peak_buffer.pop(0)

            dynamic_thresholds_1.append(threshold_1)
            dynamic_thresholds_2.append(threshold_2)

            if value > threshold_2:
                current_segment_features.append(value)
                current_segment_times.append(time)
                above_threshold = True
            elif above_threshold:
                if current_segment_features:
                    max_idx = np.argmax(current_segment_features)
                    candidate_time = current_segment_times[max_idx]
                    candidate_features = current_segment_features[max_idx]
                    if not r_peak_times or (candidate_time - r_peak_times[-1]) > refractory_period:
                        r_peak_times.append(candidate_time)
                        r_peak_features.append(candidate_features)
                current_segment_features = []
                current_segment_times = []
                above_threshold = False

        return r_peak_times, r_peak_features, dynamic_thresholds_1, dynamic_thresholds_2

    @staticmethod
    def calculate_ecg_statistics(rr_intervals):
        """
        Calculate ECG statistics based on RR intervals.
        根據 RR 間期計算 ECG 統計量。
        """
        if rr_intervals is None or len(rr_intervals) == 0:
            return {}

        mean_rr = np.mean(rr_intervals)
        sdnn = np.std(rr_intervals)
        rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))
        min_rr = np.min(rr_intervals)
        max_rr = np.max(rr_intervals)
        count_rr = len(rr_intervals)

        stats = {
            "Number of R Waves": count_rr,
            "Avg. RR Interval (s)": f"{mean_rr:.3f}",
            "RR Interval SD (s)": f"{sdnn:.3f}",
            "RMSSD (s)": f"{rmssd:.3f}",
            "Min RR Interval (s)": f"{min_rr:.3f}",
            "Max RR Interval (s)": f"{max_rr:.3f}",
            "Average Heart Rate (bpm)": f"{60 / mean_rr:.2f}" if mean_rr != 0 else "N/A"
        }
        return stats

    @staticmethod
    def post_process_r_peaks(r_peak_times, r_peak_features, time_axis, squared_second_deriv, 
                             min_rr=0.5, max_rr=1.75, morphology_window=0.05, prominence_factor=2.0):
        """
        Post-process detected R peaks to check morphology and RR interval validity.
        對偵測到的 R 波進行後處理，檢查形態與 RR 間期合理性。
        """
        valid_indices = []
        fs = 1 / (time_axis[1] - time_axis[0])
        sample_window = int(morphology_window * fs)
        
        # 形態學檢查 R 波
        for i, r_time in enumerate(r_peak_times):
            idx = np.argmin(np.abs(time_axis - r_time))
            start = max(idx - sample_window, 0)
            end = min(idx + sample_window, len(squared_second_deriv) - 1)
            local_window = squared_second_deriv[start:end+1]
            local_peak = squared_second_deriv[idx]
            if local_peak == np.max(local_window) and local_peak > prominence_factor * np.mean(local_window):
                valid_indices.append(i)
        
        morphology_filtered_times = [r_peak_times[i] for i in valid_indices]
        morphology_filtered_features = [r_peak_features[i] for i in valid_indices]
        
        filtered_r_peak_times = []
        filtered_r_peak_features = []
        
        if morphology_filtered_times:
            filtered_r_peak_times.append(morphology_filtered_times[0])
            filtered_r_peak_features.append(morphology_filtered_features[0])
        
        # RR 間期檢查
        for i in range(1, len(morphology_filtered_times)):
            rr_interval = morphology_filtered_times[i] - morphology_filtered_times[i-1]
            if min_rr <= rr_interval <= max_rr:
                filtered_r_peak_times.append(morphology_filtered_times[i])
                filtered_r_peak_features.append(morphology_filtered_features[i])
        
        return filtered_r_peak_times, filtered_r_peak_features

    @staticmethod
    def wavelet_denoise(signal, wavelet='db4', level=3):
        """
        Denoise the signal using wavelet decomposition and reconstruction.
        使用小波分解與重建對訊號進行降噪。
        """
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        coeffs[0] = np.zeros_like(coeffs[0])
        coeffs[1] = np.zeros_like(coeffs[1])
        return pywt.waverec(coeffs, wavelet)

    @staticmethod
    def moving_std(signal, window_size=200):
        """
        Compute moving standard deviation over a specified window size.
        在指定視窗大小下計算移動標準差。
        """
        return np.array([np.std(signal[i:i+window_size]) for i in range(len(signal)-window_size)])

    def process_audio(self):
        """
        Process the audio file to extract ECG statistics using squared derivative features.
        處理音訊檔案，使用平方導數特徵來提取 ECG 統計資訊。
        """
        sample_rate, audio_data = wavfile.read(self.file_path)
        if audio_data.ndim > 1:
            audio_data = audio_data[:, 0]

        # 帶通濾波：18500-19300 Hz
        bandpass_high = ButterworthFilter(
            'band', cutoff=None, fs=sample_rate, order=5, lowcut=18500, highcut=19300
        )
        filtered_audio = bandpass_high.apply(audio_data)

        frequencies, times, Sxx = spectrogram(filtered_audio, fs=sample_rate, nperseg=16384, noverlap=8192)

        # 檢查信號強度
        if np.std(filtered_audio) < 10:
            # print("⚠️ No Signal")
            self.ecg_stats = {
                "Number of R Waves": 0,
                "Avg. RR Interval (s)": 0,
                "RR Interval SD (s)": 0,
                "RMSSD (s)": 0,
                "Min RR Interval (s)": 0,
                "Max RR Interval (s)": 0,
                "Average Heart Rate (bpm)": "N/A",
                "state": 1,
                "filtered_audio_std": np.std(filtered_audio)
            }
            return self.ecg_stats, None

        # 希爾伯特變換取得包絡線並重採樣
        analytic_signal = hilbert(filtered_audio)
        amplitude_envelope = np.abs(analytic_signal)

        desired_fs = 200  
        new_length = int(len(amplitude_envelope) * desired_fs / sample_rate)
        amplitude_envelope = resample(amplitude_envelope, new_length)
        amplitude_envelope = self.wavelet_denoise(amplitude_envelope)

        # 計算移動標準差檢查訊號波動
        moving_std_values = self.moving_std(amplitude_envelope, window_size=desired_fs)
        threshold_value = 1.8 * np.mean(moving_std_values)
        threshold_duration_seconds = 7.5
        threshold_samples = threshold_duration_seconds * desired_fs
        exceeding_threshold = moving_std_values > threshold_value
        total_exceeding_samples = np.sum(exceeding_threshold)
        if total_exceeding_samples >= threshold_samples:
            # print(f"⚠️ Signal has strong fluctuation over {threshold_duration_seconds} seconds ({total_exceeding_samples})")
            self.ecg_stats = {
                "Number of R Waves": 0,
                "Avg. RR Interval (s)": 0,
                "RR Interval SD (s)": 0,
                "RMSSD (s)": 0,
                "Min RR Interval (s)": 0,
                "Max RR Interval (s)": 0,
                "Average Heart Rate (bpm)": "N/A",
                "state": 2,
                "total_exceeding_samples": total_exceeding_samples
            }
            return self.ecg_stats, None

        dt = 1.0 / desired_fs
        time_axis = np.arange(len(amplitude_envelope)) / desired_fs

        # 計算包絡線的一階與二階導數，並取平方
        first_deriv = np.gradient(amplitude_envelope, dt)
        second_deriv = np.gradient(first_deriv, dt)
        squared_second_deriv = second_deriv ** 2

        # 去除極端值
        lower_bound = np.percentile(squared_second_deriv, 1)
        upper_bound = np.percentile(squared_second_deriv, 99)
        squared_second_deriv = np.where(
            (squared_second_deriv < lower_bound) | (squared_second_deriv > upper_bound), 
            0, 
            squared_second_deriv
        )

        # 進一步濾波
        filter_test = ButterworthFilter('band', cutoff=None, fs=desired_fs, order=1, lowcut=1, highcut=10)
        squared_second_deriv = filter_test.apply(squared_second_deriv)

        # R 波偵測
        r_peak_times, r_peak_features, dynamic_thresholds_1, dynamic_thresholds_2 = self.detect_r_peaks(
            time_axis, squared_second_deriv, self.refractory_period, new_length
        )

        # 後處理 R 波
        filtered_r_peak_times, filtered_r_peak_features = self.post_process_r_peaks(
            r_peak_times, r_peak_features, time_axis, squared_second_deriv,
            min_rr=0.5, max_rr=1.75, morphology_window=0.025, prominence_factor=1.0
        )

        if len(filtered_r_peak_times) > 1:
            valid_rr_intervals = np.diff(filtered_r_peak_times)
        else:
            valid_rr_intervals = np.array([])

        self.ecg_stats = self.calculate_ecg_statistics(valid_rr_intervals)

        # 簡單檢查 RMSSD 是否異常
        if ("RMSSD (s)" in self.ecg_stats 
            and float(self.ecg_stats["RMSSD (s)"]) > 1):
            # print("⚠️ RR val fail")
            self.ecg_stats = {
                "Number of R Waves": 0,
                "Avg. RR Interval (s)": 0,
                "RR Interval SD (s)": 0,
                "RMSSD (s)": 0,
                "Min RR Interval (s)": 0,
                "Max RR Interval (s)": 0,
                "Average Heart Rate (bpm)": "N/A",
                "state": 3,
                "rmssd_value": self.ecg_stats["RMSSD (s)"],
                "total_exceeding_samples": total_exceeding_samples
            }
            return self.ecg_stats, None

        self.plot_data = {
            "time_axis": time_axis,
            "frequencies": frequencies,
            "times": times,
            "Sxx": Sxx,
            "squared_second_deriv": squared_second_deriv,
            "amplitude_envelope": amplitude_envelope,
            "moving_std_values": moving_std_values,
            "exceeding_threshold": exceeding_threshold,
            "threshold_value": threshold_value,
            "r_peak_times": filtered_r_peak_times,
            "r_peak_features": filtered_r_peak_features,
            "dynamic_thresholds_1": dynamic_thresholds_1,
            "dynamic_thresholds_2": dynamic_thresholds_2,
            "ecg_stats": self.ecg_stats
        }

        return self.ecg_stats, self.plot_data

    def analyze_ecg(self):
        """
        Analyze ECG and return statistical results.
        分析 ECG 並回傳統計結果。
        """
        self.ecg_stats, _ = self.process_audio()
        return self.ecg_stats

    def analyze_and_plot_ecg(self):
        """
        Analyze ECG, generate plots, and return statistics and plot data.
        分析 ECG、產生圖表並回傳統計結果及繪圖資料。
        """
        self.ecg_stats, self.plot_data = self.process_audio()
        if self.plot_data is None:
            return self.ecg_stats, None

        # 取得繪圖所需資料
        t = self.plot_data["time_axis"]
        frequencies = self.plot_data["frequencies"]
        times = self.plot_data["times"]
        Sxx = self.plot_data["Sxx"]
        squared_second_deriv = self.plot_data["squared_second_deriv"]
        amplitude_envelope = self.plot_data["amplitude_envelope"]
        dynamic_thresholds_1 = self.plot_data["dynamic_thresholds_1"]
        dynamic_thresholds_2 = self.plot_data["dynamic_thresholds_2"]
        r_peak_times = self.plot_data["r_peak_times"]
        r_peak_features = self.plot_data["r_peak_features"]

        # 繪製圖表
        fig, axs = plt.subplots(3, 1, figsize=(18, 32), constrained_layout=True)

        # pcm = axs[0].pcolormesh(times, frequencies, 10 * np.log10(Sxx + 1e-10), shading='gouraud', vmin=-100, vmax=10)
        pcm = axs[0].pcolormesh(times, frequencies, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
        axs[0].set_ylabel('Frequency [Hz]')
        axs[0].set_xlabel('Time [s]')
        axs[0].set_title('original KardiaMobile ultrasonic spectrogram (18.5-19.3 kHz)')
        axs[0].set_ylim(18500, 19300)
        # fig.colorbar(pcm, ax=axs[0], label='Intensity [dB]')

        axs[1].plot(t, amplitude_envelope, color='orange')
        axs[1].set_ylabel('Amplitude')
        axs[1].set_xlabel('Time [s]')
        axs[1].set_xlim(0, 60)
        axs[1].set_title('original KardiaMobile ultrasonic amplitude')
        
        axs[2].plot(t, squared_second_deriv, label="enhanced Signal", color='blue')
        axs[2].plot(t, dynamic_thresholds_1, label="Th_L", linestyle='--', color='green')
        axs[2].plot(t, dynamic_thresholds_2, label="Th_H", linestyle='--', color='red')
        axs[2].scatter(r_peak_times, r_peak_features, label="Detected QRS Peaks", color='red', marker='o')
        axs[2].set_xlabel("Time (s)")
        axs[2].set_ylabel("Feature Amplitude")
        axs[2].set_title("enhanced algorithm")
        axs[2].set_xlim(0, 60)
        axs[2].legend(loc='upper right')

        plt.subplots_adjust(left=0.06, right=0.99, top=0.955, bottom=0.05, hspace=0.3, wspace=0.2)
        plt.show()

        return self.ecg_stats, self.plot_data
    
    def analyze_and_plot_Kardia_ecg_and_PSG_ECG(self):
        """
        Analyze ECG, generate plots, and return statistics and plot data.
        分析 ECG、產生圖表並回傳統計結果及繪圖資料。
        """
        self.ecg_stats, self.plot_data = self.process_audio()
        if self.plot_data is None:
            return self.ecg_stats, None

        # 取得繪圖所需資料
        t = self.plot_data["time_axis"]
        frequencies = self.plot_data["frequencies"]
        times = self.plot_data["times"]
        Sxx = self.plot_data["Sxx"]
        squared_second_deriv = self.plot_data["squared_second_deriv"]
        amplitude_envelope = self.plot_data["amplitude_envelope"]
        dynamic_thresholds_1 = self.plot_data["dynamic_thresholds_1"]
        dynamic_thresholds_2 = self.plot_data["dynamic_thresholds_2"]
        r_peak_times = self.plot_data["r_peak_times"]
        r_peak_features = self.plot_data["r_peak_features"]
        moving_std_values = self.plot_data["moving_std_values"]
        threshold_value = self.plot_data["threshold_value"]
        exceeding_threshold = self.plot_data["exceeding_threshold"]



        # 繪製圖表
        fig, axs = plt.subplots(4, 1, figsize=(18, 32), constrained_layout=True)

        pcm = axs[0].pcolormesh(times, frequencies, 10 * np.log10(Sxx + 1e-10), shading='gouraud', vmin=-100, vmax=0)
        axs[0].set_ylabel('Frequency [Hz]')
        axs[0].set_xlabel('Time [s]')
        axs[0].set_title('original KardiaMobile ultrasonic spectrogram (18.5-19.3 kHz)')
        axs[0].set_ylim(18500, 19300)
        # fig.colorbar(pcm, ax=axs[0], label='Intensity [dB]')

        axs[1].plot(t, amplitude_envelope, color='orange')
        axs[1].set_ylabel('Amplitude')
        axs[1].set_xlabel('Time [s]')
        axs[1].set_xlim(0, 60)
        axs[1].set_title('original KardiaMobile ultrasonic amplitude')
        

        # 讀取 CSV 檔案
        file_path = './PSG_ECG.csv'  # 替換為你的 CSV 文件路徑
        df = pd.read_csv(file_path)

        if 'ECG' not in df.columns:
            raise ValueError("CSV 檔案中必須包含 'ECG' 欄位")

        original_indices = np.arange(len(df))
        axs[2].plot(original_indices, df['ECG'], label='ECG Signal', color='blue')
        axs[2].set_ylabel('ECG Value')
        axs[2].set_xlabel('Time [s]')
        x_offset  = 1540
        axs[2].set_xlim(x_offset , 18000 + x_offset )
        # 設定 X 軸刻度為 0 到 60（均勻間隔）
        axs[2].set_xticks(np.linspace(x_offset, x_offset + 18000, num=7))
        axs[2].set_xticklabels([0, 10, 20, 30, 40, 50, 60])  # 設定自訂標籤
        axs[2].set_ylim(-2000, 2000)
        axs[2].set_title('original ECG from PSG')

        axs[3].plot(t, squared_second_deriv, label="enhanced Signal", color='blue')
        # axs[3].plot(t, dynamic_thresholds_1, label="Th_L", linestyle='--', color='green')
        # axs[3].plot(t, dynamic_thresholds_2, label="Th_H", linestyle='--', color='red')
        axs[3].scatter(r_peak_times, r_peak_features, label="Detected QRS Peaks", color='red', marker='o')
        axs[3].set_xlabel("Time (s)")
        axs[3].set_ylabel("Feature Amplitude")
        axs[3].set_title("enhanced algorithm")
        axs[3].set_xlim(0, 60)
        axs[3].legend(loc='upper right')

        plt.subplots_adjust(left=0.06, right=0.99, top=0.955, bottom=0.05, hspace=0.3, wspace=0.2)
        plt.show()

        return self.ecg_stats, self.plot_data

# ============================
# Example usage
# ============================
# analyzer = ECGAnalyzer('./path_to_your_audio.wav', refractory_period=0.5)
# ecg_stats_only = analyzer.analyze_ecg()
# ecg_stats, plot_data = analyzer.analyze_and_plot_ecg()
# print(ecg_stats)