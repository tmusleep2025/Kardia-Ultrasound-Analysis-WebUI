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
    def post_process_r_peaks(r_peak_times, r_peak_features, min_rr=0.5, max_rr=1.75):
        """
        Post-process detected R peaks to check morphology and RR interval validity.
        對偵測到的 R 波進行後處理，檢查 RR 間期合理性。
        """
        filtered_r_peak_times = []
        filtered_r_peak_features = []
        valid_rr_intervals = []
        
        if r_peak_times:
            filtered_r_peak_times.append(r_peak_times[0])
            filtered_r_peak_features.append(r_peak_features[0])
        
        # RR 間期檢查（當前與前後各一個）
        for i in range(len(r_peak_times)):
            if i == 0:  # 第一個點，只能與下一個點比較
                rr_next = r_peak_times[i + 1] - r_peak_times[i]
                if min_rr <= rr_next <= max_rr:
                    valid_rr_intervals.append(rr_next)
                    filtered_r_peak_times.append(r_peak_times[i])
                    filtered_r_peak_features.append(r_peak_features[i])

            elif i == len(r_peak_times) - 1:  # 最後一個點，只能與前一個點比較
                rr_prev = r_peak_times[i] - r_peak_times[i - 1]
                if min_rr <= rr_prev <= max_rr:
                    valid_rr_intervals.append(rr_prev)
                    filtered_r_peak_times.append(r_peak_times[i])
                    filtered_r_peak_features.append(r_peak_features[i])

            else:  # 中間的點，與前後點比較
                rr_prev = r_peak_times[i] - r_peak_times[i - 1]
                rr_next = r_peak_times[i + 1] - r_peak_times[i]
                if min_rr <= rr_prev <= max_rr or min_rr <= rr_next <= max_rr:
                    valid_rr_intervals.append(rr_prev if min_rr <= rr_prev <= max_rr else rr_next)
                    filtered_r_peak_times.append(r_peak_times[i])
                    filtered_r_peak_features.append(r_peak_features[i])

        return filtered_r_peak_times, filtered_r_peak_features, valid_rr_intervals

    @staticmethod
    def wavelet_denoise(signal, wavelet='db4', level=9, remove_levels=None):
        """
        使用小波分解與重建對訊號進行降噪。
        
        :param signal: 要降噪的訊號
        :param wavelet: 小波基函數名稱 (預設為 'db4')
        :param level: 分解層級 (預設為 9)
        :param remove_levels: 需要歸零的層級 (默認去除[0, 1, 2, 3, 8, 9])
        :return: 降噪後的訊號
        """
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        
        if remove_levels is None:
            remove_levels = [0, 1, 2, 3, 9]  # 預設要去除的層級

        for i in remove_levels:
            if i < len(coeffs):
                coeffs[i] = np.zeros_like(coeffs[i])

        return pywt.waverec(coeffs, wavelet)

    def process_audio(self):
        """
        Process the audio file to extract ECG statistics using squared derivative features.
        處理音訊檔案，使用平方導數特徵來提取 ECG 統計資訊。
        """
        sample_rate, audio_data = wavfile.read(self.file_path)
        if audio_data.ndim > 1:
            audio_data = audio_data[:, 0]

        # 帶通濾波：18500-19300 Hz
        freq_min = 18500  # 18.5 kHz
        freq_max = 19300  # 19.3 kHz

        bandpass_high = ButterworthFilter(
            'band', cutoff=None, fs=sample_rate, order=5, lowcut=freq_min, highcut=freq_max
        )
        filtered_audio = bandpass_high.apply(audio_data)
        
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
        
        frequencies, times, Sxx = spectrogram(filtered_audio, fs=sample_rate, nperseg=16384, noverlap=8192)

        # 篩選頻率範圍
        freq_indices = np.where((frequencies >= freq_min) & (frequencies <= freq_max))[0]
        filtered_frequencies = frequencies[freq_indices]
        filtered_Sxx = Sxx[freq_indices, :]

        # 計算每個時間點最大振幅的頻率
        max_indices = np.argmax(filtered_Sxx, axis=0)
        max_frequencies = filtered_frequencies[max_indices]

        # 設定滑動窗口參數
        window_size = 1.0  # 1秒的窗口
        step_size = 0.25   # 每次移動0.25秒

        # 轉換窗口大小和步長為索引數
        window_samples = int(window_size / (times[1] - times[0]))  # 窗口大小對應索引
        step_samples = int(step_size / (times[1] - times[0]))  # 步長對應索引

        # 計算滑動標準差
        std_devs = []
        time_stamps = []
        threshold = 100  # 設定異常標準差閾值

        for start in range(0, len(times) - window_samples + 1, step_samples):
            window_data = max_frequencies[start:start + window_samples]
            std_dev = np.std(window_data)
            std_devs.append(std_dev)
            time_stamps.append(times[start + window_samples - 1])

        # 標記異常點
        anomalies =  np.array(std_devs) > threshold

        anomalous_time = np.sum(anomalies) * step_size  # 每個異常點代表 step_size 秒

        if (anomalous_time > 5):
            # print(f"⚠️ Signal has strong fluctuation over 5 sec ({anomalous_time})")
            self.ecg_stats = {
                "Number of R Waves": 0,
                "Avg. RR Interval (s)": 0,
                "RR Interval SD (s)": 0,
                "RMSSD (s)": 0,
                "Min RR Interval (s)": 0,
                "Max RR Interval (s)": 0,
                "Average Heart Rate (bpm)": "N/A",
                "state": 2,
                "total_exceeding_samples": anomalous_time
            }
            return self.ecg_stats, None
        
        # 希爾伯特變換取得包絡線並重採樣
        analytic_signal = hilbert(filtered_audio)
        amplitude_envelope = np.abs(analytic_signal)

        desired_fs = 200  
        new_length = int(len(amplitude_envelope) * desired_fs / sample_rate)
        amplitude_envelope = resample(amplitude_envelope, new_length)
        amplitude_envelope = self.wavelet_denoise(amplitude_envelope)



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




        from scipy.fftpack import fft
        from scipy.signal import find_peaks

        # 假設 amplitude_envelope 已經存在
        time = np.arange(len(squared_second_deriv)) * 0.005

        # 設定滑動窗口參數
        window_size_sec = 5  # 設定窗口大小為5秒
        overlap_ratio = 0.925  # 設定重疊比率為90%
        T = time[1] - time[0]  # 時間間隔
        window_size = int(window_size_sec / T)
        step_size = max(1, int(window_size * (1 - overlap_ratio)))

        detected_periods = []
        window_start_times = []

        # 滑動窗口週期檢測
        for start in range(0, len(squared_second_deriv) - window_size + 1, step_size):
            segment = squared_second_deriv[start:start + window_size]
            
            # 執行 FFT 分析
            segment_fft = np.abs(fft(segment)[:window_size // 2])
            segment_freqs = np.fft.fftfreq(window_size, d=T)[:window_size // 2]
            
            # 找出主要頻率
            peak_indices, _ = find_peaks(segment_fft, height=segment_fft.max() * 0.1)
            
            dominant_period = 1 / segment_freqs[peak_indices[0]] if peak_indices.size > 0 and segment_freqs[peak_indices[0]] > 0 else None
            
            detected_periods.append(dominant_period)
            window_start_times.append(time[start])

        # 找出連續5個 detected_periods 超過2的區域
        threshold = 1.5
        consecutive_count = 4
        exceeding_indices = [i for i in range(len(detected_periods) - consecutive_count + 1) if all(p and p > threshold for p in detected_periods[i:i + consecutive_count])]
        
        # 設置所有起始時間與結束時間範圍內的 amplitude_envelope 為0
        for idx in exceeding_indices:
            start_time = window_start_times[idx]
            end_time = window_start_times[idx + consecutive_count - 1] + 5
            squared_second_deriv[(time >= start_time) & (time <= end_time)] = 0

        # 繪製結果
        # plt.figure(figsize=(12, 6))
        # plt.plot(window_start_times, detected_periods, marker='o', linestyle='-', label='Detected Period (5-sec Window, 90% Overlap)')
        # for idx in exceeding_indices:
        #     plt.plot(window_start_times[idx:idx + consecutive_count], detected_periods[idx:idx + consecutive_count], 'ro', label='Exceeding Threshold' if idx == exceeding_indices[0] else "")
        # plt.xlabel('Time (s)')
        # plt.ylabel('Detected Period (Seconds)')
        # plt.title('Sliding Window (5-sec, 90% Overlap) Analysis of Detected Period')
        # plt.grid(True)
        # plt.legend()
        # plt.show()

        if np.sum(squared_second_deriv == 0) > 10 * 200:
            print("⚠️ TTTT")
            print(np.sum(squared_second_deriv == 0))
            self.ecg_stats = {
                "Number of R Waves": 0,
                "Avg. RR Interval (s)": 0,
                "RR Interval SD (s)": 0,
                "RMSSD (s)": 0,
                "Min RR Interval (s)": 0,
                "Max RR Interval (s)": 0,
                "Average Heart Rate (bpm)": "N/A",
                "state": 3,
            }
            return self.ecg_stats, None

        # R 波偵測
        r_peak_times, r_peak_features, dynamic_thresholds_1, dynamic_thresholds_2 = self.detect_r_peaks(
            time_axis, squared_second_deriv, self.refractory_period, new_length
        )

        # 後處理 R 波
        filtered_r_peak_times, filtered_r_peak_features, valid_rr_intervals = self.post_process_r_peaks(
            r_peak_times, r_peak_features, min_rr=self.refractory_period, max_rr=1.5
        )

        self.ecg_stats = self.calculate_ecg_statistics(valid_rr_intervals)

        self.plot_data = {
            "time_axis": time_axis,
            "frequencies": frequencies,
            "times": times,
            "Sxx": Sxx,
            "squared_second_deriv": squared_second_deriv,
            "amplitude_envelope": amplitude_envelope,
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
        axs[1].set_xlim(0, t[len(t)-1])
        axs[1].set_title('original KardiaMobile ultrasonic amplitude')
        
        axs[2].plot(t, squared_second_deriv, label="enhanced Signal", color='blue')
        axs[2].plot(t, dynamic_thresholds_1, label="Th_L", linestyle='--', color='green')
        axs[2].plot(t, dynamic_thresholds_2, label="Th_H", linestyle='--', color='red')
        axs[2].scatter(r_peak_times, r_peak_features, label="Detected QRS Peaks", color='red', marker='o')
        axs[2].set_xlabel("Time (s)")
        axs[2].set_ylabel("Feature Amplitude")
        axs[2].set_title("enhanced algorithm")
        axs[2].set_xlim(0, t[len(t)-1])
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
        r_peak_times = self.plot_data["r_peak_times"]
        r_peak_features = self.plot_data["r_peak_features"]



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


# analyzer = ECGAnalyzer(r"C:\Users\User\Documents\GitHub\Kardia_Flask_WebUI\Flask\uploads\segment_30.wav", refractory_period=0.5)
# ecg_stats, plot_data = analyzer.analyze_and_plot_ecg()
# print(ecg_stats)