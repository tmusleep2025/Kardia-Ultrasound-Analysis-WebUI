# stats_analysis.py
import numpy as np
import pandas as pd
import math

def analyze_r_peak_records_30s(r_peak_records, audio_end_idx):
    """
    依據輸入的 r_peak_records (格式: (candidate_time, candidate_features, inst_hr, avg_hr))
    進行 30 秒視窗的 HRV 分析，僅回傳下列參數：
      - Start Time (s)
      - 30s_End Time (s)
      - Avg HR (BPM)
      - SDNN (ms)
      - RMSSD (ms)
      - SD1 (ms)
      - SD2 (ms)
      - Poincare S (ms^2)
      - SD1/SD2
    """
    def compute_poincare_parameters(rr_ms):
        if len(rr_ms) < 2:
            return np.nan, np.nan, np.nan, np.nan
        x = rr_ms[:-1]
        y = rr_ms[1:]
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        x_c = x - x_mean
        y_c = y - y_mean
        cov_matrix = np.cov(x_c, y_c)
        eigvals, _ = np.linalg.eig(cov_matrix)
        eigvals_sorted = np.sort(eigvals)
        SD1 = np.sqrt(np.maximum(eigvals_sorted[0], 0))
        SD2 = np.sqrt(np.maximum(eigvals_sorted[1], 0))
        S = math.pi * SD1 * SD2
        ratio = SD1 / SD2 if SD2 != 0 else np.nan
        return SD1, SD2, S, ratio

    # 將 r_peak_records 轉換為 DataFrame
    df = pd.DataFrame(r_peak_records, columns=["R Peak Times", "Candidate Features", "Instant HR (BPM)", "Avg HR"])

    # 篩選 Instant HR (BPM) > 0 的有效資料
    valid_indices = df[df["Instant HR (BPM)"] > 0].index

    # 計算 Corrected RR Interval（利用前後 R Peak Times 差值）
    corrected_rr_intervals = []
    for idx in valid_indices:
        if idx == 0:
            corrected_rr_intervals.append(None)
        else:
            previous_r_peak_time = df.loc[idx - 1, "R Peak Times"]
            current_r_peak_time = df.loc[idx, "R Peak Times"]
            corrected_rr_intervals.append(current_r_peak_time - previous_r_peak_time)

    # 選取有效資料並刪除第一筆（因為無法計算 RR Interval）
    valid_hr_df = df.loc[valid_indices].copy().iloc[1:].copy()
    valid_hr_df["Corrected RR Interval (s)"] = corrected_rr_intervals[1:]

    window_size = 30  # 30 秒視窗
    results = []
    start_time = 0
    current_start = start_time
    end_time = audio_end_idx * 30
    while current_start + window_size <= end_time:
        window_df = valid_hr_df[(valid_hr_df["R Peak Times"] >= current_start) & 
                       (valid_hr_df["R Peak Times"] < current_start + window_size)]
        if len(window_df) > 2:
            rr_list = window_df["Corrected RR Interval (s)"].dropna().tolist()
            rr_ms = np.array(rr_list) * 1000.0  # 換算成毫秒
            SDNN = np.std(rr_ms, ddof=1) if len(rr_ms) > 1 else np.nan
            RMSSD = np.sqrt(np.mean(np.diff(rr_ms)**2)) if len(rr_ms) > 1 else np.nan
            SD1, SD2, area_S, ratio = compute_poincare_parameters(rr_ms)
            avg_hr = window_df["Instant HR (BPM)"].mean()
            result = {
                "Start Time (s)": current_start,
                "30s_End Time (s)": current_start + window_size,
                "Avg HR (BPM)": avg_hr,
                "SDNN (ms)": SDNN,
                "RMSSD (ms)": RMSSD,
                "SD1 (ms)": SD1,
                "SD2 (ms)": SD2,
                "Poincare S (ms^2)": area_S,
                "SD1/SD2": ratio,
                "R Peak total": len(window_df)
            }
        else:
            result = {
                "Start Time (s)": current_start,
                "30s_End Time (s)": current_start + window_size,
                "Avg HR (BPM)": "N/A",
                "SDNN (ms)": "N/A",
                "RMSSD (ms)": "N/A",
                "SD1 (ms)": "N/A",
                "SD2 (ms)": "N/A",
                "Poincare S (ms^2)": "N/A",
                "SD1/SD2": "N/A"
            }
        results.append(result)
        current_start += window_size
    
    return pd.DataFrame(results)
