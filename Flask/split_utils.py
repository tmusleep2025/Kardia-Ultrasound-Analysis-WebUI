# flask/split_utils.py
import logging
import csv
from scipy.io import wavfile

def split_audio(file_path, segment_duration=30):
    """
    將音訊檔案拆分成多個指定時長的片段。

    參數:
    - file_path (str): 音訊檔案的路徑。
    - segment_duration (int): 每個片段的時長（秒）。

    返回:
    - segments (list of tuples): 每個元組包含 (segment_data, sample_rate)。
    """
    try:
        sample_rate, data = wavfile.read(file_path)
    except Exception as e:
        logging.error(f"Error reading '{file_path}': {str(e)}")
        return []

    # 如果是立體聲，僅取第一聲道
    if len(data.shape) > 1 and data.shape[1] > 1:
        data = data[:,0]

    total_samples = data.shape[0]
    segment_samples = int(segment_duration * sample_rate)

    segments = []
    for start in range(0, total_samples, segment_samples):
        end = start + segment_samples
        segment_data = data[start:end]
        if len(segment_data) == 0:
            continue
        segments.append((segment_data, sample_rate))
    
    return segments

def save_results_to_csv(csv_filename, segments, original_filename):
    """
    將逐段分析結果保存至 CSV 檔案。

    參數:
    - csv_filename (str): CSV 檔案的路徑。
    - segments (list of dict): 逐段分析結果列表。
    - original_filename (str): 原始上傳檔案名稱。
    """
    if not segments:
        logging.warning("No segments to save to CSV.")
        return

    fieldnames = [
        'file_name',
        'Average Heart Rate (bpm)',
        'RMSSD (s)',
        'state',
        'filtered_audio_std',
        'total_exceeding_samples',
        'rmssd_value'
    ]

    try:
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for seg in segments:
                file_name = f"{original_filename}_segment_{seg['segment_index']}"
                writer.writerow({
                    'file_name': file_name,
                    'Average Heart Rate (bpm)': seg['avg_hr'],
                    'RMSSD (s)': seg.get('rmssd', ''),
                    'state': seg['state'],
                    'filtered_audio_std': seg.get('filtered_audio_std', ''),
                    'total_exceeding_samples': seg.get('total_exceeding_samples', ''),
                    'rmssd_value': seg.get('rmssd_value', '')
                })
        logging.info(f"Results saved to {csv_filename}")
    except Exception as e:
        logging.error(f"Error saving results to CSV: {str(e)}")

def plot_results_from_csv(csv_filename='ecg_results.csv', segment_duration=30, output_png='report.png'):
    """
    從 CSV 檔案讀取分析結果，繪製甘特圖和心率變化圖，並備註各個 state 的意義。

    參數:
    - csv_filename (str): 存有分析結果的 CSV 檔案名稱。
    - segment_duration (int): 切片段的時長（秒），用以計算時間軸。
    - output_png (str): 繪製後圖表的輸出路徑。
    """
    import matplotlib
    matplotlib.use('Agg')  # 使用非互動式後端
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator  # 引入定位器

    states_data = []
    hr_data = []

    try:
        with open(csv_filename, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                file_name = row['file_name']
                avg_hr = row['Average Heart Rate (bpm)']
                state = row['state']

                if "_segment_" in file_name:
                    try:
                        seg_num = int(file_name.rsplit("_segment_",1)[1].replace('.wav',''))
                    except:
                        seg_num = 1

                    # 假設每個 segment 持續 segment_duration 秒，從 0 開始
                    start_time = seg_num - 1
                    end_time = seg_num

                    try:
                        hr_value = float(avg_hr) if avg_hr != "N/A" else None
                    except:
                        hr_value = None

                    hr_data.append((start_time, hr_value))

                    if state in {"1","2","3"}:
                        states_data.append((start_time, end_time, state))

        fig, axs = plt.subplots(2, 1, figsize=(15, 10), sharex=True,
                                gridspec_kw={'height_ratios': [1, 3]})

        # 定義不同 state 對應的顏色
        color_map = {"1": "tab:red", "2": "tab:green", "3": "tab:blue"}
        y_positions = {"1": 2, "2": 1, "3": 0}  # 用小範圍, 每種狀態一條
        height = 1

        # 圖表1：簡單的甘特圖繪製 state 發生位置 (x=segment編號)
        for (start_x, end_x, state) in states_data:
            axs[0].broken_barh([(start_x, end_x - start_x)], (y_positions[state], height),
                               facecolors=color_map[state])
        axs[0].set_ylim(0, 3)
        axs[0].set_yticks([0,1,2])
        axs[0].set_yticklabels(["State 3","State 2","State 1"])
        axs[0].set_title("Segments State (Gantt-like)")

        # 新增圖例
        legend_elements = [
            Patch(facecolor=color_map["1"], label='State 1: Signal lost'),
            Patch(facecolor=color_map["2"], label='State 2: Signal shift'),
            Patch(facecolor=color_map["3"], label='State 3: Interference')
        ]
        axs[0].legend(handles=legend_elements, loc='upper right')

        # 圖表2：心率變化圖
        hr_data = sorted(hr_data, key=lambda x: x[0])  # 依 x(段編號) 排序
        xs = [x for (x, hr) in hr_data if hr is not None]
        ys = [hr for (x, hr) in hr_data if hr is not None]

        # 繪製折線圖
        axs[1].plot(xs, ys, marker='o', linestyle='-')
        axs[1].set_xlabel("Segment Index")
        axs[1].set_ylabel("Average Heart Rate (bpm)")
        axs[1].set_title("Heart Rate by Segment")

        # 繪製水平線
        for yval in [60, 80, 100]:
            axs[1].axhline(y=yval, color='black', linestyle='--', alpha=0.5)

        # 設定只顯示 60, 80, 100 這三個刻度
        axs[1].set_yticks([60, 80, 100])
        axs[1].set_yticklabels(['60', '80', '100'])

        # 如果仍需要限制顯示範圍，可以酌情設定 y 軸上下界
        axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
        axs[1].set_ylim(0, 120)

        plt.tight_layout()
        plt.savefig(output_png)
        plt.close(fig)
    except Exception as e:
        logging.error(f"Error plotting results: {str(e)}")
