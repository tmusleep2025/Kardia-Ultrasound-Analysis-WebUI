import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from flask import Flask, request, render_template, jsonify, url_for, make_response
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
from ECGAnalyzer import ECGAnalyzer  # Ensure ECGAnalyzer is available in your PYTHONPATH

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'wav'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global variables to store the ECGAnalyzer instance and latest hrv_stats_df
ecg_analyzer = None
last_hrv_stats_df = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    """Display the file upload page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Upload the .wav file, perform ECG analysis, and generate the report.
    
    The process_audio() method returns an hrv_stats_df DataFrame with columns:
      - "Start Time (s)"
      - "30s_End Time (s)"
      - "Avg HR (BPM)"
      - "SDNN (ms)"
      - "RMSSD (ms)"
      - "SD1 (ms)"
      - "SD2 (ms)"
      - "Poincare S (ms^2)"
      - "SD1/SD2"
      - "R Peak total"
    """
    global ecg_analyzer, last_hrv_stats_df

    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Create an instance of ECGAnalyzer and process the audio file.
        ecg_analyzer = ECGAnalyzer(filepath)
        hrv_stats_df = ecg_analyzer.process_audio()

        # Store the DataFrame globally for CSV export
        last_hrv_stats_df = hrv_stats_df.copy()

        # -------------------------------
        # Compute Overall Summary Metrics
        # -------------------------------
        # Each segment is 30 seconds → convert to minutes.
        segment_duration_min = 30 / 60.0
        total_segments = len(hrv_stats_df)
        total_analysis_time_min = total_segments * segment_duration_min

        # Convert Avg HR to numeric and filter valid segments.
        hrv_stats_df["Avg HR (BPM)_num"] = pd.to_numeric(hrv_stats_df["Avg HR (BPM)"], errors='coerce')
        analyzable_segments = hrv_stats_df[hrv_stats_df["Avg HR (BPM)_num"].notnull()]
        analyzable_count = len(analyzable_segments)
        analyzable_time_min = analyzable_count * segment_duration_min
        unanalyzable_time_min = total_analysis_time_min - analyzable_time_min
        analysis_percent = (analyzable_time_min / total_analysis_time_min * 100) if total_analysis_time_min > 0 else 0
        unanalyzable_percent = 100 - analysis_percent

        # Compute Total Heart Variation using RMSSD (ms) average over valid segments.
        analyzable_segments["RMSSD (ms)_num"] = pd.to_numeric(analyzable_segments["RMSSD (ms)"], errors='coerce')
        if analyzable_count > 0:
            total_heart_variation = analyzable_segments["RMSSD (ms)_num"].mean()
        else:
            total_heart_variation = np.nan

        summary = {
            "total_analysis_time_min": round(total_analysis_time_min, 2),
            "analysis_time_min": round(analyzable_time_min, 2),
            "analysis_percent": round(analysis_percent, 2),
            "unanalyzable_time_min": round(unanalyzable_time_min, 2),
            "unanalyzable_percent": round(unanalyzable_percent, 2),
            "total_heart_variation": round(total_heart_variation, 2) if not np.isnan(total_heart_variation) else "N/A"
        }

        # -------------------------------
        # Generate Combined Chart: Heart Rate (Left Y-axis) & RMSSD (Right Y-axis)
        # -------------------------------
        avg_hr_data = []
        rmssd_data = []
        for i, row in hrv_stats_df.iterrows():
            segment_index = i + 1
            avg_hr_value = row.get("Avg HR (BPM)_num", None)
            rmssd_value = row.get("RMSSD (ms)", None)
            
            # 轉換為數字（若無法轉換則為 NaN）
            avg_hr_value = pd.to_numeric(avg_hr_value, errors='coerce')
            rmssd_value = pd.to_numeric(rmssd_value, errors='coerce')
            
            if pd.notnull(avg_hr_value):
                avg_hr_data.append((segment_index, avg_hr_value))
            if pd.notnull(rmssd_value):
                rmssd_data.append((segment_index, rmssd_value))

        chart_dir = os.path.join('static', 'images')
        if not os.path.exists(chart_dir):
            os.makedirs(chart_dir)
        output_png = os.path.join(chart_dir, 'chart.png')

        # 檢查是否有任何有效資料
        if avg_hr_data or rmssd_data:
            fig, ax1 = plt.subplots(figsize=(8, 4))
            
            # Plot: Average HR on ax1 (Left Y-axis, 藍線)
            if avg_hr_data:
                xs_avg = [x for (x, hr) in avg_hr_data]
                ys_avg = [hr for (x, hr) in avg_hr_data]
                ax1.plot(xs_avg, ys_avg, marker='o', color='blue', label='Heart Rate (bpm)')
                ax1.set_ylabel("Heart Rate (bpm)", color='blue')
                ax1.tick_params(axis='y', labelcolor='blue')
                ax1.set_ylim(0, 120)  # 依需求調整 HR 顯示範圍
                # 在左軸上繪製參考線 (60, 80, 100 bpm)
                for yval in [60, 80, 100]:
                    ax1.axhline(y=yval, color='black', linestyle='--', alpha=0.5)
            else:
                ax1.set_ylabel("Heart Rate (bpm)")
            
            ax1.set_xlabel("Segment (each 30 secs)")
            ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
            
            # Plot: RMSSD on ax2 (Right Y-axis, 紅色虛線)
            ax2 = ax1.twinx()
            if rmssd_data:
                xs_rmssd = [x for (x, val) in rmssd_data]
                ys_rmssd = [val for (x, val) in rmssd_data]
                ax2.plot(xs_rmssd, ys_rmssd, marker='o', color='red', linestyle='--', label='HRV (ms)')
                ax2.set_ylabel("RMSSD (ms)", color='red')
                ax2.tick_params(axis='y', labelcolor='red')
                # 視需求設定 HRV 的上下界，例如 ax2.set_ylim(0, 120) 或自動由 matplotlib 決定
                ax2.set_ylim(0, max(ys_rmssd) + 50)  # 若想自動留一點空間
            else:
                ax2.set_ylabel("RMSSD (ms)", color='red')
            
            # 加上圖表標題
            plt.title("Heart Rate and HRV Analysis")
            
            # 組合兩個軸的圖例
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', ncol=2)
            
            plt.tight_layout()
            plt.savefig(output_png)
            plt.close(fig)
        else:
            # 若沒有任何有效數據，就輸出一個空圖表
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, 'No valid HR or RMSSD data available', 
                    horizontalalignment='center', verticalalignment='center')
            plt.tight_layout()
            plt.savefig(output_png)
            plt.close(fig)
            
        # -------------------------------
        # Prepare Segment Analysis Results Table
        # -------------------------------
        segment_results = []
        for i, row in hrv_stats_df.iterrows():
            segment_results.append({
                "segment_index": i + 1,
                "segment_start": row.get("Start Time (s)", "N/A"),
                "segment_end": row.get("30s_End Time (s)", "N/A"),
                "avg_hr": row.get("Avg HR (BPM)", "N/A"),
                "hrv": row.get("RMSSD (ms)", "N/A")
            })

        # Render the report page with computed data.
        return render_template('report.html',
                               summary=summary,
                               chart_url=url_for('static', filename='images/chart.png'),
                               segment_results=segment_results)
    else:
        return "File type not allowed. Please upload a .wav file.", 400

@app.route('/export_csv', methods=['GET'])
def export_csv():
    """
    Export the hrv_stats_df data as a CSV file.
    """
    global last_hrv_stats_df
    if last_hrv_stats_df is None:
        return "No data available for export.", 400
    
    csv_data = last_hrv_stats_df.to_csv(index=False)
    response = make_response(csv_data)
    response.headers["Content-Disposition"] = "attachment; filename=Results.csv"
    response.headers["Content-type"] = "text/csv"
    return response

@app.route('/progress', methods=['GET'])
def get_progress():
    """Return current analysis progress (for front-end polling)."""
    global ecg_analyzer
    if ecg_analyzer is None:
        return jsonify({
            "current_chunk": 0,
            "total_chunks": 0
        })
    return jsonify({
        "current_chunk": ecg_analyzer.current_chunk,
        "total_chunks": ecg_analyzer.total_chunks
    })

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
