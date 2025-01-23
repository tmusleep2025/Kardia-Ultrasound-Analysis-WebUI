# flask/app.py
# -*- coding: utf-8 -*-
import os
import shutil
import threading
import time
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from alivecor_package import ECGAnalyzer  # Assuming you have ECG analysis logic
from split_utils import split_audio, save_results_to_csv, plot_results_from_csv 
import numpy as np

print("=== Set environment variables ===")

app = Flask(__name__)
app.secret_key = 'some_secret_key'  # Please replace with a secure key

# Set upload limit (e.g., maximum 16MB)
app.config["MAX_CONTENT_LENGTH"] = 4096 * 1024 * 1024  # 16MB

ALLOWED_EXTENSIONS = {'wav'}

# Global variable to store the current analysis status (single user/single task)
analysis_status = {
    'status': 'idle',      # Text description: idle / preparing / analyzing segment n / creating CSV / plotting charts / completed / failed
    'progress': 0,         # Number of completed steps
    'total_steps': 1,      # Total number of steps
    'done': False,         # Whether the analysis is completed
    'error': None,         # Error message (if any)
    'results': None        # Analysis results: includes summary and segments
}

def allowed_file(filename):
    """Check if the file has an allowed format"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """
    Home Page:
      - If analysis is in progress, display a message and disable upload
      - If analysis is completed (or not started), allow choosing [Start New Analysis] or [View Last Analysis Results] (if results exist)
    """
    in_progress = (analysis_status['status'] != 'idle' and not analysis_status['done'])
    has_old_result = analysis_status['done'] and (analysis_status['results'] is not None)
    return render_template('index.html',
                           analysis_in_progress=in_progress,
                           has_old_result=has_old_result)

@app.route('/upload', methods=['POST'])
def upload():
    """
    Receive uploaded file and start a new analysis process (delete old files and reset status)
    """
    # If analysis is still in progress, do not allow another upload
    if (analysis_status['status'] != 'idle') and (not analysis_status['done']):
        flash("The system is still processing the previous audio file. Please wait for the analysis to complete before uploading again.")
        return redirect(url_for('index'))

    # Check if the file part is present
    if 'file' not in request.files:
        flash("No file selected.")
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        flash("No file selected.")
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        # Remove previous results (if any)
        uploads_dir = os.path.join(app.root_path, 'uploads')
        if os.path.exists(uploads_dir):
            shutil.rmtree(uploads_dir)
        os.makedirs(uploads_dir, exist_ok=True)

        # Reset analysis_status
        analysis_status.update({
            'status': 'Preparing...',
            'progress': 0,
            'total_steps': 1,  # Initial number of steps (to be updated in background task)
            'done': False,
            'error': None,
            'results': None
        })

        # Save the uploaded file
        filename = secure_filename(file.filename)
        upload_path = os.path.join(uploads_dir, filename)
        file.save(upload_path)

        # Start background analysis
        thread = threading.Thread(target=background_analysis, args=(upload_path,))
        thread.start()

        return redirect(url_for('progress_page'))
    else:
        flash("Only .wav file format is allowed.")
        return redirect(url_for('index'))

@app.route('/progress')
def progress_page():
    """
    Display the analysis progress page
    """
    return render_template('progress.html')

@app.route('/progress_status')
def progress_status():
    """
    API endpoint for the frontend to poll and get analysis progress
    """
    return jsonify({
        'status': analysis_status['status'],
        'progress': analysis_status['progress'],
        'total_steps': analysis_status['total_steps'],
        'done': analysis_status['done'],
        'error': analysis_status['error']
    })

@app.route('/result')
def result_page():
    """
    Display results after analysis is complete (including tables and generated charts)
    """
    # If not completed, redirect to progress page
    if not analysis_status['done']:
        return redirect(url_for('progress_page'))

    # If there was an error, redirect to home page and display error
    if analysis_status['error']:
        flash(f"An error occurred during analysis: {analysis_status['error']}")
        return redirect(url_for('index'))

    # Get results
    if not analysis_status['results']:
        flash("No analysis results available.")
        return redirect(url_for('index'))

    results = analysis_status['results']
    summary = results['summary']
    segments = results['segments']
    filename = results['filename']

    # Generated report image (report.png) is already created in background_analysis => uploads/report.png
    report_png_path = os.path.join(app.root_path, 'uploads', 'report.png')
    report_exists = os.path.isfile(report_png_path)

    return render_template('result.html',
                           summary=summary,
                           segment_results=segments,
                           filename=filename,
                           report_exists=report_exists)

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    """
    Provide access to uploaded and analyzed temporary files for download or display
    """
    uploads_dir = os.path.join(app.root_path, 'uploads')
    return send_from_directory(directory=uploads_dir, path=filename)


def background_analysis(upload_path):
    """
    Background Task:
      1) Split audio file
      2) Analyze each segment
      3) Generate CSV
      4) Plot charts based on CSV
    """
    try:
        analysis_status['status'] = "Splitting audio file..."
        time.sleep(0.5)  # Simulate processing time
        min_duration = 30
        segments = split_audio(upload_path, segment_duration=30)
        if not segments:
            raise ValueError("Unable to split audio file or the file is empty.")
        # 過濾片段
        segments = [
            (data, sr) for data, sr in segments
            if len(data) / sr >= min_duration
        ]
        if not segments:
            raise ValueError("No audio segments meet the minimum duration requirement.")

        # Total steps: splitting (1 step) + analyzing each segment (N steps) + creating CSV (1 step) + plotting (1 step)
        total_segments = len(segments)
        analysis_status['total_steps'] = 2 + total_segments  # +1 for CSV, +1 for plotting
        analysis_status['progress'] = 1  # Splitting completed

        results_per_segment = []
        heart_rates = []
        total_analyzable_secs = 0.0
        total_unanalyzable_secs = 0.0
        seg_duration = 30

        # Analyze each segment
        for idx, (segment_data, sample_rate) in enumerate(segments):
            analysis_status['status'] = f"Analyzing segment {idx+1}/{total_segments}..."
            # Write temporary segment file
            from scipy.io import wavfile
            segment_filename = f"segment_{idx+1}.wav"
            segment_path = os.path.join(app.root_path, 'uploads', segment_filename)
            wavfile.write(segment_path, sample_rate, segment_data)

            # Analyze
            analyzer = ECGAnalyzer(segment_path, refractory_period=0.5)
            ecg_stats = analyzer.analyze_ecg()

            avg_hr = ecg_stats.get("Average Heart Rate (bpm)", "N/A")
            hrv = ecg_stats.get("RMSSD (s)", "N/A")
            state = ecg_stats.get("state", None)
            if state in ['1', 1, '2', 2, '3', 3]:
                total_unanalyzable_secs += seg_duration
            else:
                total_analyzable_secs += seg_duration

            # Collect heart rates
            try:
                hr_float = float(avg_hr) if avg_hr != "N/A" else None
                if hr_float:
                    heart_rates.append(hr_float)
            except:
                pass

            results_per_segment.append({
                'segment_index': idx+1,
                'segment_start': idx * seg_duration,
                'segment_end': idx * seg_duration + seg_duration,
                'avg_hr': avg_hr,
                'rmssd': hrv,
                'state': state if state else 'OK'
            })

            # Update progress
            analysis_status['progress'] += 1

        # Calculate statistics
        total_length_sec = total_segments * seg_duration
        if total_length_sec > 0:
            analysis_percent = (total_analyzable_secs / total_length_sec) * 100
            unanalysis_percent = (total_unanalyzable_secs / total_length_sec) * 100
        else:
            analysis_percent = 0
            unanalysis_percent = 0

        # Heart rate standard deviation => total heart variation
        if len(heart_rates) > 1:
            total_heart_variation = round(np.std(heart_rates), 2)
        else:
            total_heart_variation = "N/A"

        # Prepare summary (remove start/stop times)
        summary = {
            "total_analysis_time_min": round(total_length_sec / 60.0, 2),
            "analysis_time_min": round(total_analyzable_secs / 60.0, 2),
            "analysis_percent": round(analysis_percent, 2),
            "unanalyzable_time_min": round(total_unanalyzable_secs / 60.0, 2),
            "unanalyzable_percent": round(unanalysis_percent, 2),
            "total_heart_variation": total_heart_variation
        }
        analysis_status['status'] = "Creating CSV report..."
        # Generate CSV => uploads/ecg_results.csv
        csv_path = os.path.join(app.root_path, 'uploads', 'ecg_results.csv')
        save_results_to_csv(csv_path, results_per_segment, os.path.basename(upload_path))
        analysis_status['progress'] += 1

        # Plot charts
        analysis_status['status'] = "Plotting analysis charts..."
        report_png_path = os.path.join(app.root_path, 'uploads', 'report.png')
        plot_results_from_csv(csv_filename=csv_path, segment_duration=30, output_png=report_png_path)
        analysis_status['progress'] += 1

        # Final record
        analysis_status['results'] = {
            'filename': os.path.basename(upload_path),
            'summary': summary,
            'segments': results_per_segment
        }
        analysis_status['status'] = "Analysis Completed"
        analysis_status['done'] = True

    except Exception as e:
        analysis_status['error'] = str(e)
        analysis_status['done'] = True
        analysis_status['status'] = "Analysis Failed"


print("=== Starting Flask application ===")
if __name__ == "__main__":
    app.run(debug=True, port=5000)
