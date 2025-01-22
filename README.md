# Kardia Ultrasound Analysis WebUI

This project is based on the research paper currently under submission in 2025 by Dr. Jowy Tani, Director of the Sleep Center at Taipei Medical University (TMU) Wanfang Hospital, titled [《Feasibility and Validation of a Cost-Effective Continuous Remote Cardiac Monitoring in Clinical Practice and Home-based Application》](#) and has been developed to design the audio algorithmic framework.

This application is based on the Python Flask framework and analyzes ECG ultrasound signal audio files from the Alivrcor KardiaMobile device to perform average heart rate analysis. The application can be used in clinical or home use to enhance heart rate data evaluation for further benefits.

---

## Features

- **ECG Analysis**:  
   After uploading an audio file in `.wav` format, the system will automatically analyze the ultrasound ECG signal and generate detailed reports and charts.
- **Result Display**:  
   - Time-series line charts for further analysis.
   - Average heart rate (bpm) for every 30-second segment.

---

## Prerequisites

Before using the system, please ensure the following requirements are met:

1. **Audio Files**:  
   - The audio format must be `.wav` and should contain ultrasound signals from the Alivecor KardiaMobile (18.5–19.3 kHz).  
   - It is recommended to record for at least 1 minute to ensure accuracy.

2. **Operating System**:  
   - OS Supports Windows 10 and above.

---

## Installation and Usage

### 1. Download or Clone the Repository

You can obtain the full repository code in the following ways:

- Clone via Git:

   ```bash
   git clone https://github.com/tmusleep2025/Kardia-Ultrasound-Analysis-WebUI.git
   ```

- Or download the ZIP file and extract it to your local device.

---

### 2. Running the Application

This project provides a ready-to-use `app.exe` file to launch the Web UI:

1. Double-click `app.exe` to start the application. The system will automatically launch the Flask server and run in the background.
2. Open a web browser (such as Microsoft Edge or Google Chrome) and enter the following URL, then press `Enter`:

   ```
   http://127.0.0.1:5000
   ```

![System Startup Illustration](https://github.com/tmusleep2025/Kardia-Ultrasound-Analysis-WebUI/blob/main/images/System_Startup_Successful.png)

---

### 3. Performing Audio Analysis in the Browser

Follow these steps to analyze the audio data:

1. **Upload Audio File**  
   - Click "Choose File" to upload the `.wav` file from the Kardia device.  
   - For testing purposes, please [download](https://github.com/tmusleep2025/Kardia-Ultrasound-Analysis-WebUI/blob/main/Flask/audio_files/example_audio_10min.wav) our provided 10-minute test audio file.

2. **Start Analysis**  
   - After uploading the file, click the **"Start New Analysis"** button.  
   - The system will display the processing progress; please wait patiently for the analysis to complete.

3. **View Results**  
   - After the analysis is complete, the system will generate a report containing:  
      - Time-series line charts  
      - A table showing the average heart rate (bpm) for every 30 seconds

---

## Example Usage

Below is an example system workflow diagram to help you quickly understand the operation process:

![System Workflow Illustration](https://github.com/tmusleep2025/Kardia-Ultrasound-Analysis-WebUI/blob/main/images/Page_Description.png)

---

## Additional Notes

- This analysis system currently supports **single-user mode only**; running multiple analyses simultaneously may cause abnormal behavior.
- It is recommended to record audio in a quiet environment to minimize external interference.
- The system currently supports heart rate ranges between 34-120 bpm; analysis cannot be performed outside this range.

---

## License

This project is licensed under the **MIT License**. For details, please refer to the [LICENSE](LICENSE) file.

---

## Contact Us

If you have any questions or suggestions, please contact us via the following methods or submit an issue on GitHub:

- **Email**:  
   - [jowytani@tmu.edu.tw](mailto:jowytani@tmu.edu.tw)  
   - [tmusleep2025@gmail.com](mailto:tmusleep2025@gmail.com)

- **GitHub Issue Feedback**:  
   Submit your issues or suggestions in the [GitHub Issues](https://github.com/tmusleep2025/Kardia-Ultrasound-Analysis-WebUI/issues) section.
