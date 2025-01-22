# Kardia 超音波分析 WebUI

本專案基於台北醫學大學(TMU)萬芳醫院的睡眠中心主任陳兆煒(Jowy Tani)醫師在2025年正在投稿中的研究文獻[《Feasibility and Validation of a Cost-Effective Continuous Remote Cardiac Monitoring in Clinical Practice and Home-based Application》](#) 中，展示開發的音訊演算法架構

本應用端為基於Python Flask架構，執行分析具有Alivrcor KardiaMobile之ECG超音波訊號音檔，進行平均心律分析之方法應用。應用端可在臨床或是居家檢測時，增加心率數據評估後續其他效益。

---

## 功能特色

- **ECG 分析**：  
  上傳 `.wav` 格式的音頻文件後，系統將自動分析超音波 ECG 信號，並生成詳細的報告和圖表。
- **結果顯示**：   
  - 時間序列折線圖以便進一步分析。
  - 每30秒區段的平均心率（bpm） 

---

## 先決條件

在開始使用本系統之前，請確保準備以下條件：

1. **音頻文件**：  
   - 音頻格式必須為 `.wav`，並包含 Alivecor KardiaMobile 設備的超音波信號（18.5–19.3 kHz）。  
   - 建議至少錄製 1 分鐘，以確保結果的準確性。
   
2. **作業系統**：  
   - 支援 Windows 10 及以上版本。

---

## 安裝與使用

### 1. 下載或複製存儲庫

請按照以下方式獲取存儲庫的完整代碼：

- 使用 Git 進行複製儲存庫：

  ```bash
  git clone https://github.com/tmusleep2025/Kardia-Ultrasound-Analysis-WebUI.git
  ```

- 或者下載 ZIP 壓縮檔並解壓縮至本地設備。

---

### 2. 執行應用程式

本專案提供即開即用的 `app.exe` 文件來啟動 Web UI：

1. 雙擊 `app.exe` 啟動應用程式，系統將自動啟動 Flask 伺服器並在後台運行。
2. 在瀏覽器（如 Microsoft Edge 或 Google Chrome）中，輸入以下 URL 並按 `Enter` 鍵：

   ```
   http://127.0.0.1:5000
   ```

![系統啟動示意圖](https://github.com/tmusleep2025/Kardia-Ultrasound-Analysis-WebUI/blob/main/images/System_Startup_Successful.png)

---

### 3. 使用瀏覽器進行音頻分析

請按照以下步驟完成音頻數據分析：

1. **上傳音頻文件**  
   - 點擊 "Choose File" 上傳 Kardia 設備的 `.wav` 文件。  
   - 若需測試，請 [下載](https://github.com/tmusleep2025/Kardia-Ultrasound-Analysis-WebUI/blob/main/Flask/audio_files/example_audio_10min.wav) 我們提供的 10 分鐘測試音檔。

2. **開始分析**  
   - 上傳文件後，點擊 **"Start New Analysis"** 按鈕。  
   - 系統將顯示處理進度，請耐心等待分析完成。

3. **查看結果**  
   - 分析完成後，系統將產生報告，提供以下內容：  
     - 時間序列折線圖  
     - 每 30 秒的平均心率（bpm）表格

---

## 使用範例

以下是系統的範例流程圖，幫助您快速理解操作流程：

![系統流程示意圖](https://github.com/tmusleep2025/Kardia-Ultrasound-Analysis-WebUI/blob/main/images/Page_Description.png)

---

## 其他注意事項

- 本分析系統目前僅支援**單用戶模式**，同時執行多個分析可能會導致異常行為。
- 建議在安靜環境下錄製音頻，以減少外部干擾。
- 本分析系統目前僅支援心律範圍在 34-120 bpm之間，如超過此範圍本系統無法正常分析。

---

## 授權

此專案依據 **MIT 許可證** 授權，詳情請參閱 [LICENSE](LICENSE) 文件。

---

## 聯絡我們

若您有任何問題或建議，請通過以下方式與我們聯繫，或在 GitHub 上提交問題：

- **電子郵件**：
  - [jowytani@tmu.edu.tw](mailto:jowytani@tmu.edu.tw)
  - [tmusleep2025@gmail.com](mailto:tmusleep2025@gmail.com)

- **GitHub 問題反饋**：  
  在 [GitHub Issues](https://github.com/tmusleep2025/Kardia-Ultrasound-Analysis-WebUI/issues) 中提交您的問題或建議。

 
