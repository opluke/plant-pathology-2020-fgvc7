# Plant Pathology 2020 – FGVC7 蘋果葉病害分類  
（DL_Team10）

## 目錄
1. [專案簡介](#專案簡介)  
2. [資料集下載](#資料集下載)  
3. [環境需求](#環境需求)  
4. [快速開始](#快速開始)  
5. [專案結構](#專案結構)  
6. [主要腳本說明](#主要腳本說明)  
7. [模型權重](#模型權重)  
8. [比賽成績](#比賽成績)  
9. [參考文獻](#參考文獻)

---

## 專案簡介
本專案參與 **Kaggle Plant Pathology 2020 – FGVC7** 競賽，  
目標為自動辨識蘋果葉的四種狀態：

| 類別 | 說明 |
|------|------|
| healthy | 健康葉片 |
| multiple_diseases | 同時具多種疾病 |
| rust | 銹病 |
| scab | 痂病 |

我們提出 **4 種 CNN 架構 + 5 折交叉驗證 + Test-Time Augmentation (TTA)** 的集成方法，  
在 Public Leaderboard 取得 **X.XXXX** 的 F1-score。

---

## 資料集下載
1. 登入 Kaggle 後前往 <https://www.kaggle.com/c/plant-pathology-2020-fgvc7>  
2. 下載並解壓 `images.zip`、`train.csv`、`test.csv` 至下列資料夾結構：

```
data/
 ├─ images/               ← 原始影像 (jpg)
 │   ├─ Train_0.jpg
 │   └─ ...
 ├─ train.csv
 └─ test.csv
```

---

## 環境需求
- Python ≥ 3.9  
- GPU：NVIDIA RTX 2060 以上（建議）  
- CUDA 11/12  
- 主要套件版本（完整內容請見 `requirements.txt`）：
  ```text
  torch==2.2.0
  torchvision==0.17.0
  pandas==2.2.0
  scikit-learn==1.4.0
  Pillow==10.3.0
  python-docx==1.1.0   # 僅用於自動產生報告，可移除
  ```

### 一鍵安裝
```bash
# 建議使用 virtualenv 或 conda
python -m venv venv
source venv/bin/activate  # Windows 改用 venv\Scripts\activate
pip install -r requirements.txt
```

---

## 快速開始

### 1. 訓練 1 折 (示範)
```bash
python train_pipeline_full_upgraded.py     --fold 0     --data_dir data     --epochs 5     --batch_size 32
```

### 2. 全 5 折 + 推論 & 產生提交檔
```bash
# 5 折訓練（會自動保存 x4 models × 5 folds）
python train_pipeline_full_upgraded.py --train_all

# 產生 submission_ensemble_upgraded.csv
python train_pipeline_full_upgraded.py --inference
```

> **注意**  
> - 指令旗標可在程式開頭自行修改或透過 argparse；  
> - 請確定 `images/` 與 `train.csv`、`test.csv` 目錄名稱與路徑一致。

---

## 專案結構
```
.
├─ train_pipeline_full_upgraded.py   # 主訓練 / 推論腳本
├─ requirements.txt
├─ README.md
├─ convnext_base_fold0.pth
├─ densenet201_fold0.pth
├─ efficientnet_b3_fold0.pth
├─ resnet50_fold0.pth
└─ reports/
   ├─ DL_team_10.docx         # 書面報告
```

---

## 主要腳本說明

| 檔案 | 功能 |
|------|------|
| `train_pipeline_full_upgraded.py` | *核心腳本*：包含資料讀取、模型定義、訓練迴圈、TTA 集成與輸出提交檔。 |
| `requirements.txt`                | Python 套件版本鎖定。 |

---

## 模型權重
 HF: (https://huggingface.co/MeteorAAA/plant-pathology-2020-fgvc7/tree/main)  
---

## 比賽成績
| 方法 | 公開榜分數 |
|------|-----------|
| 單一 ConvNeXt-Base | 0.968 |
| 4 模型 × 5 折集成 | **0.977** |

> 最終提交檔名稱：`submission_ensemble_upgraded.csv`

---

## 參考文獻
1. Mohanty, D. P., Hughes, D., Salathé, M. *Using Deep Learning for Image-Based Plant Disease Detection*, 2016.  
2. Tan, M., Le, Q. *EfficientNet: Rethinking Model Scaling for CNNs*, ICML 2019.  
3. Liu, C. *et al.* *ConvNeXt: Rescaling ConvNets for Training with Large Datasets*, CVPR 2022.  

---

