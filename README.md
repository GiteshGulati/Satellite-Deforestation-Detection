# 🌿 ForestGuard — Satellite Deforestation Detection

> AI-powered deforestation detection from satellite imagery using deep learning.  
> Built with ResNet-50 transfer learning on the EuroSAT dataset.

---

## ✨ Features

- **🛰️ Patch-based change detection** — splits satellite images into patches, classifies each one, and compares two time periods to flag deforestation
- **🤖 ResNet-50 transfer learning** — pre-trained on ImageNet, fine-tuned on 27,000 EuroSAT satellite images across 10 land cover classes
- **🌐 Streamlit web app** — upload before & after images through a browser UI, or run the built-in demo with one click
- **📊 Rich visualizations** — land cover maps, 3-panel change detection map, transition breakdown charts, and a downloadable analysis report
- **⚡ ~94.5% validation accuracy** on the EuroSAT test set

---

## 🖥️ App Preview

| Sidebar & Hero | Analysis Mode | Results |
|---|---|---|
| Model info card, class legend | Demo mode or image upload | Stat cards, maps, charts |

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/GiteshGulati/Satellite-Deforestation-Detection.git
cd Satellite-Deforestation-Detection
```

### 2. Set up a virtual environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the pre-trained model
Download `satellite_classifier.pth` from the **[latest release](https://github.com/GiteshGulati/Satellite-Deforestation-Detection/releases/latest)** and place it at:
```
models/satellite_classifier.pth
```

### 5. (Demo mode only) Download the EuroSAT dataset
The demo mode requires the EuroSAT image dataset.  
Download from: **[Kaggle — EuroSAT Dataset](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset)**

Extract so the structure looks like:
```
data/
└── EuroSAT/
    └── 2750/
        ├── AnnualCrop/
        ├── Forest/
        ├── HerbaceousVegetation/
        ├── Highway/
        ├── Industrial/
        ├── Pasture/
        ├── PermanentCrop/
        ├── Residential/
        ├── River/
        └── SeaLake/
```

### 6. Launch the app
```bash
streamlit run app.py
```
The app opens automatically at **http://localhost:8501**

---

## 📁 Project Structure

```
Satellite-Deforestation-Detection/
├── app.py                        ← Streamlit web app (main entry point)
├── requirements.txt              ← Python dependencies
├── .streamlit/
│   └── config.toml               ← Dark forest theme
├── src/
│   ├── train.py                  ← Train the ResNet-50 classifier
│   ├── evaluate.py               ← Evaluate model (confusion matrix, report)
│   └── detect_deforestation.py  ← Core detection pipeline
├── data/
│   └── EuroSAT/2750/             ← EuroSAT dataset (not in repo, download separately)
├── models/
│   └── satellite_classifier.pth  ← Trained model (not in repo, download from releases)
└── outputs/                      ← Generated visualizations and reports
```

---

## 🧠 How It Works

```
Satellite Image (Before)          Satellite Image (After)
        │                                   │
        ▼                                   ▼
  Split into 64×64 patches          Split into 64×64 patches
        │                                   │
        ▼                                   ▼
  ResNet-50 Classifier              ResNet-50 Classifier
  (10 land cover classes)           (10 land cover classes)
        │                                   │
        ▼                                   ▼
   Land Cover Map (Before)          Land Cover Map (After)
                    │               │
                    ▼               ▼
              Change Detection
        (Forest → AnnualCrop/Residential/
              Industrial/Pasture/PermanentCrop)
                    │
                    ▼
         Deforestation Report + Visualization
```

### Land Cover Classes

| Class | Deforestation Target? |
|---|:---:|
| 🌲 Forest | — |
| 🌾 AnnualCrop | ✅ |
| 🌿 HerbaceousVegetation | — |
| 🏘️ Residential | ✅ |
| 🏭 Industrial | ✅ |
| 🐄 Pasture | ✅ |
| 🍇 PermanentCrop | ✅ |
| 🛣️ Highway | — |
| 🌊 River | — |
| 🌊 SeaLake | — |

---

## 🏋️ Training Your Own Model

If you want to retrain instead of using the pre-trained weights:

```bash
# Step 1 — Train (requires EuroSAT dataset)
python src/train.py

# Step 2 — Evaluate
python src/evaluate.py

# Step 3 — Launch app
streamlit run app.py
```

**Training config:**
- Architecture: ResNet-50 (frozen backbone, trained FC head only)
- Dataset split: 80% train / 10% val / 10% test
- Optimizer: Adam (lr = 0.001)
- Epochs: 10, Batch size: 32
- GPU recommended (RTX 3050: ~10–15 min | CPU: ~1–2 hrs)

---

## 🖼️ Using the App

### Demo Mode
Click **"▶ Run Demo Analysis"** — no satellite data needed. The app builds two synthetic scenes from EuroSAT images (a forested "before" and a partially cleared "after") and runs the full pipeline.

### Upload Mode
1. Switch to **"📁 Upload My Own Images"**
2. Upload a **before** image and an **after** image of the same area from different dates
3. Choose patch size (32 / 64 / 128 px)
4. Click **"▶ Run Analysis"**
5. View results in the **📊 Results** tab and download the report

**Supported formats:** PNG, JPG, JPEG, TIF, TIFF  
**Free satellite imagery sources:**
- [Copernicus Open Access Hub](https://scihub.copernicus.eu) (Sentinel-2, free with account)
- [Google Earth Engine](https://earthengine.google.com) (free with Google account)

---

## 📦 Requirements

| Package | Purpose |
|---|---|
| `torch` / `torchvision` | Model training & inference |
| `streamlit` | Web application UI |
| `pillow` | Image processing |
| `numpy` | Array operations |
| `matplotlib` | Visualizations |
| `scikit-learn` | Evaluation metrics |
| `seaborn` | Confusion matrix heatmap |
| `pandas` | Data table display |
| `tqdm` | Progress bars |

---

## 📄 License

This project is for educational purposes. The EuroSAT dataset is licensed under [MIT](https://github.com/phelber/EuroSAT).

---

<div align="center">
  Built with 🌿 using PyTorch, EuroSAT, and Streamlit
</div>
