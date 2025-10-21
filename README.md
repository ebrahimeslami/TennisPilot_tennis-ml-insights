# 🎾 Tennis ML Insights
Comprehensive tennis analytics and machine learning framework to quantify player performance, match dynamics, and tour-level insights.

## 🔍 Overview
This project combines ATP & WTA match data (1991–2025) to develop novel indices describing player performance under different conditions.  
Each index is ML-driven and interpretable, offering insight into:
- **Performance resilience** (Clutch Pressure Index)
- **Surface adaptability** (Surface Transition Index)
- **Momentum sustainability**
- **Recovery, fatigue, and scheduling efficiency**

## ⚙️ Features
- Unified **meta dataset builder** from 1968–2025
- Modular **index framework** (each index = independent script)
- ML-ready tensors for PyTorch models
- Era-based splits for historical and modern comparisons

## 🧠 ML Stack
- `PyTorch` for modeling
- `pandas` + `pyarrow` for data engineering
- `scikit-learn` for validation
- `matplotlib` for analytics visualization

## 🗂️ Folder Structure
See [`docs/project_overview.md`](docs/project_overview.md) for the full breakdown.

## 🚀 Getting Started
```bash
conda env create -f environment.yml
conda activate tennisml
python scripts/build_master.py --raw_root "data/TML-Database-master" --out_root "data" --start_year 1991
python scripts/build_meta.py --master "data/master/tennis_master_1991.parquet" --out_root "data"
