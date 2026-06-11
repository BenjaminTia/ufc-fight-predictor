---
language: python
license: mit
tags:
- ufc
- mma
- fight-prediction
- machine-learning
- pytorch
- xgboost
- lightgbm
- gpu
- sports-analytics
- ensemble
datasets:
- wikipedia
- ufcstats
library_name: transformers
pipeline_tag: tabular-classification
---

<div align="center">

# UFC Fight Predictor

**A GPU-accelerated ensemble ML system that predicts UFC fight outcomes using quantitative stats, NLP news sentiment, and weighted expert consensus.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-ee4c2c)](https://pytorch.org)
[![XGBoost GPU](https://img.shields.io/badge/XGBoost-GPU-green)](https://xgboost.readthedocs.io)
[![LightGBM GPU](https://img.shields.io/badge/LightGBM-GPU-brightgreen)](https://lightgbm.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-Interpretable-ff69b4)](https://shap.readthedocs.io)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-yellow?logo=huggingface)](https://huggingface.co/benjamintia/ufc-fight-predictor)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

</div>

---

## How This System Works

The model combines three distinct signal sources into a stacked ensemble:

### 1. Data Sources

```
┌─────────────────────────────┐     ┌──────────────────────────────┐
│  UFC Event Data             │     │  Fighter Profiles            │
│  (Wikipedia / UFCStats)     │     │  (70+ fighters with career   │
│  ─ Events & matchups        │     │   stats from Sherdog / UFC)  │
│  ─ Real fight outcomes      │     │  ─ SLPM, SAPM, TD avg       │
│  ─ Method & round info      │     │  ─ Strike accuracy/defense   │
└────────────┬────────────────┘     │  ─ TD accuracy/defense       │
             │                      │  ─ Submissions, reach, etc.  │
             ▼                      └────────────┬─────────────────┘
┌─────────────────────────────┐                  │
│  Expert Picks               │                  │
│  (Tapology / Synthetic)     │                  │
│  ─ 10 experts, 1000+ picks  │                  │
│  ─ Reliability-weighted     │                  │
│    consensus scores         │                  │
└────────────┬────────────────┘                  │
             │                                   │
┌────────────▼────────────────┐                  │
│  NLP Sentiment Analysis     │                  │
│  (HuggingFace Transformers) │                  │
│  ─ twitter-roberta-base     │                  │
│  ─ GPU-accelerated inference│                  │
│  ─ 78 fighters scored       │                  │
│  ─ Momentum score computed  │                  │
└────────────┬────────────────┘                  │
             │                                   │
             └───────────┬───────────────────────┘
                         │
                         ▼
            ┌──────────────────────┐
            │  Feature Engineering │
            │  ─ 52 features       │
            │  ─ Style matchups    │
            │  ─ Round-by-round    │
            │  ─ Sentiment diff    │
            │  ─ Expert consensus  │
            └──────────┬───────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │  Stacked Ensemble    │
            │                      │
            │  ┌────────────────┐  │
            │  │   XGBoost      │  │ (GPU: tree_method='hist', device='cuda')
            │  └───────┬────────┘  │
            │  ┌────────────────┐  │
            │  │  LightGBM      │  │ (GPU: device_type='gpu')
            │  └───────┬────────┘  │
            │  ┌────────────────┐  │
            │  │ PyTorch NN     │  │ (GPU: .to('cuda'))
            │  │ 3 hidden layers│  │
            │  │ BatchNorm+Drop │  │
            │  └───────┬────────┘  │
            │          │           │
            │  ┌───────▼────────┐ │
            │  │LogisticReg     │ │ ← Meta-learner
            │  │CalibratedProbs │ │
            │  └────────────────┘ │
            └──────────────────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │   Fight Outcome!     │
            │  Islam Makhachev     │
            │  56.5% win prob      │
            └──────────────────────┘
```

### 2. Three Signal Pillars

| Pillar | What | How |
|--------|------|-----|
| **Quantitative** | Career striking volume (SLPM), striking accuracy, takedown avg/defense, submission avg, reach/height differences | Scraped from UFCStats / Sherdog; round-by-round stamina dropoff computed from fatigue curves |
| **Qualitative (NLP)** | Recent news sentiment for each fighter | `cardiffnlp/twitter-roberta-base-sentiment-latest` on HuggingFace Transformers, GPU-accelerated. Momentum score = avg_sentiment * 0.4 + article_volume * 0.2 + (1 - volatility) * 0.4 |
| **Expert Consensus** | Aggregated Tapology/Sherdog picks weighted by historical accuracy | Reliability weight = accuracy * pick_volume_normalized * confidence_normalized |

### 3. Feature Engineering (52 features)

Key feature categories:
- **Style Matchup:** `diff_sig_str`, `ratio_td`, `diff_ctrl`, `ratio_td_def` — how each fighter's strengths match up against the other's weaknesses
- **Stamina/Endurance:** Round-by-round dropoff in sig strikes and takedown attempts (fatigue modeling)
- **Physical:** Height/reach differentials, stance matchup
- **Experience:** Number of UFC fights, win rate
- **Sentiment:** `a_momentum - b_momentum`, `sentiment_diff`
- **Expert Consensus:** `expert_consensus_a`, `consensus_diff`, `expert_agreement`

### 4. Model Architecture (Stacked Ensemble)

**Base Learners** (all GPU-accelerated):
1. **XGBoost** — `tree_method='hist'`, `device='cuda'`, max_depth=6, 500 trees
2. **LightGBM** — `device_type='gpu'`, num_leaves=63, 500 trees
3. **PyTorch Neural Network** — `.to('cuda')`, 3 hidden layers [256→128→64], BatchNorm1d, ReLU, Dropout(0.35), early stopping

**Meta-Learner:**
- Logistic Regression trained on: [xgb_proba, lgb_proba, nn_proba, avg_proba, max_proba, min_proba, disagreement]
- Outputs calibrated probabilities via Platt scaling

**Regularization:** L2 penalty, learning rate scheduling, early stopping, class-weight balancing, gradient clipping

### 5. Evaluation

- **Chronological split** (80/20) — no data leakage from future fights
- **Metrics:** Accuracy, Log Loss, ROC-AUC, Brier Score
- **SHAP:** TreeExplainer for XGBoost/LightGBM → feature importance plots saved to `plots/`

---

## Installation

### Windows CUDA 12.1 Setup

```powershell
# Prerequisites
# 1. Install NVIDIA CUDA Toolkit 12.1 from https://developer.nvidia.com/cuda-12-1-0-download-archive
# 2. Install cuDNN 8.9 for CUDA 12.x
# 3. Install Miniconda from https://docs.conda.io/en/latest/miniconda.html

# Verify CUDA
nvidia-smi
nvcc --version

# Create environment
conda create -n ufc_predictor python=3.10 -y
conda activate ufc_predictor

# PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# Install everything
pip install -r requirements.txt

# Verify environment
python scripts/check_environment.py
```

### Linux / macOS (CPU only)

```bash
conda create -n ufc_predictor python=3.10 -y
conda activate ufc_predictor
pip install -r requirements.txt
```

---

## How To Train The Model Yourself

### Full Pipeline (step by step)

```powershell
conda activate ufc_predictor
cd ufc-fight-predictor

# ── Step 1: Scrape Data ──────────────────────────────────
# Scrapes Wikipedia for real UFC events + generates realistic round stats
# based on 70+ known fighter career profiles
python scripts/scrape_ufcstats.py --limit-events 100

# Generates synthetic expert picks with reliability weights
python scripts/scrape_expert_predictions.py

# Downloads HuggingFace model + generates news sentiment scores
python scripts/scrape_news_sentiment.py

# ── Step 2: Feature Engineering ───────────────────────────
# Combines all CSV files → 52 engineered features → training_data.csv
python scripts/feature_engineering.py

# ── Step 3: Train Ensemble ────────────────────────────────
# Trains XGBoost + LightGBM + NN → LogisticRegression meta
# Saves models to /models/, plots to /plots/
python scripts/model_training.py
```

### Outputs After Training

```
models/
  ├── xgb_model.json       # XGBoost base learner
  ├── lgb_model.txt        # LightGBM base learner
  ├── nn_model.pt          # PyTorch neural network
  ├── meta_learner.pkl     # Logistic regression meta-learner
  ├── scaler.pkl           # StandardScaler (fitted)
  └── feature_names.pkl    # Feature column order

plots/
  ├── roc_curve.png        # ROC curves for all 4 models
  ├── confusion_matrix.png  # Ensemble confusion matrix
  ├── shap_xgb_summary.png  # XGBoost SHAP feature importance
  ├── shap_xgb_bar.png      # XGBoost top-20 features (bar)
  └── shap_lgb_summary.png  # LightGBM SHAP summary
```

### Predicting a Fight

```powershell
# Basic prediction
python scripts/predict_fight.py -a "Islam Makhachev" -b "Charles Oliveira"

# JSON output
python scripts/predict_fight.py -a "Jon Jones" -b "Tom Aspinall" --json

# See all available fighters
python scripts/predict_fight.py --list-fighters
```

Example output:
```
============================================================
  UFC FIGHT PREDICTION
============================================================

  Islam Makhachev vs Charles Oliveira
  ----------------------------------------

  Islam Makhachev        #################### 56.5%
  Charles Oliveira       ################ 43.5%

  Predicted Winner:      Islam Makhachev
  Confidence:            12.9%

  Individual Model Predictions (Fighter A win probability):
  XGBoost:               32.4%
  LightGBM:              55.3%
  Neural Net:            96.7%

  Model Agreement:       35.7% (Low - models disagree)
============================================================
```

### Re-scraping (on-demand)

```powershell
# Force re-scrape everything
python scripts/scrape_ufcstats.py --refresh --limit-events 200
python scripts/scrape_expert_predictions.py --refresh
python scripts/scrape_news_sentiment.py --refresh

# Then rerun the pipeline
python scripts/feature_engineering.py
python scripts/model_training.py
```

### Making the Model Better

The model's performance scales with data quality. Here's how to improve it:

| Improvement | What To Do |
|-------------|-----------|
| **More fights** | Increase `--limit-events 500` in `scrape_ufcstats.py` |
| **Real round-by-round stats** | The scraper currently generates stats from career averages. To get real stats, bypass UFCStats.com Cloudflare (try Selenium + undetected-chromedriver) and update `scrape_ufcstats.py` to use the actual `scrape_fight_details()` function. |
| **Real expert picks** | Tapology blocks automation. Try using Selenium with user login cookies, or scrape manually and save to CSV. |
| **Real news scraping** | MMA news sites block bots. Try using `newspaper3k` library or a news API. |
| **Hyperparameter tuning** | Modify `model_training.py` — adjust `max_depth`, `learning_rate`, `NN_HIDDEN_LAYERS`, etc. |

---

## Project Structure

```
ufc-fight-predictor/
├── data/                           # CSV data files (gitignored)
│   ├── ufc_fight_stats.csv         # Fight records + round-by-round stats
│   ├── fighter_profiles.csv        # 70+ real fighter career stats
│   ├── expert_picks.csv            # 1000+ expert picks with confidence
│   ├── expert_history.csv          # Expert accuracy history
│   └── fighter_news_sentiment.csv  # NLP sentiment for 78 fighters
│
├── models/                         # Trained model files (gitignored)
│   ├── xgb_model.json              # XGBoost
│   ├── lgb_model.txt               # LightGBM
│   ├── nn_model.pt                 # PyTorch NN
│   ├── meta_learner.pkl            # Logistic regression
│   ├── scaler.pkl                  # StandardScaler
│   └── feature_names.pkl           # Feature names
│
├── plots/                          # Evaluation plots (gitignored)
│   ├── roc_curve.png
│   ├── confusion_matrix.png
│   ├── shap_xgb_summary.png
│   └── shap_xgb_bar.png
│
├── scripts/
│   ├── check_environment.py        # Verify CUDA + dependencies
│   ├── scrape_ufcstats.py          # Multi-source data scraper
│   ├── scrape_expert_predictions.py # Expert picks + reliability weights
│   ├── scrape_news_sentiment.py    # HuggingFace GPU sentiment analysis
│   ├── feature_engineering.py      # → 52 engineered features
│   ├── model_training.py           # Stacked ensemble training
│   ├── inference.py                # Model loading + prediction engine
│   └── predict_fight.py            # CLI prediction interface
│
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## Technical Stack

| Component | Technology | GPU Accelerated |
|-----------|-----------|----------------|
| **Gradient Boosted Trees** | XGBoost 2.1 (`tree_method='hist'`) | ✅ CUDA |
| **Gradient Boosted Trees** | LightGBM 4.5 (`device_type='gpu'`) | ✅ CUDA |
| **Deep Neural Network** | PyTorch 2.3 (`model.to('cuda')`) | ✅ CUDA |
| **Meta-Learner** | scikit-learn LogisticRegression | ❌ CPU |
| **NLP Sentiment** | HuggingFace Transformers (RoBERTa) | ✅ CUDA |
| **Web Scraping** | BeautifulSoup + requests | ❌ CPU |
| **Feature Importance** | SHAP (TreeExplainer, DeepExplainer) | ❌ CPU |
| **Data Manipulation** | Pandas 2.2 + NumPy 1.26 | ❌ CPU |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `CUDA not available` | Reinstall NVIDIA drivers + CUDA Toolkit 12.1. Run `nvidia-smi` to verify. |
| `ImportError: libcublas.so` | Ensure CUDA_PATH is in your system `%PATH%`. |
| XGBoost `device='cuda'` fails | Fall back to `device='cpu'`. Edit `model_training.py` line. |
| LightGBM `device_type='gpu'` fails | On Windows, needs Visual Studio Build Tools. Use `device_type='cpu'` instead. |
| `CUDA out of memory` | Reduce `NN_BATCH_SIZE` (64→32) or `max_depth` (6→4). |
| `403 Forbidden` scraping | Sites block bots. The system falls back to synthetic data automatically. |
| Model predictions are ~50% | Normal with small dataset. Scrape more fights (`--limit-events 500+`). |
| `ModuleNotFoundError` | `pip install -r requirements.txt` — all dependencies listed. |

---

## License

MIT

## Disclaimer

This project is for **educational and research purposes only**. Sports betting is gambling. The predictions are probabilistic estimates, not guarantees. Never bet money you can't afford to lose.
