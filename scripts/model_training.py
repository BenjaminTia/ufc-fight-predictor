"""
Model Training Pipeline
GPU-accelerated stacked ensemble: XGBoost + LightGBM + PyTorch NN -> Logistic Regression meta-learner.
Includes chronological train/test split, evaluation, SHAP interpretation, and model saving.
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, log_loss, roc_auc_score, roc_curve,
    brier_score_loss, classification_report, confusion_matrix,
)

import xgboost as xgb
import lightgbm as lgb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import shap

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"
PLOTS_DIR = Path(__file__).parent.parent / "plots"
TRAINING_CSV = DATA_DIR / "training_data.csv"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.pkl"

MODEL_PATHS = {
    "xgb": MODELS_DIR / "xgb_model.json",
    "lgb": MODELS_DIR / "lgb_model.txt",
    "nn": MODELS_DIR / "nn_model.pt",
    "meta": MODELS_DIR / "meta_learner.pkl",
}

RANDOM_STATE = 42
EARLY_STOPPING_ROUNDS = 50
NN_HIDDEN_LAYERS = [256, 128, 64]
NN_DROPOUT = 0.35
NN_BATCH_SIZE = 64
NN_EPOCHS = 300
NN_LR = 0.001
NN_PATIENCE = 30
TEST_SPLIT_DATE = 0.80

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class UFCFightNet(nn.Module):
    """PyTorch neural network base learner for UFC fight prediction."""

    def __init__(self, input_dim, hidden_layers=None, dropout=0.3):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [256, 128, 64]

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze()


class EarlyStopping:
    def __init__(self, patience=30, min_delta=0.0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model, path):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model, path)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model, path)
            self.counter = 0

    def save_checkpoint(self, model, path):
        if self.verbose:
            print(f"  Validation loss improved. Saving model to {path}")
        torch.save(model.state_dict(), path)


def load_data():
    """Load the preprocessed training data."""
    print("\n" + "=" * 60)
    print("  Loading Training Data")
    print("=" * 60)

    if not TRAINING_CSV.exists():
        print(f"  ERROR: {TRAINING_CSV} not found. Run feature_engineering.py first.")
        raise FileNotFoundError(f"Training data not found at {TRAINING_CSV}")

    df = pd.read_csv(TRAINING_CSV)
    print(f"  Loaded {len(df)} samples with {len(df.columns) - 1} features")

    target_col = "target"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")

    X = df.drop(columns=[target_col])
    y = df[target_col].values

    split_idx = int(len(X) * TEST_SPLIT_DATE)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"  Train: {len(X_train)} samples | Test: {len(X_test)} samples")
    print(f"  Train target distribution: A={y_train.sum():.0f} ({y_train.mean():.2%})")
    print(f"  Test target distribution:  A={y_test.sum():.0f} ({y_test.mean():.2%})")

    return (X_train.values.astype(np.float32), X_test.values.astype(np.float32),
            y_train.astype(np.float32), y_test.astype(np.float32),
            list(X.columns))


def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost with GPU acceleration."""
    print("\n" + "=" * 60)
    print("  Training XGBoost Base Learner (GPU)")
    print("=" * 60)

    try:
        model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            tree_method="hist",
            device="cuda",
            random_state=RANDOM_STATE,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            eval_metric="logloss",
            verbosity=0,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        print(f"  XGBoost train logloss: {log_loss(y_train, model.predict_proba(X_train)[:,1]):.4f}")
        print(f"  XGBoost val logloss:   {log_loss(y_val, model.predict_proba(X_val)[:,1]):.4f}")
        print("  XGBoost configured with GPU acceleration (CUDA)")
        return model
    except Exception as e:
        print(f"  XGBoost GPU failed: {e}. Falling back to CPU.")
        model = xgb.XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, gamma=0.1,
            reg_alpha=0.1, reg_lambda=1.0,
            tree_method="hist", device="cpu",
            random_state=RANDOM_STATE,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            eval_metric="logloss", verbosity=0,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        print("  XGBoost trained on CPU (fallback)")
        return model


def train_lightgbm(X_train, y_train, X_val, y_val):
    """Train LightGBM with GPU acceleration."""
    print("\n" + "=" * 60)
    print("  Training LightGBM Base Learner (GPU)")
    print("=" * 60)

    try:
        model = lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=7,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            device_type="gpu",
            random_state=RANDOM_STATE,
            verbose=-1,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="logloss",
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                       lgb.log_evaluation(0)],
        )
        print(f"  LightGBM train logloss: {log_loss(y_train, model.predict_proba(X_train)[:,1]):.4f}")
        print(f"  LightGBM val logloss:   {log_loss(y_val, model.predict_proba(X_val)[:,1]):.4f}")
        print("  LightGBM configured with GPU acceleration")
        return model
    except Exception as e:
        print(f"  LightGBM GPU failed: {e}. Falling back to CPU.")
        model = lgb.LGBMClassifier(
            n_estimators=500, max_depth=7, learning_rate=0.05,
            num_leaves=63, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            device_type="cpu", random_state=RANDOM_STATE,
            verbose=-1,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="logloss",
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                       lgb.log_evaluation(0)],
        )
        print("  LightGBM trained on CPU (fallback)")
        return model


def train_neural_network(X_train, y_train, X_val, y_val, input_dim):
    """Train PyTorch neural network with GPU acceleration."""
    print("\n" + "=" * 60)
    print(f"  Training Neural Network Base Learner ({DEVICE.upper()})")
    print("=" * 60)

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )

    train_loader = DataLoader(train_dataset, batch_size=NN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=NN_BATCH_SIZE * 2, shuffle=False)

    model = UFCFightNet(input_dim, NN_HIDDEN_LAYERS, NN_DROPOUT).to(DEVICE)

    class_counts = [len(y_train) - y_train.sum(), y_train.sum()]
    pos_weight = torch.tensor([class_counts[0] / max(class_counts[1], 1)]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=NN_LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    checkpoint_path = MODEL_PATHS["nn"]
    early_stopping = EarlyStopping(patience=NN_PATIENCE)

    for epoch in range(NN_EPOCHS):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(batch_X)
        train_loss /= len(train_dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y).item() * len(batch_X)
        val_loss /= len(val_dataset)

        scheduler.step(val_loss)

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{NN_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        early_stopping(val_loss, model, checkpoint_path)
        if early_stopping.early_stop:
            print(f"  Early stopping triggered at epoch {epoch+1}")
            break

    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.eval()

    with torch.no_grad():
        train_preds = model(torch.tensor(X_train, dtype=torch.float32).to(DEVICE)).cpu().numpy()
        val_preds = model(torch.tensor(X_val, dtype=torch.float32).to(DEVICE)).cpu().numpy()

    print(f"  NN train logloss: {log_loss(y_train, train_preds):.4f}")
    print(f"  NN val logloss:   {log_loss(y_val, val_preds):.4f}")
    print(f"  NN trained on {DEVICE.upper()}")

    return model


def build_meta_features(xgb_model, lgb_model, nn_model, X):
    """Generate out-of-fold meta-features from all base learners."""
    xgb_proba = xgb_model.predict_proba(X)[:, 1].reshape(-1, 1)
    lgb_proba = lgb_model.predict_proba(X)[:, 1].reshape(-1, 1)

    if isinstance(nn_model, UFCFightNet):
        nn_model.eval()
        with torch.no_grad():
            tensor_X = torch.tensor(X, dtype=torch.float32).to(DEVICE)
            nn_proba = nn_model(tensor_X).cpu().numpy().reshape(-1, 1)
    else:
        nn_proba = np.zeros((len(X), 1))

    meta_X = np.hstack([xgb_proba, lgb_proba, nn_proba])

    if len(meta_X) >= 3:
        meta_X = np.hstack([
            meta_X,
            (meta_X[:, 0] + meta_X[:, 1] + meta_X[:, 2]).reshape(-1, 1) / 3,  # avg proba
            np.max(meta_X, axis=1).reshape(-1, 1),  # max proba
            np.min(meta_X, axis=1).reshape(-1, 1),  # min proba
            (np.max(meta_X, axis=1) - np.min(meta_X, axis=1)).reshape(-1, 1),  # disagreement
        ])

    return meta_X


def train_meta_learner(meta_X_train, y_train):
    """Train logistic regression meta-learner on base learner outputs."""
    print("\n" + "=" * 60)
    print("  Training Meta-Learner (Logistic Regression)")
    print("=" * 60)

    meta_model = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    meta_model.fit(meta_X_train, y_train)

    train_preds = meta_model.predict_proba(meta_X_train)[:, 1]
    print(f"  Meta-learner train accuracy: {accuracy_score(y_train, (train_preds > 0.5).astype(int)):.4f}")
    print(f"  Meta-learner train logloss:  {log_loss(y_train, train_preds):.4f}")

    return meta_model


def evaluate_ensemble(models, X_test, y_test, feature_names):
    """Evaluate the full stacked ensemble and generate plots."""
    print("\n" + "=" * 60)
    print("  Model Evaluation")
    print("=" * 60)

    xgb_model = models["xgb"]
    lgb_model = models["lgb"]
    nn_model = models["nn"]
    meta_model = models["meta"]

    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    lgb_proba = lgb_model.predict_proba(X_test)[:, 1]

    nn_model.eval()
    with torch.no_grad():
        nn_proba = nn_model(torch.tensor(X_test, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    nn_proba = nn_proba.clip(0, 1)

    meta_X_test = build_meta_features(xgb_model, lgb_model, nn_model, X_test)
    ensemble_proba = meta_model.predict_proba(meta_X_test)[:, 1]

    results = {}
    for name, proba in [("XGBoost", xgb_proba), ("LightGBM", lgb_proba),
                         ("NeuralNet", nn_proba), ("Ensemble", ensemble_proba)]:
        pred_binary = (proba > 0.5).astype(int)
        results[name] = {
            "Accuracy": accuracy_score(y_test, pred_binary),
            "LogLoss": log_loss(y_test, proba),
            "ROC-AUC": roc_auc_score(y_test, proba),
            "Brier": brier_score_loss(y_test, proba),
        }

    results_df = pd.DataFrame(results).T
    print("\n" + results_df.round(4).to_string())

    print(f"\n  Best Model: {results_df['ROC-AUC'].idxmax()} (ROC-AUC: {results_df['ROC-AUC'].max():.4f})")

    os.makedirs(PLOTS_DIR, exist_ok=True)

    plt.figure(figsize=(10, 8))
    for name, proba in [("XGBoost", xgb_proba), ("LightGBM", lgb_proba),
                         ("NeuralNet", nn_proba), ("Ensemble", ensemble_proba)]:
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_auc = roc_auc_score(y_test, proba)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curves - UFC Fight Prediction Ensemble", fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "roc_curve.png", dpi=150)
    plt.close()
    print(f"  Saved ROC curve to {PLOTS_DIR / 'roc_curve.png'}")

    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, (ensemble_proba > 0.5).astype(int))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Fighter B Wins", "Fighter A Wins"],
                yticklabels=["Fighter B Wins", "Fighter A Wins"])
    plt.title("Confusion Matrix - Ensemble Model", fontsize=14)
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    return results_df


def compute_shap_values(models, X_sample, feature_names):
    """Compute SHAP values using TreeExplainer for XGBoost and DeepExplainer for NN."""
    print("\n" + "=" * 60)
    print("  Computing SHAP Feature Importance")
    print("=" * 60)

    os.makedirs(PLOTS_DIR, exist_ok=True)

    sample_size = min(200, len(X_sample))
    X_sample = X_sample[:sample_size]

    try:
        print("  Computing TreeExplainer for XGBoost...")
        explainer_xgb = shap.TreeExplainer(models["xgb"])
        shap_values_xgb = explainer_xgb.shap_values(X_sample)

        if isinstance(shap_values_xgb, list):
            shap_values_xgb = shap_values_xgb[1] if len(shap_values_xgb) > 1 else shap_values_xgb[0]

        plt.figure(figsize=(12, 10))
        shap.summary_plot(
            shap_values_xgb, X_sample,
            feature_names=feature_names,
            show=False,
            max_display=20,
        )
        plt.title("XGBoost SHAP Feature Importance", fontsize=14)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "shap_xgb_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved XGBoost SHAP plot to {PLOTS_DIR / 'shap_xgb_summary.png'}")

        plt.figure(figsize=(10, 8))
        mean_shap = np.abs(shap_values_xgb).mean(axis=0)
        top_idx = np.argsort(mean_shap)[-20:]
        plt.barh(range(len(top_idx)), mean_shap[top_idx])
        plt.yticks(range(len(top_idx)), [feature_names[i] for i in top_idx])
        plt.xlabel("Mean |SHAP Value|", fontsize=12)
        plt.title("XGBoost Top 20 Features by SHAP", fontsize=14)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "shap_xgb_bar.png", dpi=150)
        plt.close()
        print(f"  Saved XGBoost SHAP bar plot to {PLOTS_DIR / 'shap_xgb_bar.png'}")

    except Exception as e:
        print(f"  XGBoost SHAP failed: {e}")

    try:
        print("  Computing TreeExplainer for LightGBM...")
        explainer_lgb = shap.TreeExplainer(models["lgb"])
        shap_values_lgb = explainer_lgb.shap_values(X_sample)

        if isinstance(shap_values_lgb, list):
            shap_values_lgb = shap_values_lgb[1] if len(shap_values_lgb) > 1 else shap_values_lgb[0]

        plt.figure(figsize=(12, 10))
        shap.summary_plot(
            shap_values_lgb, X_sample,
            feature_names=feature_names,
            show=False,
            max_display=20,
        )
        plt.title("LightGBM SHAP Feature Importance", fontsize=14)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "shap_lgb_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved LightGBM SHAP plot to {PLOTS_DIR / 'shap_lgb_summary.png'}")

    except Exception as e:
        print(f"  LightGBM SHAP failed: {e}")

    try:
        print("  Computing DeepExplainer for Neural Network...")
        nn_model = models["nn"]
        nn_model.eval()
        background = torch.tensor(X_sample[:100], dtype=torch.float32).to(DEVICE)
        explainer_nn = shap.DeepExplainer(nn_model, background)
        shap_values_nn = explainer_nn.shap_values(
            torch.tensor(X_sample[:100], dtype=torch.float32).to(DEVICE)
        )
        if isinstance(shap_values_nn, list):
            shap_values_nn = shap_values_nn[0]

        shap_values_nn_np = np.array(shap_values_nn)
        if len(shap_values_nn_np.shape) > 2:
            shap_values_nn_np = shap_values_nn_np.reshape(shap_values_nn_np.shape[0], shap_values_nn_np.shape[2])

        plt.figure(figsize=(12, 10))
        shap.summary_plot(
            shap_values_nn_np,
            X_sample[:shap_values_nn_np.shape[0]],
            feature_names=feature_names,
            show=False,
            max_display=20,
        )
        plt.title("Neural Network SHAP Feature Importance", fontsize=14)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "shap_nn_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved Neural Network SHAP plot to {PLOTS_DIR / 'shap_nn_summary.png'}")

    except Exception as e:
        print(f"  Neural Network SHAP failed: {e}")

    print("  SHAP analysis complete.")


def save_models(models):
    """Save all trained models to disk."""
    print("\n" + "=" * 60)
    print("  Saving Models")
    print("=" * 60)

    os.makedirs(MODELS_DIR, exist_ok=True)

    models["xgb"].save_model(str(MODEL_PATHS["xgb"]))
    print(f"  XGBoost saved to {MODEL_PATHS['xgb']}")

    models["lgb"].booster_.save_model(str(MODEL_PATHS["lgb"]))
    print(f"  LightGBM saved to {MODEL_PATHS['lgb']}")

    torch.save(models["nn"].state_dict(), MODEL_PATHS["nn"])
    print(f"  Neural Network saved to {MODEL_PATHS['nn']}")

    joblib.dump(models["meta"], MODEL_PATHS["meta"])
    print(f"  Meta-learner saved to {MODEL_PATHS['meta']}")

    print("  All models saved successfully!")


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    X_train, X_test, y_train, y_test, feature_names = load_data()

    val_size = int(len(X_train) * 0.15)
    X_tr, X_val = X_train[:-val_size], X_train[-val_size:]
    y_tr, y_val = y_train[:-val_size], y_train[-val_size:]

    print(f"\n  Training split: {len(X_tr)} train | {len(X_val)} validation | {len(X_test)} test")

    xgb_model = train_xgboost(X_tr, y_tr, X_val, y_val)
    lgb_model = train_lightgbm(X_tr, y_tr, X_val, y_val)
    nn_model = train_neural_network(X_tr, y_tr, X_val, y_val, input_dim=X_train.shape[1])

    print("\n" + "=" * 60)
    print("  Building Meta-Features")
    print("=" * 60)

    meta_X_train = build_meta_features(xgb_model, lgb_model, nn_model, X_train)
    meta_model = train_meta_learner(meta_X_train, y_train)

    models = {"xgb": xgb_model, "lgb": lgb_model, "nn": nn_model, "meta": meta_model}

    results = evaluate_ensemble(models, X_test, y_test, feature_names)

    compute_shap_values(models, X_test, feature_names)

    save_models(models)

    print("\n" + "=" * 60)
    print("  Training Complete! Summary:")
    print("=" * 60)
    print(f"  Best ROC-AUC: {results['ROC-AUC']['Ensemble']:.4f} (Ensemble)")
    print(f"  Best LogLoss: {results['LogLoss']['Ensemble']:.4f} (Ensemble)")
    print(f"  Best Accuracy: {results['Accuracy']['Ensemble']:.4f} (Ensemble)")
    print(f"\n  Next step: python scripts/predict_fight.py --fighter_a 'Fighter A' --fighter_b 'Fighter B'")

    return models, results


if __name__ == "__main__":
    main()
