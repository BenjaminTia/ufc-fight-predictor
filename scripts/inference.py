"""
Inference Engine
Loads the trained stacked ensemble and generates fight predictions.
Used by predict_fight.py for the CLI interface.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

import xgboost as xgb
import lightgbm as lgb

import torch
import torch.nn as nn

DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"

MODEL_PATHS = {
    "xgb": MODELS_DIR / "xgb_model.json",
    "lgb": MODELS_DIR / "lgb_model.txt",
    "nn": MODELS_DIR / "nn_model.pt",
    "meta": MODELS_DIR / "meta_learner.pkl",
    "scaler": MODELS_DIR / "scaler.pkl",
    "feature_names": MODELS_DIR / "feature_names.pkl",
    "nn_temp": MODELS_DIR / "nn_temperature.pkl",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NN_HIDDEN_LAYERS = [128, 64]
NN_DROPOUT = 0.25


class UFCFightNet(nn.Module):
    def __init__(self, input_dim, hidden_layers=None, dropout=0.25):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [128, 64]
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze()


class UFCPredictor:
    """Complete inference pipeline for UFC fight prediction."""

    def __init__(self, model_dir=None):
        self.model_dir = Path(model_dir) if model_dir else MODELS_DIR
        self.models = {}
        self.scaler = None
        self.feature_names = None
        self._loaded = False
        self._load_models()

    def _load_models(self):
        """Load all trained models and preprocessing artifacts."""
        try:
            self.models["xgb"] = xgb.XGBClassifier()
            self.models["xgb"].load_model(str(MODEL_PATHS["xgb"]))
            print(f"  Loaded XGBoost from {MODEL_PATHS['xgb']}")

            self.models["lgb"] = lgb.Booster(model_file=str(MODEL_PATHS["lgb"]))
            print(f"  Loaded LightGBM from {MODEL_PATHS['lgb']}")

            self.scaler = joblib.load(MODEL_PATHS["scaler"])
            self.feature_names = joblib.load(MODEL_PATHS["feature_names"])
            input_dim = len(self.feature_names)

            self.models["nn"] = UFCFightNet(input_dim, NN_HIDDEN_LAYERS, NN_DROPOUT).to(DEVICE)
            self.models["nn"].load_state_dict(
                torch.load(MODEL_PATHS["nn"], map_location=DEVICE, weights_only=True)
            )
            self.models["nn"].eval()
            print(f"  Loaded Neural Network from {MODEL_PATHS['nn']} ({DEVICE})")

            self.nn_temperature = 2.0
            if MODEL_PATHS["nn_temp"].exists():
                self.nn_temperature = float(joblib.load(MODEL_PATHS["nn_temp"]))
                print(f"  Loaded NN temperature: {self.nn_temperature:.2f}")

            self.models["meta"] = joblib.load(MODEL_PATHS["meta"])
            print(f"  Loaded Meta-learner from {MODEL_PATHS['meta']}")

            self._loaded = True
            print("  Predictor initialized successfully!")

        except FileNotFoundError as e:
            print(f"  ERROR: Model file not found: {e}")
            print("  Run model_training.py first to train the models.")
            raise
        except Exception as e:
            print(f"  ERROR loading models: {e}")
            raise

    def _encode_features(self, fighter_a_features, fighter_b_features):
        """
        Convert two fighter feature dicts into the scaled feature vector
        expected by the ensemble, with matchup differential features.
        """
        features = {}

        for key in self.feature_names:
            features[key] = 0.0

        for key, val in fighter_a_features.items():
            f_key = f"a_{key}"
            if f_key in self.feature_names or f_key in features:
                features[f_key] = float(val) if val is not None else 0.0

        for key, val in fighter_b_features.items():
            f_key = f"b_{key}"
            if f_key in self.feature_names or f_key in features:
                features[f_key] = float(val) if val is not None else 0.0

        return self._compute_matchup_features(fighter_a_features, fighter_b_features, features)

    def _compute_matchup_features(self, fa, fb, features):
        """Compute style matchup differentials and ratios."""
        metrics_pairs = [
            ("sig_str", "sig_str"), ("total_str", "total_str"), ("td", "td"),
            ("sub_att", "sub_att"), ("ctrl", "ctrl"),
            ("slpm", "slpm"), ("sapm", "sapm"), ("td_avg", "td_avg"),
            ("strike_acc", "strike_acc"), ("strike_def", "strike_def"),
            ("td_acc", "td_acc"), ("td_def", "td_def"),
            ("height_inches", "height_inches"), ("reach_inches", "reach_inches"),
            ("win_rate", "win_rate"), ("weight_class", "weight_class"),
            ("sentiment", "sentiment"), ("momentum", "momentum"),
        ]

        for fa_key, fb_key in metrics_pairs:
            a_val = fa.get(fa_key, 0)
            b_val = fb.get(fb_key, 0)
            if a_val is not None and b_val is not None:
                base = fa_key.replace("total_", "").replace("sub_", "")
                features[f"diff_{base}"] = float(a_val) - float(b_val)
                total = abs(float(a_val)) + abs(float(b_val)) + 0.001
                features[f"ratio_{base}"] = float(a_val) / total

        a_exp = fa.get("num_fights", 0)
        b_exp = fb.get("num_fights", 0)
        features["a_experience"] = float(a_exp) if a_exp else 0.0
        features["b_experience"] = float(b_exp) if b_exp else 0.0
        features["experience_diff"] = float(a_exp) - float(b_exp)

        a_wc = fa.get("weight_class", 0)
        b_wc = fb.get("weight_class", 0)
        features["same_weight_class"] = 1.0 if a_wc == b_wc else 0.0

        features["sentiment_diff"] = features.get("diff_sentiment", features.get("diff_momentum", 0.0))

        consensus_a = features.get("expert_consensus_a", 0.5)
        consensus_b = features.get("expert_consensus_b", 0.5)
        features["consensus_diff"] = consensus_a - consensus_b
        features["expert_agreement"] = max(consensus_a, consensus_b)

        features["a_news_articles"] = features.get("a_news_articles", 0)
        features["b_news_articles"] = features.get("b_news_articles", 0)

        return features

    def _features_to_array(self, feature_dict):
        """Convert feature dict to properly ordered, scaled numpy array."""
        feature_values = []
        for name in self.feature_names:
            val = feature_dict.get(name, 0.0)
            feature_values.append(float(val) if val is not None else 0.0)

        arr = np.array(feature_values, dtype=np.float32).reshape(1, -1)
        arr_scaled = self.scaler.transform(arr)
        return arr_scaled

    def predict(self, fighter_a_features, fighter_b_features):
        """
        Predict outcome of a fight between two fighters.

        Parameters:
            fighter_a_features: dict with fighter A's attributes
            fighter_b_features: dict with fighter B's attributes

        Returns:
            dict with win probabilities, confidence, and individual model predictions
        """
        if not self._loaded:
            raise RuntimeError("Predictor not loaded. Call _load_models() first.")

        raw_features = self._encode_features(fighter_a_features, fighter_b_features)
        X = self._features_to_array(raw_features)

        xgb_proba = self.models["xgb"].predict_proba(X)[0, 1]
        lgb_proba = self.models["lgb"].predict(X, raw_score=False)[0]

        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
            nn_logit = self.models["nn"](X_tensor).item()
            nn_proba = 1.0 / (1.0 + np.exp(-nn_logit / self.nn_temperature))

        meta_X = np.array([[
            xgb_proba, lgb_proba, nn_proba,
            (xgb_proba + lgb_proba + nn_proba) / 3,
            max(xgb_proba, lgb_proba, nn_proba),
            min(xgb_proba, lgb_proba, nn_proba),
            max(xgb_proba, lgb_proba, nn_proba) - min(xgb_proba, lgb_proba, nn_proba),
        ]])
        ensemble_proba = float(self.models["meta"].predict_proba(meta_X)[0, 1])

        return {
            "fighter_a_win_probability": float(ensemble_proba),
            "fighter_b_win_probability": float(1 - ensemble_proba),
            "predicted_winner": fighter_a_features.get("name", "Fighter A") if ensemble_proba > 0.5 else fighter_b_features.get("name", "Fighter B"),
            "confidence": float(abs(ensemble_proba - 0.5) * 2),
            "individual_predictions": {
                "xgb_fighter_a_prob": float(xgb_proba),
                "lgb_fighter_a_prob": float(lgb_proba),
                "nn_fighter_a_prob": float(nn_proba),
            },
            "model_agreement": float(
                1 - (max(xgb_proba, lgb_proba, nn_proba) - min(xgb_proba, lgb_proba, nn_proba))
            ),
        }

    def predict_proba(self, fighter_a_name, fighter_b_name, features_a=None, features_b=None):
        """Convenience method for predict_fight.py CLI."""
        if features_a is None:
            features_a = {}
        if features_b is None:
            features_b = {}
        features_a["name"] = fighter_a_name
        features_b["name"] = fighter_b_name
        return self.predict(features_a, features_b)
