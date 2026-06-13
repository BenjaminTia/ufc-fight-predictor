"""
Feature Engineering Pipeline
Loads scraped CSVs, builds comprehensive features for UFC fight prediction.
Outputs: data/training_data.csv, data/feature_names.json
"""

import os
import re
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"
FIGHTS_CSV = DATA_DIR / "ufc_fight_stats.csv"
PROFILES_CSV = DATA_DIR / "fighter_profiles.csv"
EXPERT_PICKS_CSV = DATA_DIR / "expert_picks.csv"
EXPERT_HISTORY_CSV = DATA_DIR / "expert_history.csv"
SENTIMENT_CSV = DATA_DIR / "fighter_news_sentiment.csv"
TRAINING_CSV = DATA_DIR / "training_data.csv"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.pkl"


def parse_height(height_str):
    """Convert height string like 5' 11\" to inches."""
    if pd.isna(height_str) or not height_str:
        return np.nan
    try:
        match = re.match(r"(\d+)[\'′].*?(\d+)[\"″]?", str(height_str))
        if match:
            feet = int(match.group(1))
            inches = int(match.group(2))
            return feet * 12 + inches
    except Exception:
        pass
    return np.nan


def parse_reach(reach_str):
    """Convert reach string to float inches."""
    if pd.isna(reach_str) or not reach_str:
        return np.nan
    try:
        return float(str(reach_str).replace('"', "").replace("′", "").strip())
    except Exception:
        return np.nan


def parse_percent(pct_str):
    """Convert percentage string to float."""
    if pd.isna(pct_str) or not pct_str:
        return np.nan
    try:
        return float(str(pct_str).replace("%", "").strip()) / 100
    except Exception:
        return np.nan


def parse_float(val):
    """Safely parse a value to float."""
    if pd.isna(val) or val == "" or val == "--":
        return np.nan
    try:
        return float(str(val).replace(",", "").strip())
    except Exception:
        return np.nan


def load_and_clean_fight_stats():
    """Load and clean UFC fight stats CSV."""
    print("[1/5] Loading fight statistics...")

    if not FIGHTS_CSV.exists():
        print(f"  WARNING: {FIGHTS_CSV} not found. Generating synthetic data for demonstration.")
        return generate_synthetic_fight_data()

    df = pd.read_csv(FIGHTS_CSV, low_memory=False)
    print(f"  Loaded {len(df)} fight records")

    numeric_cols = ["a_sig_str", "a_sig_str_pct", "a_total_str", "a_td", "a_td_pct",
                    "a_sub_att", "a_rev", "a_ctrl", "a_head", "a_body", "a_leg",
                    "a_distance", "a_clinch", "a_ground",
                    "b_sig_str", "b_sig_str_pct", "b_total_str", "b_td", "b_td_pct",
                    "b_sub_att", "b_rev", "b_ctrl", "b_head", "b_body", "b_leg",
                    "b_distance", "b_clinch", "b_ground",
                    "a_sig_str_landed", "a_sig_str_attempted", "a_td_landed", "a_td_attempted",
                    "b_sig_str_landed", "b_sig_str_attempted", "b_td_landed", "b_td_attempted"]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(parse_float)

    return df


def load_profiles():
    """Load fighter profiles."""
    print("[2/5] Loading fighter profiles...")

    if not PROFILES_CSV.exists():
        print(f"  WARNING: {PROFILES_CSV} not found. Using default profile data.")
        return pd.DataFrame()

    df = pd.read_csv(PROFILES_CSV)
    print(f"  Loaded {len(df)} fighter profiles")

    if "height" in df.columns:
        df["height_inches"] = df["height"].apply(parse_height)
    if "reach" in df.columns:
        df["reach_inches"] = df["reach"].apply(parse_reach)
    if "sig_strike_accuracy" in df.columns:
        df["strike_acc"] = df["sig_strike_accuracy"].apply(parse_percent)
    if "sig_strike_defense" in df.columns:
        df["strike_def"] = df["sig_strike_defense"].apply(parse_percent)
    if "takedown_accuracy" in df.columns:
        df["td_acc"] = df["takedown_accuracy"].apply(parse_percent)
    if "takedown_defense" in df.columns:
        df["td_def"] = df["takedown_defense"].apply(parse_percent)
    if "sig_strikes_landed_per_min" in df.columns:
        df["slpm"] = df["sig_strikes_landed_per_min"].apply(parse_float)
    if "sig_strikes_absorbed_per_min" in df.columns:
        df["sapm"] = df["sig_strikes_absorbed_per_min"].apply(parse_float)
    if "takedown_avg" in df.columns:
        df["td_avg"] = df["takedown_avg"].apply(parse_float)
    if "submission_avg" in df.columns:
        df["sub_avg"] = df["submission_avg"].apply(parse_float)
    if "wins" in df.columns:
        df["career_wins"] = df["wins"].apply(parse_float)
    else:
        df["career_wins"] = 15.0
    if "losses" in df.columns:
        df["career_losses"] = df["losses"].apply(parse_float)
    else:
        df["career_losses"] = 5.0

    if "full_name" not in df.columns:
        df["full_name"] = (df.get("first_name", "").fillna("") + " " + df.get("last_name", "").fillna("")).str.strip()

    df["win_rate"] = np.where(
        (df["career_wins"].notna()) & (df["career_losses"].notna()),
        df["career_wins"] / (df["career_wins"] + df["career_losses"] + 0.001),
        np.nan,
    )

    weight_map = {"Strawweight": 1, "Flyweight": 2, "Bantamweight": 3,
                  "Featherweight": 4, "Lightweight": 5, "Welterweight": 6,
                  "Middleweight": 7, "Light Heavyweight": 8, "Heavyweight": 9}
    if "weight" in df.columns:
        df["weight_class"] = df["weight"].map(weight_map).fillna(5.0)

    return df


def build_fight_level_features(fight_df):
    """Aggregate round-by-round data into fight-level features."""
    print("  Building fight-level features...")

    fight_df = fight_df.copy()
    if "winner" not in fight_df.columns:
        fight_df["winner"] = "A"

    fight_features = []

    for (fighter_a, fighter_b), group in fight_df.groupby(["fighter_a", "fighter_b"], sort=False):
        group = group.sort_values("round")
        rounds = len(group)

        features = {
            "fighter_a": fighter_a,
            "fighter_b": fighter_b,
            "winner": group["winner"].iloc[0],
            "num_rounds": rounds,
            "method": group["method_type"].iloc[0] if "method_type" in group.columns else "",
        }

        strike_cols_a = [c for c in group.columns if c.startswith("a_sig_str") and "pct" not in c]
        strike_cols_b = [c for c in group.columns if c.startswith("b_sig_str") and "pct" not in c]

        for col_a, col_b in zip(
            ["a_sig_str", "a_total_str", "a_td", "a_sub_att", "a_ctrl", "a_head", "a_body", "a_leg", "a_distance", "a_clinch", "a_ground"],
            ["b_sig_str", "b_total_str", "b_td", "b_sub_att", "b_ctrl", "b_head", "b_body", "b_leg", "b_distance", "b_clinch", "b_ground"],
        ):
            name = col_a.replace("a_", "")
            if col_a in group.columns:
                features[f"{name}_a_total"] = group[col_a].sum()
            if col_b in group.columns:
                features[f"{name}_b_total"] = group[col_b].sum()

        for col_a, col_b in [("a_sig_str_pct", "b_sig_str_pct"), ("a_td_pct", "b_td_pct")]:
            name = col_a.replace("a_", "").replace("_pct", "")
            if col_a in group.columns:
                features[f"{name}_a_avg_pct"] = group[col_a].mean()
            if col_b in group.columns:
                features[f"{name}_b_avg_pct"] = group[col_b].mean()

        if rounds >= 2:
            for col_a, col_b in [("a_sig_str", "b_sig_str"), ("a_td", "b_td")]:
                name = col_a.replace("a_", "")
                if col_a in group.columns:
                    r1_val = group[col_a].iloc[0] if len(group) > 0 else 0
                    r_last_val = group[col_a].iloc[-1] if len(group) > 0 else 0
                    features[f"{name}_a_rd1"] = r1_val
                    features[f"{name}_a_rd_last"] = r_last_val
                    features[f"{name}_a_dropoff"] = r1_val - r_last_val if r1_val > 0 else 0
                if col_b in group.columns:
                    r1_val = group[col_b].iloc[0] if len(group) > 0 else 0
                    r_last_val = group[col_b].iloc[-1] if len(group) > 0 else 0
                    features[f"{name}_b_rd1"] = r1_val
                    features[f"{name}_b_rd_last"] = r_last_val
                    features[f"{name}_b_dropoff"] = r1_val - r_last_val if r1_val > 0 else 0

        if "a_sig_str" in group.columns and "b_sig_str" in group.columns:
            features["a_sig_str_per_round"] = group["a_sig_str"].sum() / max(rounds, 1)
            features["b_sig_str_per_round"] = group["b_sig_str"].sum() / max(rounds, 1)

        if "a_str_def_pct" in group.columns and "b_str_def_pct" in group.columns:
            features["a_striking_def"] = group["a_str_def_pct"].mean()
            features["b_striking_def"] = group["b_str_def_pct"].mean()

        fight_features.append(features)

    return pd.DataFrame(fight_features)


def build_style_matchup_features(fight_df, profiles_df):
    """Create style matchup metrics: striking differential, grappling differential."""
    print("[3/5] Building style matchup features...")

    features = []

    unique_fighters = set(fight_df["fighter_a"].unique()) | set(fight_df["fighter_b"].unique())

    fighter_stats = {}
    for fighter in unique_fighters:
        stats = {}

        fights_as_a = fight_df[fight_df["fighter_a"] == fighter]
        fights_as_b = fight_df[fight_df["fighter_b"] == fighter]

        for prefix, fights in [("a_", fights_as_a), ("b_", fights_as_b)]:
            for metric in ["sig_str", "total_str", "td", "sub_att", "ctrl", "head", "body", "leg"]:
                col = f"{prefix}{metric}"
                if col in fight_df.columns and len(fights) > 0:
                    val = fights[col].sum()
                    stats[f"{metric}_career"] = stats.get(f"{metric}_career", 0) + val

        num_fights = len(fights_as_a) + len(fights_as_b)
        stats["num_fights"] = num_fights

        profile_row = profiles_df[profiles_df["full_name"].str.lower() == fighter.lower()] if not profiles_df.empty else pd.DataFrame()
        if not profile_row.empty:
            for col in ["height_inches", "reach_inches", "strike_acc", "strike_def", "td_acc", "td_def",
                        "slpm", "sapm", "td_avg", "sub_avg", "win_rate", "weight_class", "career_wins", "career_losses"]:
                if col in profile_row.columns:
                    stats[f"profile_{col}"] = profile_row[col].values[0]

        fighter_stats[fighter] = stats

    for _, fight in fight_df.iterrows():
        fa_name = fight["fighter_a"]
        fb_name = fight["fighter_b"]

        fa = fighter_stats.get(fa_name, {})
        fb = fighter_stats.get(fb_name, {})

        row = {
            "fighter_a": fa_name,
            "fighter_b": fb_name,
            "winner": fight.get("winner", "A"),
            "method": fight.get("method_type", fight.get("method", "")),
        }

        for metric in ["sig_str", "total_str", "td", "sub_att", "ctrl", "head", "body", "leg"]:
            a_val = fa.get(f"{metric}_career", 0)
            b_val = fb.get(f"{metric}_career", 0)
            total = a_val + b_val + 0.001

            row[f"diff_{metric}"] = a_val - b_val
            row[f"ratio_{metric}"] = a_val / total

        for prof_metric in ["height_inches", "reach_inches", "strike_acc", "strike_def",
                            "td_acc", "td_def", "slpm", "sapm", "td_avg", "sub_avg", "win_rate", "weight_class"]:
            a_val = fa.get(f"profile_{prof_metric}", np.nan)
            b_val = fb.get(f"profile_{prof_metric}", np.nan)
            if not pd.isna(a_val) and not pd.isna(b_val):
                row[f"diff_{prof_metric}"] = a_val - b_val
                row[f"ratio_{prof_metric}"] = a_val / (a_val + b_val + 0.001)

            a_fights = fa.get("num_fights", 0)
            b_fights = fb.get("num_fights", 0)
            row["a_experience"] = a_fights
            row["b_experience"] = b_fights
            row["experience_diff"] = a_fights - b_fights

            a_wc = fa.get("profile_weight_class", np.nan)
            b_wc = fb.get("profile_weight_class", np.nan)
            if not pd.isna(a_wc) and not pd.isna(b_wc):
                row["same_weight_class"] = 1.0 if a_wc == b_wc else 0.0
            else:
                row["same_weight_class"] = 1.0

        features.append(row)

    return pd.DataFrame(features)


def build_sentiment_features(fight_df):
    """Integrate NLP sentiment scores."""
    print("[4/5] Building sentiment features...")

    if not SENTIMENT_CSV.exists():
        print(f"  WARNING: {SENTIMENT_CSV} not found. Using neutral sentiment (0.0).")
        return fight_df

    sentiment_df = pd.read_csv(SENTIMENT_CSV)

    sentiment_summary = sentiment_df.groupby("fighter_name").agg(
        avg_sentiment=("sentiment_score", "mean"),
        article_count=("sentiment_score", "count"),
        sentiment_std=("sentiment_score", "std"),
        max_positive=("sentiment_score", "max"),
        min_negative=("sentiment_score", "min"),
    ).reset_index()

    sentiment_summary["momentum_score"] = (
        sentiment_summary["avg_sentiment"].fillna(0) * 0.4
        + (sentiment_summary["article_count"] / max(1, int(sentiment_summary["article_count"].max()))) * 0.2
        + (1 - sentiment_summary["sentiment_std"].fillna(1).clip(0, 1)) * 0.4
    )

    result = fight_df.copy()
    for side in ["a", "b"]:
        col_name = f"fighter_{side}"
        result = result.merge(
            sentiment_summary[["fighter_name", "avg_sentiment", "article_count", "momentum_score"]],
            left_on=col_name,
            right_on="fighter_name",
            how="left",
            suffixes=("", f"_{side}"),
        )
        result.rename(columns={
            "avg_sentiment": f"{side}_sentiment",
            "article_count": f"{side}_news_articles",
            "momentum_score": f"{side}_momentum",
        }, inplace=True)
        if "fighter_name" in result.columns:
            result.drop(columns=["fighter_name"], inplace=True)

    for col in ["a_sentiment", "b_sentiment", "a_momentum", "b_momentum"]:
        if col in result.columns:
            result[col] = result[col].fillna(0.0)

    if "a_momentum" in result.columns and "b_momentum" in result.columns:
        result["sentiment_diff"] = result["a_momentum"] - result["b_momentum"]

    return result


def build_expert_consensus_features(fight_df):
    """Build weighted expert consensus features."""
    print("[5/5] Building expert consensus features...")

    result = fight_df.copy()

    if not EXPERT_PICKS_CSV.exists() or not EXPERT_HISTORY_CSV.exists():
        print(f"  WARNING: Expert data not found. Using neutral consensus (0.5).")
        result["expert_consensus_a"] = 0.5
        result["expert_consensus_b"] = 0.5
        result["consensus_diff"] = 0.0
        result["expert_agreement"] = 0.5
        return result

    picks_df = pd.read_csv(EXPERT_PICKS_CSV)
    history_df = pd.read_csv(EXPERT_HISTORY_CSV)

    if "expert_name" not in picks_df.columns or "reliability_weight" not in history_df.columns:
        result["expert_consensus_a"] = 0.5
        result["expert_consensus_b"] = 0.5
        result["consensus_diff"] = 0.0
        result["expert_agreement"] = 0.5
        return result

    if "expert_name" in picks_df.columns and "reliability_weight" in history_df.columns and "expert_name" in history_df.columns:
        picks_weighted = picks_df.merge(
            history_df[["expert_name", "reliability_weight"]],
            on="expert_name",
            how="left",
        )
        picks_weighted["reliability_weight"] = picks_weighted["reliability_weight"].fillna(0.5)

        consensus_list = []
        for (fa, fb), group in picks_weighted.groupby(["fighter_a", "fighter_b"]):
            total_weight = group["reliability_weight"].sum()
            if total_weight > 0:
                a_picks = group[group["predicted_winner"] == fa]["reliability_weight"].sum()
                b_picks = group[group["predicted_winner"] == fb]["reliability_weight"].sum()
                consensus_list.append({
                    "fighter_a": fa,
                    "fighter_b": fb,
                    "expert_consensus_a": a_picks / total_weight,
                    "expert_consensus_b": b_picks / total_weight,
                    "expert_agreement": max(a_picks, b_picks) / total_weight,
                })

        if consensus_list:
            consensus_df = pd.DataFrame(consensus_list)
            result = result.merge(consensus_df, on=["fighter_a", "fighter_b"], how="left")

    for col in ["expert_consensus_a", "expert_consensus_b", "expert_agreement"]:
        if col in result.columns:
            result[col] = result[col].fillna(0.5)
        else:
            result[col] = 0.5

    result["consensus_diff"] = result["expert_consensus_a"] - result["expert_consensus_b"]

    return result


def generate_synthetic_fight_data(n_fights=500):
    """Generate realistic synthetic UFC fight data for testing."""
    import random
    random.seed(42)
    np.random.seed(42)

    fighters = [
        "Jon Jones", "Islam Makhachev", "Alexander Volkanovski", "Israel Adesanya",
        "Alex Pereira", "Charles Oliveira", "Leon Edwards", "Max Holloway",
        "Dustin Poirier", "Justin Gaethje", "Sean O'Malley", "Ilia Topuria",
        "Sean Strickland", "Jiri Prochazka", "Khamzat Chimaev", "Tom Aspinall",
        "Ciryl Gane", "Brandon Moreno", "Colby Covington", "Robert Whittaker",
        "Petr Yan", "Cory Sandhagen", "Marlon Vera", "Gilbert Burns",
        "Belal Muhammad", "Shavkat Rakhmonov", "Arman Tsarukyan", "Beneil Dariush",
    ]

    data = []
    for i in range(n_fights):
        fa = random.choice(fighters)
        fb = random.choice([f for f in fighters if f != fa])

        striker_types = ["Jon Jones", "Israel Adesanya", "Max Holloway", "Sean O'Malley", "Israel Adesanya"]
        grappler_types = ["Islam Makhachev", "Khamzat Chimaev", "Charles Oliveira", "Beneil Dariush"]

        fa_sig_str = np.random.gamma(25, 1.5) if fa in striker_types else np.random.gamma(15, 1.2)
        fb_sig_str = np.random.gamma(25, 1.5) if fb in striker_types else np.random.gamma(15, 1.2)

        fa_td = np.random.gamma(3, 0.8) if fa in grappler_types else np.random.gamma(1, 0.5)
        fb_td = np.random.gamma(3, 0.8) if fb in grappler_types else np.random.gamma(1, 0.5)

        fa_ctrl = np.random.gamma(120, 1) if fa in grappler_types else np.random.gamma(30, 1)
        fb_ctrl = np.random.gamma(120, 1) if fb in grappler_types else np.random.gamma(30, 1)

        fa_strength = fa_sig_str * 0.4 + fa_td * 1.5 + fa_ctrl * 0.02 + np.random.normal(0, 2)
        fb_strength = fb_sig_str * 0.4 + fb_td * 1.5 + fb_ctrl * 0.02 + np.random.normal(0, 2)

        winner = "A" if fa_strength > fb_strength else "B"

        rounds = np.random.choice([1, 3, 5], p=[0.15, 0.60, 0.25])
        method = np.random.choice(["KO/TKO", "Submission", "Decision"], p=[0.35, 0.20, 0.45])

        for r in range(1, rounds + 1):
            fatigue_factor = max(0.7, 1.0 - (r - 1) * 0.1 - np.random.uniform(0, 0.05))
            round_data = {
                "fighter_a": fa,
                "fighter_b": fb,
                "winner": winner,
                "round": r,
                "method_type": method,
                "a_sig_str": round(fa_sig_str * fatigue_factor * np.random.uniform(0.8, 1.2)),
                "b_sig_str": round(fb_sig_str * fatigue_factor * np.random.uniform(0.8, 1.2)),
                "a_total_str": round(fa_sig_str * fatigue_factor * 1.5 * np.random.uniform(0.8, 1.2)),
                "b_total_str": round(fb_sig_str * fatigue_factor * 1.5 * np.random.uniform(0.8, 1.2)),
                "a_td": round(fa_td * fatigue_factor * np.random.uniform(0, 1.5)),
                "b_td": round(fb_td * fatigue_factor * np.random.uniform(0, 1.5)),
                "a_sub_att": round(np.random.uniform(0, 1.5) if fa in grappler_types else np.random.uniform(0, 0.5)),
                "b_sub_att": round(np.random.uniform(0, 1.5) if fb in grappler_types else np.random.uniform(0, 0.5)),
                "a_ctrl": round(fa_ctrl * fatigue_factor * np.random.uniform(0.5, 1.5)),
                "b_ctrl": round(fb_ctrl * fatigue_factor * np.random.uniform(0.5, 1.5)),
                "a_head": round(fa_sig_str * fatigue_factor * 0.7),
                "b_head": round(fb_sig_str * fatigue_factor * 0.7),
                "a_body": round(fa_sig_str * fatigue_factor * 0.2),
                "b_body": round(fb_sig_str * fatigue_factor * 0.2),
                "a_leg": round(fa_sig_str * fatigue_factor * 0.1),
                "b_leg": round(fb_sig_str * fatigue_factor * 0.1),
                "a_sig_str_pct": round(np.random.uniform(0.35, 0.65), 2),
                "b_sig_str_pct": round(np.random.uniform(0.35, 0.65), 2),
                "a_td_pct": round(np.random.uniform(0.2, 0.6), 2),
                "b_td_pct": round(np.random.uniform(0.2, 0.6), 2),
            }
            data.append(round_data)

    df = pd.DataFrame(data)
    print(f"  Generated {len(df)} synthetic fight records")
    return df


def encode_and_export(features_df, scaler=None):
    """Encode categoricals, normalize numeric features, and export."""
    print("\nEncoding features and preparing for training...")

    df = features_df.copy()

    if "winner" in df.columns:
        df["target"] = (df["winner"] == "A").astype(int)

    if "method" in df.columns:
        method_dummies = pd.get_dummies(df["method"], prefix="method")
        df = pd.concat([df, method_dummies], axis=1)

    drop_cols = ["fighter_a", "fighter_b", "winner", "method", "target"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    feats = df[feature_cols].copy()

    for col in feats.columns:
        if feats[col].dtype == "object":
            feats[col] = pd.to_numeric(feats[col], errors="coerce")

    feats = feats.fillna(feats.median(numeric_only=True))
    feats = feats.replace([np.inf, -np.inf], np.nan)
    feats = feats.fillna(0.0)

    if scaler is None:
        scaler = StandardScaler()

    numeric_cols = feats.select_dtypes(include=[np.number]).columns
    if not hasattr(scaler, "mean_"):
        scaled_array = scaler.fit_transform(feats[numeric_cols])
    else:
        scaled_array = scaler.transform(feats[numeric_cols])

    scaled_df = pd.DataFrame(scaled_array, columns=numeric_cols, index=feats.index)
    scaled_df["target"] = df["target"].values

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(list(numeric_cols), FEATURE_NAMES_PATH)

    scaled_df.to_csv(TRAINING_CSV, index=False)
    print(f"  Saved {len(scaled_df)} training samples to {TRAINING_CSV}")
    print(f"  Features: {len(numeric_cols)}")
    print(f"  Targets: A wins = {scaled_df['target'].sum():.0f}, B wins = {(1 - scaled_df['target']).sum():.0f}")

    return scaled_df, scaler, list(numeric_cols)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    fight_df = load_and_clean_fight_stats()
    profiles_df = load_profiles()

    styled_df = build_style_matchup_features(fight_df, profiles_df)

    if "fighter_a" in fight_df.columns and "fighter_b" in fight_df.columns:
        styled_df["method"] = fight_df["method_type"].values if "method_type" in fight_df.columns else ""

    sentiment_df = build_sentiment_features(styled_df)
    consensus_df = build_expert_consensus_features(sentiment_df)

    training_df, scaler, feature_names = encode_and_export(consensus_df)

    print(f"\nFeature engineering complete!")
    print(f"  Training data: {TRAINING_CSV}")
    print(f"  Scaler: {SCALER_PATH}")
    print(f"  Next step: python scripts/model_training.py")

    return training_df


if __name__ == "__main__":
    main()
