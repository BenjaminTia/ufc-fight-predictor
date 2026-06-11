"""
UFC Fight Predictor CLI
Predict the outcome of a specific UFC matchup using the trained ensemble model.

Usage:
    python scripts/predict_fight.py --fighter_a "Islam Makhachev" --fighter_b "Charles Oliveira"
    python scripts/predict_fight.py -a "Jon Jones" -b "Tom Aspinall" --verbose
    python scripts/predict_fight.py --list-fighters
"""

import os
import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

DATA_DIR = Path(__file__).parent.parent / "data"

KNOWN_FIGHTERS = {
    "Jon Jones": {"slpm": 4.29, "sapm": 2.22, "td_avg": 1.93, "td_def": 95, "strike_acc": 57, "strike_def": 65,
                  "sub_avg": 0.5, "win_rate": 0.95, "height_inches": 76, "reach_inches": 84.5, "num_fights": 28,
                  "momentum": 0.8, "sentiment": 0.6},
    "Islam Makhachev": {"slpm": 3.51, "sapm": 1.45, "td_avg": 3.17, "td_def": 89, "strike_acc": 62, "strike_def": 67,
                        "sub_avg": 0.7, "win_rate": 0.94, "height_inches": 70, "reach_inches": 72.0, "num_fights": 26,
                        "momentum": 0.9, "sentiment": 0.7},
    "Alexander Volkanovski": {"slpm": 6.23, "sapm": 3.35, "td_avg": 1.72, "td_def": 72, "strike_acc": 58, "strike_def": 63,
                              "sub_avg": 0.4, "win_rate": 0.89, "height_inches": 66, "reach_inches": 71.5, "num_fights": 29,
                              "momentum": 0.3, "sentiment": 0.4},
    "Israel Adesanya": {"slpm": 4.93, "sapm": 3.77, "td_avg": 0.08, "td_def": 77, "strike_acc": 51, "strike_def": 61,
                        "sub_avg": 0.1, "win_rate": 0.83, "height_inches": 76, "reach_inches": 80.0, "num_fights": 27,
                        "momentum": -0.2, "sentiment": -0.2},
    "Alex Pereira": {"slpm": 5.26, "sapm": 4.33, "td_avg": 0.00, "td_def": 70, "strike_acc": 62, "strike_def": 54,
                     "sub_avg": 0.0, "win_rate": 0.81, "height_inches": 76, "reach_inches": 79.0, "num_fights": 11,
                     "momentum": 0.7, "sentiment": 0.6},
    "Charles Oliveira": {"slpm": 3.45, "sapm": 3.41, "td_avg": 2.88, "td_def": 63, "strike_acc": 54, "strike_def": 51,
                         "sub_avg": 1.4, "win_rate": 0.79, "height_inches": 70, "reach_inches": 74.0, "num_fights": 40,
                         "momentum": 0.4, "sentiment": 0.3},
    "Leon Edwards": {"slpm": 2.74, "sapm": 2.89, "td_avg": 1.60, "td_def": 68, "strike_acc": 48, "strike_def": 55,
                     "sub_avg": 0.3, "win_rate": 0.85, "height_inches": 74, "reach_inches": 74.0, "num_fights": 22,
                     "momentum": 0.5, "sentiment": 0.5},
    "Max Holloway": {"slpm": 7.24, "sapm": 4.75, "td_avg": 0.20, "td_def": 84, "strike_acc": 51, "strike_def": 60,
                     "sub_avg": 0.1, "win_rate": 0.79, "height_inches": 71, "reach_inches": 69.0, "num_fights": 30,
                     "momentum": 0.6, "sentiment": 0.5},
    "Dustin Poirier": {"slpm": 5.57, "sapm": 4.38, "td_avg": 1.70, "td_def": 62, "strike_acc": 50, "strike_def": 52,
                       "sub_avg": 0.7, "win_rate": 0.77, "height_inches": 69, "reach_inches": 72.0, "num_fights": 35,
                       "momentum": 0.2, "sentiment": 0.2},
    "Justin Gaethje": {"slpm": 7.70, "sapm": 7.99, "td_avg": 0.11, "td_def": 75, "strike_acc": 55, "strike_def": 53,
                       "sub_avg": 0.1, "win_rate": 0.75, "height_inches": 71, "reach_inches": 70.0, "num_fights": 28,
                       "momentum": 0.3, "sentiment": 0.1},
    "Sean O'Malley": {"slpm": 6.79, "sapm": 3.75, "td_avg": 0.00, "td_def": 70, "strike_acc": 62, "strike_def": 62,
                      "sub_avg": 0.0, "win_rate": 0.88, "height_inches": 71, "reach_inches": 72.0, "num_fights": 18,
                      "momentum": 0.7, "sentiment": 0.7},
    "Ilia Topuria": {"slpm": 4.91, "sapm": 3.47, "td_avg": 1.36, "td_def": 74, "strike_acc": 52, "strike_def": 56,
                     "sub_avg": 0.6, "win_rate": 1.0, "height_inches": 67, "reach_inches": 69.0, "num_fights": 15,
                     "momentum": 0.95, "sentiment": 0.9},
    "Sean Strickland": {"slpm": 5.94, "sapm": 4.03, "td_avg": 0.82, "td_def": 74, "strike_acc": 41, "strike_def": 68,
                        "sub_avg": 0.0, "win_rate": 0.75, "height_inches": 73, "reach_inches": 76.0, "num_fights": 30,
                        "momentum": -0.1, "sentiment": 0.0},
    "Jiri Prochazka": {"slpm": 5.55, "sapm": 5.08, "td_avg": 0.33, "td_def": 60, "strike_acc": 54, "strike_def": 44,
                       "sub_avg": 0.3, "win_rate": 0.88, "height_inches": 76, "reach_inches": 80.0, "num_fights": 31,
                       "momentum": 0.5, "sentiment": 0.4},
    "Khamzat Chimaev": {"slpm": 4.82, "sapm": 1.92, "td_avg": 4.29, "td_def": 60, "strike_acc": 62, "strike_def": 50,
                        "sub_avg": 0.4, "win_rate": 1.0, "height_inches": 74, "reach_inches": 75.0, "num_fights": 13,
                        "momentum": 0.7, "sentiment": 0.6},
    "Tom Aspinall": {"slpm": 7.63, "sapm": 3.67, "td_avg": 2.42, "td_def": 100, "strike_acc": 53, "strike_def": 63,
                    "sub_avg": 0.6, "win_rate": 0.92, "height_inches": 77, "reach_inches": 78.0, "num_fights": 18,
                    "momentum": 0.85, "sentiment": 0.8},
    "Ciryl Gane": {"slpm": 5.08, "sapm": 2.51, "td_avg": 0.76, "td_def": 63, "strike_acc": 60, "strike_def": 62,
                   "sub_avg": 0.3, "win_rate": 0.84, "height_inches": 76, "reach_inches": 83.0, "num_fights": 15,
                   "momentum": 0.1, "sentiment": 0.2},
    "Colby Covington": {"slpm": 4.16, "sapm": 3.35, "td_avg": 4.12, "td_def": 72, "strike_acc": 36, "strike_def": 56,
                        "sub_avg": 0.4, "win_rate": 0.79, "height_inches": 71, "reach_inches": 72.0, "num_fights": 20,
                        "momentum": -0.3, "sentiment": -0.3},
    "Robert Whittaker": {"slpm": 4.78, "sapm": 3.62, "td_avg": 1.62, "td_def": 85, "strike_acc": 42, "strike_def": 61,
                         "sub_avg": 0.1, "win_rate": 0.81, "height_inches": 72, "reach_inches": 73.5, "num_fights": 30,
                         "momentum": 0.2, "sentiment": 0.1},
    "Arman Tsarukyan": {"slpm": 4.42, "sapm": 2.86, "td_avg": 3.62, "td_def": 72, "strike_acc": 46, "strike_def": 58,
                        "sub_avg": 0.5, "win_rate": 0.83, "height_inches": 67, "reach_inches": 72.0, "num_fights": 23,
                        "momentum": 0.6, "sentiment": 0.5},
    "Brandon Moreno": {"slpm": 4.41, "sapm": 3.59, "td_avg": 1.71, "td_def": 64, "strike_acc": 43, "strike_def": 55,
                       "sub_avg": 0.7, "win_rate": 0.72, "height_inches": 67, "reach_inches": 70.0, "num_fights": 27,
                       "momentum": -0.1, "sentiment": 0.0},
    "Shavkat Rakhmonov": {"slpm": 4.20, "sapm": 2.38, "td_avg": 1.63, "td_def": 100, "strike_acc": 51, "strike_def": 58,
                          "sub_avg": 1.0, "win_rate": 1.0, "height_inches": 73, "reach_inches": 77.0, "num_fights": 18,
                          "momentum": 0.9, "sentiment": 0.8},
    "Petr Yan": {"slpm": 5.62, "sapm": 3.99, "td_avg": 1.77, "td_def": 88, "strike_acc": 47, "strike_def": 60,
                 "sub_avg": 0.1, "win_rate": 0.70, "height_inches": 67, "reach_inches": 66.5, "num_fights": 20,
                 "momentum": -0.2, "sentiment": 0.0},
    "Cory Sandhagen": {"slpm": 5.49, "sapm": 3.85, "td_avg": 1.40, "td_def": 64, "strike_acc": 41, "strike_def": 66,
                       "sub_avg": 0.3, "win_rate": 0.76, "height_inches": 71, "reach_inches": 70.0, "num_fights": 19,
                       "momentum": 0.3, "sentiment": 0.2},
    "Gilbert Burns": {"slpm": 2.72, "sapm": 3.32, "td_avg": 2.55, "td_def": 42, "strike_acc": 45, "strike_def": 52,
                      "sub_avg": 0.7, "win_rate": 0.72, "height_inches": 69, "reach_inches": 71.0, "num_fights": 27,
                      "momentum": -0.2, "sentiment": 0.0},
    "Belal Muhammad": {"slpm": 4.66, "sapm": 3.41, "td_avg": 1.94, "td_def": 100, "strike_acc": 51, "strike_def": 56,
                       "sub_avg": 0.1, "win_rate": 0.80, "height_inches": 70, "reach_inches": 72.0, "num_fights": 24,
                       "momentum": 0.8, "sentiment": 0.6},
}


def list_fighters():
    """Print all known fighters."""
    print(f"\n  Available Fighters ({len(KNOWN_FIGHTERS)}):")
    print("  " + "-" * 60)
    for name, stats in sorted(KNOWN_FIGHTERS.items()):
        momentum = stats.get("momentum", 0)
        momentum_indicator = "+" if momentum > 0.3 else "-" if momentum < -0.3 else "~"
        record = f"{stats.get('win_rate', 0)*100:.0f}% win rate" if stats.get("win_rate") else ""
        print(f"  {momentum_indicator} {name:<25} {record}")
    print()


def format_prediction(result, fighter_a, fighter_b):
    """Pretty-print prediction results."""
    print("\n" + "=" * 60)
    print("  UFC FIGHT PREDICTION")
    print("=" * 60)
    print(f"\n  {fighter_a} vs {fighter_b}")
    print(f"  {'-' * 40}")

    prob_a = result["fighter_a_win_probability"]
    prob_b = result["fighter_b_win_probability"]

    bar_width = 36
    a_bars = int(bar_width * prob_a)
    b_bars = bar_width - a_bars

    bar_a = "#" * a_bars
    bar_b = "#" * b_bars

    print(f"\n  {fighter_a:<22} {bar_a} {prob_a:.1%}")
    print(f"  {fighter_b:<22} {bar_b} {prob_b:.1%}")

    print(f"\n  {'Predicted Winner:':<22} {result['predicted_winner']}")
    print(f"  {'Confidence:':<22} {result['confidence']:.1%}")

    individual = result.get("individual_predictions", {})
    if individual:
        print(f"\n  Individual Model Predictions (Fighter A win probability):")
        print(f"  {'XGBoost:':<22} {individual['xgb_fighter_a_prob']:.1%}")
        print(f"  {'LightGBM:':<22} {individual['lgb_fighter_a_prob']:.1%}")
        print(f"  {'Neural Net:':<22} {individual['nn_fighter_a_prob']:.1%}")

    agreement = result.get("model_agreement", 0)
    print(f"\n  {'Model Agreement:':<22} {agreement:.1%} ({'High' if agreement > 0.85 else 'Moderate' if agreement > 0.7 else 'Low - models disagree'})")

    print("\n" + "=" * 60)


def get_fighter_features(fighter_name):
    """Get feature dict for a fighter, with graceful fuzzy matching."""
    if fighter_name in KNOWN_FIGHTERS:
        return KNOWN_FIGHTERS[fighter_name]

    name_lower = fighter_name.lower()
    for known_name, stats in KNOWN_FIGHTERS.items():
        if name_lower in known_name.lower() or known_name.lower() in name_lower:
            print(f"  (Matched '{fighter_name}' to '{known_name}')")
            return stats

    print(f"  WARNING: '{fighter_name}' not found in database. Using generic fighter profile.")
    return {
        "slpm": 3.5, "sapm": 3.5, "td_avg": 1.5, "td_def": 65,
        "strike_acc": 48, "strike_def": 55, "sub_avg": 0.3,
        "win_rate": 0.70, "height_inches": 71, "reach_inches": 72.0,
        "num_fights": 20, "momentum": 0.0, "sentiment": 0.0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="UFC Fight Predictor - Predict fight outcomes using ML ensemble",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict_fight.py -a "Islam Makhachev" -b "Charles Oliveira"
  python predict_fight.py -a "Jon Jones" -b "Tom Aspinall" --verbose --json
  python predict_fight.py --list-fighters
        """,
    )
    parser.add_argument("-a", "--fighter_a", type=str, help="Name of Fighter A")
    parser.add_argument("-b", "--fighter_b", type=str, help="Name of Fighter B")
    parser.add_argument("--list-fighters", action="store_true", help="List all available fighters")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    parser.add_argument("--json", action="store_true", help="Output result as JSON")
    parser.add_argument("--model-dir", type=str, default=None, help="Custom model directory path")

    args = parser.parse_args()

    if args.list_fighters:
        list_fighters()
        return

    if not args.fighter_a or not args.fighter_b:
        parser.print_help()
        print("\n  Please specify both fighters. Examples:")
        print('    python scripts/predict_fight.py -a "Islam Makhachev" -b "Charles Oliveira"')
        print('    python scripts/predict_fight.py --list-fighters')
        return

    features_a = get_fighter_features(args.fighter_a)
    features_b = get_fighter_features(args.fighter_b)

    try:
        from inference import UFCPredictor
        predictor = UFCPredictor(model_dir=args.model_dir)
    except (FileNotFoundError, ImportError, ModuleNotFoundError):
        print("\n  Models not found or dependencies missing. Using fallback heuristic.")
        print("  For full ML predictions, run: conda activate ufc_predictor && pip install -r requirements.txt")
        features_a["name"] = args.fighter_a
        features_b["name"] = args.fighter_b

        a_score = (
            features_a.get("slpm", 3.5) * 0.15
            - features_a.get("sapm", 3.5) * 0.10
            + features_a.get("td_avg", 1.5) * 0.15
            + features_a.get("strike_acc", 48) * 0.10
            + features_a.get("strike_def", 55) * 0.10
            + features_a.get("td_def", 65) * 0.10
            + features_a.get("sub_avg", 0.3) * 0.10
            + features_a.get("win_rate", 0.7) * 0.10
            + features_a.get("momentum", 0) * 0.10
        )
        b_score = (
            features_b.get("slpm", 3.5) * 0.15
            - features_b.get("sapm", 3.5) * 0.10
            + features_b.get("td_avg", 1.5) * 0.15
            + features_b.get("strike_acc", 48) * 0.10
            + features_b.get("strike_def", 55) * 0.10
            + features_b.get("td_def", 65) * 0.10
            + features_b.get("sub_avg", 0.3) * 0.10
            + features_b.get("win_rate", 0.7) * 0.10
            + features_b.get("momentum", 0) * 0.10
        )
        total = a_score + b_score + 1e-9
        prob_a = a_score / total

        result = {
            "fighter_a_win_probability": prob_a,
            "fighter_b_win_probability": 1 - prob_a,
            "predicted_winner": args.fighter_a if prob_a > 0.5 else args.fighter_b,
            "confidence": abs(prob_a - 0.5) * 2,
            "individual_predictions": {},
            "model_agreement": 0.0,
        }
    else:
        result = predictor.predict_proba(args.fighter_a, args.fighter_b, features_a, features_b)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        format_prediction(result, args.fighter_a, args.fighter_b)


if __name__ == "__main__":
    main()
