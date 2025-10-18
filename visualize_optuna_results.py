#!/usr/bin/env python3
"""
Optuna Results Visualization Script
Loads HPO results and creates matplotlib visualizations
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

import json

import matplotlib.pyplot as plt
import optuna


def load_optuna_results():
    """Load Optuna study and results."""

    # Paths
    hpo_dir = Path(
        "C:/Users/pbori/Documents/Coureses/MIT/Projects/Capstone Projects/FacialEmotionRecognition/runs/hpo"
    )
    best_params_file = hpo_dir / "best_parameters.json"
    storage_file = hpo_dir / "optuna_baseline_refinement.db"

    print("=" * 80)
    print("LOADING OPTUNA RESULTS")
    print("=" * 80)

    # Load best parameters
    with open(best_params_file) as f:
        best_data = json.load(f)

    print(f"[OK] Best parameters loaded from: {best_params_file}")
    print(f"[OK] Best validation accuracy: {best_data['best_accuracy']:.4f}")
    print(f"[OK] Best trial number: {best_data['best_trial_number']}")

    # Load study from database
    study_name = "fer_baseline_refinement"
    storage_url = f"sqlite:///{storage_file}"

    try:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
        print(f"[OK] Study loaded: {study_name}")
        print(f"[OK] Total trials: {len(study.trials)}")
        print(f"[OK] Best value: {study.best_value:.4f}")
        return study, best_data
    except Exception as e:
        print(f"[ERROR] Error loading study: {e}")
        return None, best_data


def create_visualizations(study, best_data):
    """Create matplotlib visualizations."""

    viz_dir = Path(
        "C:/Users/pbori/Documents/Coureses/MIT/Projects/Capstone Projects/FacialEmotionRecognition/runs/hpo/visualizations"
    )
    viz_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)

    # Get trial data
    trials_df = study.trials_dataframe()

    # 1. Optimization History
    plt.figure(figsize=(12, 6))
    plt.plot(
        trials_df["number"],
        -trials_df["value"],
        "o-",
        linewidth=2,
        markersize=6,
        alpha=0.7,
    )
    plt.axhline(
        y=-study.best_value,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Best: {-study.best_value:.4f}",
    )
    plt.xlabel("Trial Number", fontsize=12)
    plt.ylabel("Validation Accuracy", fontsize=12)
    plt.title("Optuna Optimization History", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(viz_dir / "optimization_history.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("[OK] Optimization history saved")

    # 2. Parameter distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Hyperparameter Values Across Trials", fontsize=14, fontweight="bold")

    params_to_plot = [
        "dropout",
        "l2_reg",
        "learning_rate",
        "batch_size",
        "channels_multiplier",
    ]
    for idx, param in enumerate(params_to_plot):
        ax = axes[idx // 3, idx % 3]
        col_name = f"params_{param}"
        if col_name in trials_df.columns:
            ax.scatter(trials_df["number"], trials_df[col_name], alpha=0.6, s=50)
            ax.set_xlabel("Trial")
            ax.set_ylabel(param)
            ax.grid(True, alpha=0.3)

            # Highlight best trial
            best_trial_num = best_data["best_trial_number"]
            best_val = trials_df[trials_df["number"] == best_trial_num][
                col_name
            ].values[0]
            ax.axhline(
                y=best_val,
                color="r",
                linestyle="--",
                linewidth=2,
                label=f"Best: {best_val:.4f}",
            )
            ax.legend(fontsize=8)

    # Augmentation distribution
    ax = axes[1, 2]
    if "params_augmentation" in trials_df.columns:
        aug_counts = trials_df["params_augmentation"].value_counts()
        ax.bar(aug_counts.index, aug_counts.values, alpha=0.7)
        ax.set_xlabel("Augmentation")
        ax.set_ylabel("Count")
        ax.set_title("Augmentation Distribution")
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(viz_dir / "parameters_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("[OK] Parameters distribution saved")

    # 3. Top trials comparison
    top_trials = trials_df.nlargest(10, "value")[
        [
            "number",
            "value",
            "params_dropout",
            "params_l2_reg",
            "params_learning_rate",
            "params_batch_size",
        ]
    ].copy()
    top_trials["accuracy"] = -top_trials["value"]

    plt.figure(figsize=(12, 6))
    bars = plt.barh(top_trials["number"].astype(str), top_trials["accuracy"], alpha=0.7)

    # Color best trial differently
    best_trial_num = best_data["best_trial_number"]
    best_idx = top_trials[top_trials["number"] == best_trial_num].index
    if len(best_idx) > 0:
        bars[best_idx[0]].set_color("red")

    plt.xlabel("Validation Accuracy", fontsize=12)
    plt.ylabel("Trial Number", fontsize=12)
    plt.title("Top 10 Trials by Accuracy", fontsize=14, fontweight="bold")
    plt.axvline(
        x=-study.best_value,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Best: {-study.best_value:.4f}",
    )
    plt.legend()
    plt.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(viz_dir / "top_trials.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("[OK] Top trials comparison saved")

    print(f"\n[OK] All visualizations saved to: {viz_dir}")

    # Print summary
    print("\n" + "=" * 80)
    print("OPTUNA OPTIMIZATION SUMMARY")
    print("=" * 80)
    print(f"Total trials: {len(study.trials)}")
    print(f"Best validation accuracy: {study.best_value:.4f}")
    print(f"Best trial: #{best_data['best_trial_number']}")
    print(f"Optimization time: {best_data['optimization_time_minutes']:.1f} minutes")
    print("\nBest parameters:")
    for k, v in best_data["best_params"].items():
        print(f"  {k:20s}: {v}")
    print("=" * 80)


def main():
    """Main function."""
    study, best_data = load_optuna_results()

    if study is None:
        print("[ERROR] Could not load study. Exiting.")
        return

    create_visualizations(study, best_data)


if __name__ == "__main__":
    main()
