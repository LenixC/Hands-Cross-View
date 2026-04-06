import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import argparse


EXPERIMENTS = {
    "resnet_contrastive": {
        "name": "ResNet50 + Contrastive",
        "checkpoint_dir": "./checkpoints",
        "color": "#1f77b4",
    },
    "resnet_triplet": {
        "name": "ResNet50 + Triplet+HN",
        "checkpoint_dir": "./checkpoints_resnet_triplet",
        "color": "#ff7f0e",
    },
    "dino_contrastive": {
        "name": "DINOv2 + Contrastive",
        "checkpoint_dir": "./checkpoints_dinov2_contrastive",
        "color": "#2ca02c",
    },
    "dino_triplet": {
        "name": "DINOv2 + Triplet+HN",
        "checkpoint_dir": "./checkpoints_dinov2",
        "color": "#d62728",
    },
}


def load_results(experiment_key: str) -> Dict:
    """Load test results from experiment"""
    exp = EXPERIMENTS[experiment_key]
    results_path = Path(exp["checkpoint_dir"]) / "test_results.json"

    if not results_path.exists():
        print(f"Warning: {results_path} not found")
        return None

    with open(results_path) as f:
        return json.load(f)


def load_training_history(experiment_key: str) -> Dict:
    """Load training history if available"""
    exp = EXPERIMENTS[experiment_key]
    history_path = Path(exp["checkpoint_dir"]) / "training_history.json"

    if not history_path.exists():
        return None

    with open(history_path) as f:
        return json.load(f)


def create_comparison_table() -> pd.DataFrame:
    """Create comparison table across all experiments"""
    rows = []

    for key, exp in EXPERIMENTS.items():
        results = load_results(key)

        if results is None:
            continue

        # Handle both formats: direct metrics (contrastive) and nested (triplet)
        if "test_metrics" in results:
            metrics = results["test_metrics"]
        elif "metrics" in results:
            metrics = results["metrics"]
        else:
            metrics = results  # Direct format from contrastive

        rows.append(
            {
                "Experiment": exp["name"],
                "Mean Rank": f"{metrics.get('mean_rank', float('nan')):.2f}",
                "R@1": f"{metrics.get('recall@1', 0):.1%}",
                "R@5": f"{metrics.get('recall@5', 0):.1%}",
                "R@10": f"{metrics.get('recall@10', 0):.1%}",
            }
        )

    return pd.DataFrame(rows)


def plot_comparison_bars(ax, metrics: List[str], title: str):
    """Plot bar chart comparing metrics across experiments"""
    data = {}

    for key, exp in EXPERIMENTS.items():
        results = load_results(key)
        if results is None:
            continue

        # Handle both formats
        if "test_metrics" in results:
            test_metrics = results["test_metrics"]
        elif "metrics" in results:
            test_metrics = results["metrics"]
        else:
            test_metrics = results

        data[exp["name"]] = [test_metrics.get(m, 0) for m in metrics]

    if not data:
        ax.text(0.5, 0.5, "No results found", ha="center", va="center")
        return

    x = np.arange(len(metrics))
    width = 0.2
    offset = 0

    for name, values in data.items():
        color = EXPERIMENTS[
            [k for k, v in EXPERIMENTS.items() if v["name"] == name][0]
        ]["color"]
        ax.bar(x + offset, values, width, label=name, color=color)
        offset += width

    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(metrics)
    ax.legend(loc="upper left", fontsize=8)

    # Different scaling for mean_rank vs recall metrics
    if "mean_rank" in metrics:
        ax.set_ylim(0, max(sum(data.values(), [])) * 1.2 if data else 15)
    else:
        ax.set_ylim(0, 1.1)


def plot_recall_curves(ax):
    """Plot recall@k curves for all experiments"""
    for key, exp in EXPERIMENTS.items():
        results = load_results(key)
        if results is None:
            continue

        test_metrics = results.get("test_metrics") or results.get("metrics") or results

        k_values = [1, 5, 10]
        recall_values = [
            test_metrics.get("recall@1", 0),
            test_metrics.get("recall@5", 0),
            test_metrics.get("recall@10", 0),
        ]

        ax.plot(
            k_values,
            recall_values,
            "o-",
            label=exp["name"],
            color=exp["color"],
            linewidth=2,
            markersize=8,
        )

    ax.set_xlabel("k")
    ax.set_ylabel("Recall@k")
    ax.set_title("Recall@k Comparison")
    ax.set_xticks([1, 5, 10])
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)


def plot_mean_rank_comparison(ax):
    """Plot mean rank comparison"""
    names = []
    ranks = []
    colors = []

    for key, exp in EXPERIMENTS.items():
        results = load_results(key)
        if results is None:
            continue

        test_metrics = results.get("test_metrics") or results.get("metrics") or results
        names.append(exp["name"])
        ranks.append(test_metrics.get("mean_rank", 0))
        colors.append(exp["color"])

    if not names:
        ax.text(0.5, 0.5, "No results found", ha="center", va="center")
        return

    bars = ax.barh(names, ranks, color=colors)
    ax.set_xlabel("Mean Rank (lower is better)")
    ax.set_title("Mean Rank Comparison")
    ax.set_xlim(0, max(ranks) * 1.2)  # Scale based on actual data

    for bar, rank in zip(bars, ranks):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{rank:.1f}",
            va="center",
            fontsize=10,
        )


def plot_training_curves(ax):
    """Plot training loss curves"""
    has_data = False

    for key, exp in EXPERIMENTS.items():
        history = load_training_history(key)
        if history is None:
            continue

        has_data = True
        train_loss = history.get("train_loss", [])
        val_loss = history.get("val_loss", [])

        # For triplet experiments, use val_mean_rank as validation curve
        if not val_loss and "val_mean_rank" in history:
            val_loss = history.get("val_mean_rank", [])

        epochs = range(1, len(train_loss) + 1)

        # Train = dotted, Val = solid (same color)
        ax.plot(
            epochs,
            train_loss,
            ":",
            label=f"{exp['name']} (train)",
            color=exp["color"],
            alpha=0.8,
        )

        if val_loss:
            # Triplet experiments log val every 5 epochs (starting from epoch 0)
            # Use indices directly, scaled to training epochs
            val_epochs = np.linspace(1, len(train_loss), len(val_loss))
            ax.plot(
                val_epochs,
                val_loss,
                "-",
                label=f"{exp['name']} (val)",
                color=exp["color"],
                alpha=0.8,
            )

    if not has_data:
        ax.text(
            0.5,
            0.5,
            "No training history found\nRun experiments with --log-history",
            ha="center",
            va="center",
        )
        return

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Curves")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_recall_over_training(ax):
    """Plot recall@1 over training epochs"""
    has_data = False

    for key, exp in EXPERIMENTS.items():
        history = load_training_history(key)
        if history is None:
            continue

        val_recall = history.get("val_recall@1", [])
        if not val_recall:
            continue

        has_data = True
        train_loss = history.get("train_loss", [])

        # Scale val epochs to match training epochs
        val_epochs = np.linspace(1, len(train_loss), len(val_recall))
        ax.plot(
            val_epochs,
            val_recall,
            "-",
            label=exp["name"],
            color=exp["color"],
            linewidth=2,
            marker="o",
            markersize=4,
        )

    if not has_data:
        ax.text(
            0.5,
            0.5,
            "No training history found",
            ha="center",
            va="center",
        )
        return

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Recall@1")
    ax.set_title("Recall@1 Over Training")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)


def create_ablation_summary(ax):
    """Create heatmap showing all 4 experiment combinations"""
    results = {}

    for key, exp in EXPERIMENTS.items():
        r = load_results(key)
        if r:
            results[key] = r.get("test_metrics") or r.get("metrics") or r

    if len(results) < 4:
        ax.text(
            0.5,
            0.5,
            "Need all 4 experiments for ablation heatmap",
            ha="center",
            va="center",
        )
        return

    row_labels = ["ResNet50", "DINOv2"]
    col_labels = ["Contrastive", "Triplet+HN"]

    data = np.zeros((2, 2))
    data[0, 0] = results.get("resnet_contrastive", {}).get("recall@1", 0) * 100
    data[0, 1] = results.get("resnet_triplet", {}).get("recall@1", 0) * 100
    data[1, 0] = results.get("dino_contrastive", {}).get("recall@1", 0) * 100
    data[1, 1] = results.get("dino_triplet", {}).get("recall@1", 0) * 100

    im = ax.imshow(data, cmap="YlGn", vmin=0, vmax=70)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            text = ax.text(
                j,
                i,
                f"{data[i, j]:.1f}%",
                ha="center",
                va="center",
                color="black",
                fontsize=12,
            )

    ax.set_title("Recall@1 Heatmap", pad=10)

    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Recall@1 (%)", rotation=270, labelpad=20)


def main():
    parser = argparse.ArgumentParser(description="Visualize experiment results")
    parser.add_argument(
        "--output", "-o", default="experiment_comparison.png", help="Output filename"
    )
    parser.add_argument("--show", action="store_true", help="Show plot interactively")
    args = parser.parse_args()

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        "Cross-View Hand Matching: Experiment Comparison",
        fontsize=16,
        fontweight="bold",
    )

    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)
    ax4 = fig.add_subplot(2, 3, 4)
    ax5 = fig.add_subplot(2, 3, 5)
    ax6 = fig.add_subplot(2, 3, 6)

    print("Creating comparison table...")
    table = create_comparison_table()
    if not table.empty:
        print("\n" + table.to_string(index=False))

    print("\nGenerating plots...")
    plot_recall_curves(ax1)
    plot_mean_rank_comparison(ax2)
    plot_training_curves(ax3)
    create_ablation_summary(ax4)

    plot_comparison_bars(ax5, ["recall@1", "recall@5", "recall@10"], "Recall Metrics")
    plot_recall_over_training(ax6)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nSaved visualization to {args.output}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
