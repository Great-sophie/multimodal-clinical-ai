from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUTS = Path("outputs")
FIGURES = Path("figures")
FIGURES.mkdir(parents=True, exist_ok=True)


MODEL_ORDER = ["image", "tabular", "late", "cross"]
MODEL_LABELS = {
    "image": "Image-only",
    "tabular": "Tabular-only",
    "late": "Late fusion",
    "cross": "Cross-attention",
}


def load_metric(model):
    path = OUTPUTS / f"{model}_test_metrics.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def build_model_table():
    rows = []

    for model in MODEL_ORDER:
        m = load_metric(model)
        rows.append(
            {
                "model": MODEL_LABELS[model],
                "AUROC": m["auroc"],
                "AUPRC": m["auprc"],
                "F1": m["f1"],
                "Balanced accuracy": m["balanced_accuracy"],
                "Sensitivity": m["sensitivity"],
                "Specificity": m["specificity"],
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUTS / "final_model_comparison.csv", index=False)
    return df


def build_bootstrap_table():
    files = [
        OUTPUTS / "bootstrap_late_vs_image.json",
        OUTPUTS / "bootstrap_cross_vs_image.json",
        OUTPUTS / "bootstrap_cross_vs_late.json",
    ]

    rows = []

    for path in files:
        if not path.exists():
            raise FileNotFoundError(path)

        x = json.loads(path.read_text())

        rows.append(
            {
                "comparison": f"{x['model_b']} - {x['model_a']}",
                "delta_AUROC": x["delta_auroc"],
                "AUROC_CI_low": x["delta_auroc_ci_lower"],
                "AUROC_CI_high": x["delta_auroc_ci_upper"],
                "delta_AUPRC": x["delta_auprc"],
                "AUPRC_CI_low": x["delta_auprc_ci_lower"],
                "AUPRC_CI_high": x["delta_auprc_ci_upper"],
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUTS / "bootstrap_comparison.csv", index=False)
    return df


def plot_discrimination(df):
    labels = df["model"].tolist()
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, df["AUROC"], width, label="AUROC")
    ax.bar(x + width / 2, df["AUPRC"], width, label="AUPRC")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Held-out test discrimination")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "01_discrimination_comparison.png", dpi=200)
    plt.close(fig)


def plot_operating_metrics(df):
    labels = df["model"].tolist()
    x = np.arange(len(labels))
    width = 0.22

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, df["Sensitivity"], width, label="Sensitivity")
    ax.bar(x, df["Specificity"], width, label="Specificity")
    ax.bar(x + width, df["F1"], width, label="F1")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Operating-point metrics at threshold 0.5")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "02_operating_metrics.png", dpi=200)
    plt.close(fig)


def plot_delta(df, metric, low_col, high_col, filename, title):
    labels = df["comparison"].tolist()
    values = df[metric].to_numpy()
    low = df[low_col].to_numpy()
    high = df[high_col].to_numpy()

    yerr = np.vstack([values - low, high - values])

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.errorbar(
        np.arange(len(labels)),
        values,
        yerr=yerr,
        fmt="o",
        capsize=5,
    )
    ax.axhline(0, linewidth=1)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Difference")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=200)
    plt.close(fig)


def write_markdown(model_df, bootstrap_df):
    lines = []
    lines.append("# Day 4–5 Final Results\n")
    lines.append("## Held-out test performance\n")
    lines.append(model_df.to_markdown(index=False, floatfmt=".4f"))
    lines.append("\n## Paired bootstrap comparisons\n")
    lines.append(bootstrap_df.to_markdown(index=False, floatfmt=".4f"))
    lines.append(
        "\n## Interpretation\n\n"
        "Clinical metadata showed independent predictive signal, but neither "
        "late fusion nor cross-attention demonstrated a clear improvement over "
        "the image-only baseline. All paired-bootstrap 95% confidence intervals "
        "for ΔAUROC and ΔAUPRC crossed zero.\n"
    )
    (OUTPUTS / "FINAL_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    model_df = build_model_table()
    bootstrap_df = build_bootstrap_table()

    plot_discrimination(model_df)
    plot_operating_metrics(model_df)

    plot_delta(
        bootstrap_df,
        "delta_AUROC",
        "AUROC_CI_low",
        "AUROC_CI_high",
        "03_bootstrap_delta_auroc.png",
        "Paired bootstrap ΔAUROC (model B - model A)",
    )

    plot_delta(
        bootstrap_df,
        "delta_AUPRC",
        "AUPRC_CI_low",
        "AUPRC_CI_high",
        "04_bootstrap_delta_auprc.png",
        "Paired bootstrap ΔAUPRC (model B - model A)",
    )

    write_markdown(model_df, bootstrap_df)

    print(model_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print(bootstrap_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSaved final tables and figures.")


if __name__ == "__main__":
    main()
