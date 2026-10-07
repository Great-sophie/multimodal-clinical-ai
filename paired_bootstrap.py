from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score


def bootstrap_differences(y_true, prob_a, prob_b, n_bootstrap=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)

    delta_auc = []
    delta_auprc = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y = y_true[idx]

        if len(np.unique(y)) < 2:
            continue

        a = prob_a[idx]
        b = prob_b[idx]

        delta_auc.append(
            roc_auc_score(y, b) - roc_auc_score(y, a)
        )

        delta_auprc.append(
            average_precision_score(y, b)
            - average_precision_score(y, a)
        )

    delta_auc = np.asarray(delta_auc, dtype=float)
    delta_auprc = np.asarray(delta_auprc, dtype=float)

    return {
        "delta_auroc_ci_lower": float(np.percentile(delta_auc, 2.5)),
        "delta_auroc_ci_upper": float(np.percentile(delta_auc, 97.5)),
        "delta_auprc_ci_lower": float(np.percentile(delta_auprc, 2.5)),
        "delta_auprc_ci_upper": float(np.percentile(delta_auprc, 97.5)),
        "n_valid_bootstrap": int(len(delta_auc)),
    }


def load_and_align(path_a, path_b):
    a = pd.read_csv(path_a)
    b = pd.read_csv(path_b)

    required = {"image_id", "lesion_id", "y_true", "y_prob"}

    for path, df in [(path_a, a), (path_b, b)]:
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")

    a = a.rename(columns={"y_prob": "y_prob_a"})
    b = b.rename(columns={"y_prob": "y_prob_b"})

    merged = a.merge(
        b[["image_id", "lesion_id", "y_true", "y_prob_b"]],
        on=["image_id", "lesion_id", "y_true"],
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(a) or len(merged) != len(b):
        raise ValueError("Prediction files are not aligned to the same test samples.")

    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--name-a", required=True)
    parser.add_argument("--name-b", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path.",
    )
    args = parser.parse_args()

    merged = load_and_align(args.a, args.b)

    y = merged["y_true"].to_numpy(dtype=int)
    pa = merged["y_prob_a"].to_numpy(dtype=float)
    pb = merged["y_prob_b"].to_numpy(dtype=float)

    auc_a = roc_auc_score(y, pa)
    auc_b = roc_auc_score(y, pb)
    auprc_a = average_precision_score(y, pa)
    auprc_b = average_precision_score(y, pb)

    result = {
        "model_a": args.name_a,
        "model_b": args.name_b,
        "difference_definition": "model_b - model_a",
        "n": int(len(y)),
        "auroc_a": float(auc_a),
        "auroc_b": float(auc_b),
        "delta_auroc": float(auc_b - auc_a),
        "auprc_a": float(auprc_a),
        "auprc_b": float(auprc_b),
        "delta_auprc": float(auprc_b - auprc_a),
        **bootstrap_differences(
            y,
            pa,
            pb,
            n_bootstrap=args.bootstrap,
            seed=args.seed,
        ),
    }

    print(f"{args.name_b} - {args.name_a}")
    print(
        "delta_AUROC: "
        f"{result['delta_auroc']:+.4f} "
        f"(95% CI {result['delta_auroc_ci_lower']:+.4f} "
        f"to {result['delta_auroc_ci_upper']:+.4f})"
    )
    print(
        "delta_AUPRC: "
        f"{result['delta_auprc']:+.4f} "
        f"(95% CI {result['delta_auprc_ci_lower']:+.4f} "
        f"to {result['delta_auprc_ci_upper']:+.4f})"
    )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
