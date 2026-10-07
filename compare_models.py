import argparse
import json
from pathlib import Path
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    rows = []
    for model_name in ["image", "tabular", "late"]:
        path = output_dir / f"{model_name}_{args.split}_metrics.json"
        if not path.exists():
            print(f"Skipping missing: {path}")
            continue
        with path.open() as f:
            metrics = json.load(f)
        rows.append({
            "model": model_name,
            "AUROC": metrics["auroc"],
            "AUPRC": metrics["auprc"],
            "F1": metrics["f1"],
            "Balanced accuracy": metrics["balanced_accuracy"],
            "Sensitivity": metrics["sensitivity"],
            "Specificity": metrics["specificity"],
        })
    if not rows:
        raise RuntimeError("No model metric files found.")
    df = pd.DataFrame(rows)
    path = output_dir / f"model_comparison_{args.split}.csv"
    df.to_csv(path, index=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
