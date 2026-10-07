import argparse
import json
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.dataset import HAM10000Dataset
from src.models import ImageOnlyModel, TabularOnlyModel, LateFusionModel
from src.metrics import binary_metrics

def build_model_from_checkpoint(checkpoint):
    schema = checkpoint["schema"]
    model_name = checkpoint["model_name"]
    n_sex = len(schema["sex_categories"])
    n_loc = len(schema["localization_categories"])
    if model_name == "image":
        model = ImageOnlyModel(pretrained=False)
    elif model_name == "tabular":
        model = TabularOnlyModel(n_sex, n_loc)
    elif model_name == "late":
        model = LateFusionModel(n_sex, n_loc, pretrained=False)
    else:
        raise ValueError(model_name)
    model.load_state_dict(checkpoint["model_state"])
    return model, model_name

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model, model_name = build_model_from_checkpoint(checkpoint)
    model = model.to(device)
    model.eval()

    ds = HAM10000Dataset(
        args.manifest, args.split,
        image_size=checkpoint.get("image_size", 224),
        train_augment=False,
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )

    all_true, all_prob, all_image_id, all_lesion_id = [], [], [], []
    for batch in loader:
        logits = model(
            batch["image"].to(device),
            batch["age_z"].to(device),
            batch["sex_idx"].to(device),
            batch["localization_idx"].to(device),
        )
        prob = torch.sigmoid(logits)
        all_true.extend(batch["target"].numpy().tolist())
        all_prob.extend(prob.cpu().numpy().tolist())
        all_image_id.extend(batch["image_id"])
        all_lesion_id.extend(batch["lesion_id"])

    metrics = binary_metrics(all_true, all_prob, threshold=0.5)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"{model_name}_{args.split}_metrics.json"
    pred_path = output_dir / f"{model_name}_{args.split}_predictions.csv"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame({
        "image_id": all_image_id,
        "lesion_id": all_lesion_id,
        "y_true": all_true,
        "y_prob": all_prob,
    }).to_csv(pred_path, index=False)
    print(json.dumps(metrics, indent=2))
    print(f"Saved: {metrics_path}")
    print(f"Saved: {pred_path}")

if __name__ == "__main__":
    main()
