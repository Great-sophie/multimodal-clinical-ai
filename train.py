import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.dataset import HAM10000Dataset
from src.models import ImageOnlyModel, TabularOnlyModel, LateFusionModel
from src.metrics import binary_metrics

def load_schema(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_model(model_name, schema, pretrained):
    n_sex = len(schema["sex_categories"])
    n_loc = len(schema["localization_categories"])
    if model_name == "image":
        return ImageOnlyModel(pretrained=pretrained)
    if model_name == "tabular":
        return TabularOnlyModel(n_sex, n_loc)
    if model_name == "late":
        return LateFusionModel(n_sex, n_loc, pretrained=pretrained)
    raise ValueError(f"Unknown model: {model_name}")

def forward_batch(model, batch, device):
    return model(
        batch["image"].to(device),
        batch["age_z"].to(device),
        batch["sex_idx"].to(device),
        batch["localization_idx"].to(device),
    )

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_true, all_prob, all_ids, losses = [], [], [], []
    criterion = torch.nn.BCEWithLogitsLoss()
    for batch in loader:
        target = batch["target"].to(device)
        logits = forward_batch(model, batch, device)
        loss = criterion(logits, target)
        prob = torch.sigmoid(logits)
        losses.append(float(loss.item()))
        all_true.extend(target.cpu().numpy().tolist())
        all_prob.extend(prob.cpu().numpy().tolist())
        all_ids.extend(batch["image_id"])
    metrics = binary_metrics(all_true, all_prob, threshold=0.5)
    metrics["loss"] = float(np.mean(losses))
    pred_df = pd.DataFrame({"image_id": all_ids, "y_true": all_true, "y_prob": all_prob})
    return metrics, pred_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["image", "tabular", "late"], required=True)
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--schema", default="data/tabular_schema.json")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--init-image-checkpoint", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    schema = load_schema(args.schema)
    train_ds = HAM10000Dataset(args.manifest, "train", args.image_size, True)
    val_ds = HAM10000Dataset(args.manifest, "val", args.image_size, False)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )

    model = build_model(args.model, schema, pretrained=not args.no_pretrained).to(device)

    if args.model == "late" and args.init_image_checkpoint:
        checkpoint = torch.load(args.init_image_checkpoint, map_location="cpu")
        image_state = {
            key.replace("image_encoder.", "", 1): value
            for key, value in checkpoint["model_state"].items()
            if key.startswith("image_encoder.")
        }
        model.image_encoder.load_state_dict(image_state, strict=False)
        print(f"Initialized late-fusion image encoder from {args.init_image_checkpoint}")

    manifest_df = pd.read_csv(args.manifest)
    train_rows = manifest_df[manifest_df["split"] == "train"]
    n_pos = int(train_rows["target"].sum())
    n_neg = int(len(train_rows) - n_pos)
    pos_weight = n_neg / max(n_pos, 1)
    print(f"Train positives={n_pos}, negatives={n_neg}, pos_weight={pos_weight:.3f}")

    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_auc = -1.0
    history = []
    checkpoint_path = output_dir / f"{args.model}_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            target = batch["target"].to(device)
            optimizer.zero_grad()
            logits = forward_batch(model, batch, device)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        val_metrics, _ = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            **{f"val_{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float))}
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d} | train_loss={row['train_loss']:.4f} | "
            f"val_AUROC={val_metrics['auroc']:.4f} | "
            f"val_AUPRC={val_metrics['auprc']:.4f} | "
            f"val_F1={val_metrics['f1']:.4f}"
        )
        if val_metrics["auroc"] > best_auc:
            best_auc = val_metrics["auroc"]
            torch.save(
                {
                    "model_name": args.model,
                    "model_state": model.state_dict(),
                    "schema": schema,
                    "image_size": args.image_size,
                    "best_val_auroc": best_auc,
                },
                checkpoint_path,
            )
            print(f"  -> saved new best checkpoint: {checkpoint_path}")

    with (output_dir / f"{args.model}_history.json").open("w") as f:
        json.dump(history, f, indent=2)
    print(f"Best validation AUROC: {best_auc:.4f}")

if __name__ == "__main__":
    main()
