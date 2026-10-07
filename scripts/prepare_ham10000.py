import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

POSITIVE_DX = {"mel", "bcc", "akiec"}
NEGATIVE_DX = {"nv", "bkl", "df", "vasc"}

def normalize_text(value):
    if pd.isna(value):
        return "unknown"
    text = str(value).strip().lower()
    return text if text else "unknown"

def build_image_map(dataset_root):
    image_map = {}
    for path in dataset_root.rglob("*.jpg"):
        image_map[path.stem] = path.resolve()
    return image_map

def make_group_split(df, seed):
    lesion_df = df[["lesion_id", "target"]].drop_duplicates().reset_index(drop=True)
    if lesion_df.groupby("lesion_id")["target"].nunique().max() != 1:
        raise ValueError("At least one lesion_id maps to multiple targets.")
    train_lesions, temp_lesions = train_test_split(
        lesion_df, test_size=0.30, random_state=seed, stratify=lesion_df["target"]
    )
    val_lesions, test_lesions = train_test_split(
        temp_lesions, test_size=0.50, random_state=seed, stratify=temp_lesions["target"]
    )
    split_map = {}
    for lesion_id in train_lesions["lesion_id"]:
        split_map[lesion_id] = "train"
    for lesion_id in val_lesions["lesion_id"]:
        split_map[lesion_id] = "val"
    for lesion_id in test_lesions["lesion_id"]:
        split_map[lesion_id] = "test"
    return split_map

def fit_tabular_schema(train_df):
    age = pd.to_numeric(train_df["age"], errors="coerce")
    age_median = float(age.median())
    age_imputed = age.fillna(age_median)
    age_mean = float(age_imputed.mean())
    age_std = float(age_imputed.std(ddof=0))
    if not np.isfinite(age_std) or age_std < 1e-8:
        age_std = 1.0
    sex_categories = sorted(set(train_df["sex_clean"]))
    localization_categories = sorted(set(train_df["localization_clean"]))
    if "unknown" not in sex_categories:
        sex_categories = ["unknown"] + sex_categories
    if "unknown" not in localization_categories:
        localization_categories = ["unknown"] + localization_categories
    return {
        "age_median": age_median,
        "age_mean": age_mean,
        "age_std": age_std,
        "sex_categories": sex_categories,
        "localization_categories": localization_categories,
    }

def apply_tabular_schema(df, schema):
    df = df.copy()
    age = pd.to_numeric(df["age"], errors="coerce").fillna(schema["age_median"])
    df["age_z"] = (age - schema["age_mean"]) / schema["age_std"]
    sex_map = {value: i for i, value in enumerate(schema["sex_categories"])}
    loc_map = {value: i for i, value in enumerate(schema["localization_categories"])}
    df["sex_idx"] = df["sex_clean"].map(sex_map).fillna(sex_map["unknown"]).astype(int)
    df["localization_idx"] = (
        df["localization_clean"].map(loc_map).fillna(loc_map["unknown"]).astype(int)
    )
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="data/raw")
    parser.add_argument("--output-manifest", default="data/manifest.csv")
    parser.add_argument("--output-schema", default="data/tabular_schema.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    metadata_candidates = list(dataset_root.rglob("HAM10000_metadata.csv"))
    if len(metadata_candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one HAM10000_metadata.csv under {dataset_root}; "
            f"found {len(metadata_candidates)}"
        )
    metadata_path = metadata_candidates[0]
    df = pd.read_csv(metadata_path)

    required = {"lesion_id", "image_id", "dx", "age", "sex", "localization"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing metadata columns: {sorted(missing)}")

    valid_dx = POSITIVE_DX | NEGATIVE_DX
    df = df[df["dx"].isin(valid_dx)].copy().reset_index(drop=True)
    df["target"] = df["dx"].isin(POSITIVE_DX).astype(int)
    df["sex_clean"] = df["sex"].apply(normalize_text)
    df["localization_clean"] = df["localization"].apply(normalize_text)

    image_map = build_image_map(dataset_root)
    df["image_path"] = df["image_id"].map(image_map)
    missing_images = df[df["image_path"].isna()]
    if len(missing_images):
        raise FileNotFoundError(f"{len(missing_images)} metadata rows have no matching JPG.")
    df["image_path"] = df["image_path"].map(lambda p: str(Path(p)))

    split_map = make_group_split(df, seed=args.seed)
    df["split"] = df["lesion_id"].map(split_map)

    split_sets = {
        split: set(df.loc[df["split"] == split, "lesion_id"])
        for split in ["train", "val", "test"]
    }
    assert split_sets["train"].isdisjoint(split_sets["val"])
    assert split_sets["train"].isdisjoint(split_sets["test"])
    assert split_sets["val"].isdisjoint(split_sets["test"])

    schema = fit_tabular_schema(df[df["split"] == "train"])
    df = apply_tabular_schema(df, schema)

    keep = [
        "lesion_id", "image_id", "image_path", "dx", "target", "age", "sex",
        "localization", "age_z", "sex_idx", "localization_idx", "split",
    ]
    output_manifest = Path(args.output_manifest)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    df[keep].to_csv(output_manifest, index=False)

    output_schema = Path(args.output_schema)
    output_schema.parent.mkdir(parents=True, exist_ok=True)
    with output_schema.open("w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    print(f"Metadata: {metadata_path}")
    print(f"Images found: {len(image_map)}")
    print(f"Manifest rows: {len(df)}")
    print()
    for split in ["train", "val", "test"]:
        part = df[df["split"] == split]
        print(
            f"{split:5s}: {len(part):5d} images | "
            f"{part['lesion_id'].nunique():5d} lesions | "
            f"positive prevalence={part['target'].mean():.4f}"
        )
    print()
    print("No lesion leakage: YES")
    print(f"Saved manifest: {output_manifest}")
    print(f"Saved schema: {output_schema}")

if __name__ == "__main__":
    main()
