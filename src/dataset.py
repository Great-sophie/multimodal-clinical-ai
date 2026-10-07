from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def build_transforms(train: bool, image_size: int = 224):
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.10, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

class HAM10000Dataset(Dataset):
    def __init__(self, manifest_csv, split, image_size=224, train_augment=False):
        self.df = pd.read_csv(manifest_csv)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No rows found for split={split!r}")
        self.transform = build_transforms(train=(train_augment and split == "train"),
                                          image_size=image_size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(Path(row["image_path"])).convert("RGB")
        image = self.transform(image)
        return {
            "image": image,
            "age_z": torch.tensor([float(row["age_z"])], dtype=torch.float32),
            "sex_idx": torch.tensor(int(row["sex_idx"]), dtype=torch.long),
            "localization_idx": torch.tensor(int(row["localization_idx"]), dtype=torch.long),
            "target": torch.tensor(float(row["target"]), dtype=torch.float32),
            "image_id": str(row["image_id"]),
            "lesion_id": str(row["lesion_id"]),
        }
