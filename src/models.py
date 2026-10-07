import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ImageEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        self.feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

    def forward(self, image):
        return self.backbone(image)

class TabularEncoder(nn.Module):
    def __init__(self, n_sex_categories, n_localization_categories,
                 sex_emb_dim=4, localization_emb_dim=8, hidden_dim=32, out_dim=64):
        super().__init__()
        self.sex_embedding = nn.Embedding(n_sex_categories, sex_emb_dim)
        self.localization_embedding = nn.Embedding(n_localization_categories, localization_emb_dim)
        in_dim = 1 + sex_emb_dim + localization_emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )
        self.feature_dim = out_dim

    def forward(self, age_z, sex_idx, localization_idx):
        sex = self.sex_embedding(sex_idx)
        loc = self.localization_embedding(localization_idx)
        x = torch.cat([age_z, sex, loc], dim=1)
        return self.mlp(x)

class ImageOnlyModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.image_encoder = ImageEncoder(pretrained=pretrained)
        self.classifier = nn.Linear(self.image_encoder.feature_dim, 1)

    def forward(self, image, age_z=None, sex_idx=None, localization_idx=None):
        z = self.image_encoder(image)
        return self.classifier(z).squeeze(1)

class TabularOnlyModel(nn.Module):
    def __init__(self, n_sex_categories, n_localization_categories):
        super().__init__()
        self.tabular_encoder = TabularEncoder(n_sex_categories, n_localization_categories)
        self.classifier = nn.Linear(self.tabular_encoder.feature_dim, 1)

    def forward(self, image, age_z, sex_idx, localization_idx):
        z = self.tabular_encoder(age_z, sex_idx, localization_idx)
        return self.classifier(z).squeeze(1)

class LateFusionModel(nn.Module):
    def __init__(self, n_sex_categories, n_localization_categories, pretrained=True):
        super().__init__()
        self.image_encoder = ImageEncoder(pretrained=pretrained)
        self.tabular_encoder = TabularEncoder(n_sex_categories, n_localization_categories)
        fusion_dim = self.image_encoder.feature_dim + self.tabular_encoder.feature_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(256, 1),
        )

    def forward(self, image, age_z, sex_idx, localization_idx):
        image_z = self.image_encoder(image)
        tabular_z = self.tabular_encoder(age_z, sex_idx, localization_idx)
        fused = torch.cat([image_z, tabular_z], dim=1)
        return self.fusion_head(fused).squeeze(1)
