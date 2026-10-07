import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class SpatialImageEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.feature_dim = 512

    def forward(self, x):
        b = self.backbone
        x = b.conv1(x); x = b.bn1(x); x = b.relu(x); x = b.maxpool(x)
        x = b.layer1(x); x = b.layer2(x); x = b.layer3(x); x = b.layer4(x)
        return x


class ClinicalTokenEncoder(nn.Module):
    def __init__(self, n_sex, n_loc, d_model=128):
        super().__init__()
        self.age_proj = nn.Sequential(nn.Linear(1, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.sex_emb = nn.Embedding(n_sex, d_model)
        self.loc_emb = nn.Embedding(n_loc, d_model)
        self.type_emb = nn.Parameter(torch.randn(1, 3, d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, age_z, sex_idx, loc_idx):
        age = self.age_proj(age_z).unsqueeze(1)
        sex = self.sex_emb(sex_idx).unsqueeze(1)
        loc = self.loc_emb(loc_idx).unsqueeze(1)
        x = torch.cat([age, sex, loc], dim=1)
        return self.norm(x + self.type_emb)


class CrossAttentionModel(nn.Module):
    """Clinical tokens query spatial image tokens."""
    def __init__(self, n_sex, n_loc, pretrained=True, d_model=128, heads=4):
        super().__init__()
        self.image_encoder = SpatialImageEncoder(pretrained=pretrained)
        self.image_proj = nn.Linear(512, d_model)
        self.clinical = ClinicalTokenEncoder(n_sex, n_loc, d_model=d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=0.2, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, 256), nn.ReLU(), nn.Dropout(0.30), nn.Linear(256, 1)
        )

    def forward(self, image, age_z, sex_idx, localization_idx, return_attention=False):
        fmap = self.image_encoder(image)                    # [B,512,7,7]
        image_tokens = fmap.flatten(2).transpose(1, 2)     # [B,49,512]
        image_tokens = self.image_proj(image_tokens)       # [B,49,D]
        clinical_tokens = self.clinical(age_z, sex_idx, localization_idx)  # [B,3,D]
        attended, weights = self.attn(
            clinical_tokens, image_tokens, image_tokens,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        attended = self.norm(clinical_tokens + attended)
        clinical_summary = attended.mean(dim=1)
        image_summary = image_tokens.mean(dim=1)
        logits = self.head(torch.cat([clinical_summary, image_summary], dim=1)).squeeze(1)
        return (logits, weights) if return_attention else logits
