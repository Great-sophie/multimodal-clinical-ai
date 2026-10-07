import torch
from src.models import ImageOnlyModel, TabularOnlyModel, LateFusionModel

def fake_batch(batch_size=2):
    image = torch.randn(batch_size, 3, 64, 64)
    age_z = torch.randn(batch_size, 1)
    sex_idx = torch.tensor([0, 1], dtype=torch.long)
    localization_idx = torch.tensor([0, 2], dtype=torch.long)
    return image, age_z, sex_idx, localization_idx

def test_image_only_shape():
    model = ImageOnlyModel(pretrained=False)
    image, age_z, sex_idx, loc_idx = fake_batch()
    out = model(image, age_z, sex_idx, loc_idx)
    assert out.shape == (2,)

def test_tabular_only_shape():
    model = TabularOnlyModel(3, 5)
    image, age_z, sex_idx, loc_idx = fake_batch()
    out = model(image, age_z, sex_idx, loc_idx)
    assert out.shape == (2,)

def test_late_fusion_shape():
    model = LateFusionModel(3, 5, pretrained=False)
    image, age_z, sex_idx, loc_idx = fake_batch()
    out = model(image, age_z, sex_idx, loc_idx)
    assert out.shape == (2,)
