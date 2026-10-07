import torch
from src.cross_attention import CrossAttentionModel

def test_cross_attention_shapes():
    m=CrossAttentionModel(4,12,pretrained=False,d_model=64,heads=4)
    image=torch.randn(2,3,224,224); age=torch.randn(2,1); sex=torch.tensor([1,2]); loc=torch.tensor([3,4])
    logits,attn=m(image,age,sex,loc,return_attention=True)
    assert logits.shape==(2,)
    assert attn.shape==(2,4,3,49)
