import pytest
import torch

from mmseg.models.backbones.vit import VisionTransformer
from .utils import check_norm_state


def test_vit_backbone():
    with pytest.raises(TypeError):
        # pretrained must be a string path
        model = VisionTransformer()
        model.init_weights(pretrained=0)

    with pytest.raises(TypeError):
        # img_size must be int or tuple
        model = VisionTransformer(img_size=512.0)

    with pytest.raises(TypeError):
        # test upsample_pos_embed function
        x = torch.randn(1, 196)
        VisionTransformer.upsample_pos_embed(x, 512, 512, 224, 224)

    # Test ViT backbone with input size of 224 and patch size of 16
    model = VisionTransformer()
    model.init_weights()
    model.train()

    assert check_norm_state(model.modules(), True)

    imgs = torch.randn(4, 3, 224, 224)
    feat = model(imgs)
    assert feat[0].shape == torch.Size([4, 768, 14, 14])
