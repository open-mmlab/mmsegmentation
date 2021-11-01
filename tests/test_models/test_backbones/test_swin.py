import pytest
import torch

from mmseg.models.backbones.swin import SwinBlock, SwinTransformer


def test_swin_block():
    # test SwinBlock structure and forward
    block = SwinBlock(embed_dims=32, num_heads=4, feedforward_channels=128)
    assert block.ffn.embed_dims == 32
    assert block.attn.w_msa.num_heads == 4
    assert block.ffn.feedforward_channels == 128
    x = torch.randn(1, 56 * 56, 32)
    x_out = block(x, (56, 56))
    assert x_out.shape == torch.Size([1, 56 * 56, 32])

    # Test BasicBlock with checkpoint forward
    block = SwinBlock(
        embed_dims=64, num_heads=4, feedforward_channels=256, with_cp=True)
    assert block.with_cp
    x = torch.randn(1, 56 * 56, 64)
    x_out = block(x, (56, 56))
    assert x_out.shape == torch.Size([1, 56 * 56, 64])


def test_swin_transformer():
    """Test Swin Transformer backbone."""

    with pytest.raises(TypeError):
        # Pretrained arg must be str or None.
        SwinTransformer(pretrained=123)

    with pytest.raises(AssertionError):
        # Because swin uses non-overlapping patch embed, so the stride of patch
        # embed must be equal to patch size.
        SwinTransformer(strides=(2, 2, 2, 2), patch_size=4)

    # test pretrained image size
    with pytest.raises(AssertionError):
        SwinTransformer(pretrain_img_size=(112, 112, 112))

    # Test absolute position embedding
    temp = torch.randn((1, 3, 112, 112))
    model = SwinTransformer(pretrain_img_size=112, use_abs_pos_embed=True)
    model.init_weights()
    model(temp)

    # Test patch norm
    model = SwinTransformer(patch_norm=False)
    model(temp)

    # Test normal inference
    temp = torch.randn((1, 3, 256, 256))
    model = SwinTransformer()
    outs = model(temp)
    assert outs[0].shape == (1, 96, 64, 64)
    assert outs[1].shape == (1, 192, 32, 32)
    assert outs[2].shape == (1, 384, 16, 16)
    assert outs[3].shape == (1, 768, 8, 8)

    # Test abnormal inference size
    temp = torch.randn((1, 3, 255, 255))
    model = SwinTransformer()
    outs = model(temp)
    assert outs[0].shape == (1, 96, 64, 64)
    assert outs[1].shape == (1, 192, 32, 32)
    assert outs[2].shape == (1, 384, 16, 16)
    assert outs[3].shape == (1, 768, 8, 8)

    # Test abnormal inference size
    temp = torch.randn((1, 3, 112, 137))
    model = SwinTransformer()
    outs = model(temp)
    assert outs[0].shape == (1, 96, 28, 35)
    assert outs[1].shape == (1, 192, 14, 18)
    assert outs[2].shape == (1, 384, 7, 9)
    assert outs[3].shape == (1, 768, 4, 5)

    # Test frozen
    model = SwinTransformer(frozen_stages=4)
    model.train()
    for p in model.parameters():
        assert not p.requires_grad

    # Test absolute position embedding frozen
    model = SwinTransformer(frozen_stages=4, use_abs_pos_embed=True)
    model.train()
    for p in model.parameters():
        assert not p.requires_grad

    # Test Swin with checkpoint forward
    temp = torch.randn((1, 3, 56, 56))
    model = SwinTransformer(with_cp=True)
    for m in model.modules():
        if isinstance(m, SwinBlock):
            assert m.with_cp
    model.init_weights()
    model.train()
    model(temp)
