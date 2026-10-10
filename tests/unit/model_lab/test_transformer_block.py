
import torch

from small_agentic_rag.model_lab.transformer_block import TransformerDecoderBlock


def test_output_shape():
    model = TransformerDecoderBlock(
        d_model=64,
        num_heads=4,
        d_ff=256,
        dropout=0.0,
    )

    x = torch.randn(2, 10, 64)
    y = model(x)

    assert y.shape == x.shape


def test_backward_pass():
    model = TransformerDecoderBlock(
        d_model=64,
        num_heads=4,
        d_ff=256,
        dropout=0.0,
    )

    x = torch.randn(2, 10, 64, requires_grad=True)
    y = model(x)

    y.mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
