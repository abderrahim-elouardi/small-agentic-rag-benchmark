
import torch

from small_agentic_rag.model_lab.ffn import (
    FeedForwardNetwork,
)


def test_output_shape():
    model = FeedForwardNetwork(d_model=64, d_ff=256)
    x = torch.randn(2, 10, 64)

    y = model(x)

    assert y.shape == x.shape


def test_backward_pass():
    model = FeedForwardNetwork(d_model=64, d_ff=256)
    x = torch.randn(2, 10, 64, requires_grad=True)

    y = model(x)
    y.mean().backward()

    assert x.grad is not None
    assert model.network[0].weight.grad is not None
