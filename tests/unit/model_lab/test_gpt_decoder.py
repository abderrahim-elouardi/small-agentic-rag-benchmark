import torch
from small_agentic_rag.model_lab.gpt_decoder import GPTDecoder

def test_decoder_output_shape():
    model = GPTDecoder(
        vocab_size=5000,
        num_layers=4,
        d_model=64,
        num_heads=4,
        d_ff=256,
        dropout=0.0,
    )

    x = torch.randn(2, 10, 64)
    logits = model(x)

    assert logits.shape == (2, 10, 5000)


def test_decoder_hidden_shape():
    model = GPTDecoder(
        vocab_size=500,
        num_layers=2,
        d_model=64,
        num_heads=4,
        d_ff=256,
        dropout=0.0,
    )

    x = torch.randn(2, 10, 64)
    logits, hidden = model(x, return_hidden=True)

    assert logits.shape == (2, 10, 500)
    assert hidden.shape == (2, 10, 64)


def test_decoder_backward_pass():
    model = GPTDecoder(
        vocab_size=100,
        num_layers=2,
        d_model=64,
        num_heads=4,
        d_ff=256,
        dropout=0.0,
    )

    x = torch.randn(2, 10, 64)
    logits = model(x)

    loss = logits.mean()
    loss.backward()

    assert model.lm_head.weight.grad is not None
    assert torch.isfinite(model.lm_head.weight.grad).all()
