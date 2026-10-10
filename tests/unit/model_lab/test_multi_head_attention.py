import torch 
from small_agentic_rag.model_lab.multi_head_attention import MaskedMultiHeadAttention

def test_MaskedMultiHeadAttention():
    attention = MaskedMultiHeadAttention(
        d_model=64,
        num_heads=4
    )

    x = torch.randn(2, 10, 64)

    output = attention(x)

    print("Input :", x.shape)
    print("Output:", output.shape)

    assert output.shape == (2, 10, 64)
    assert torch.isfinite(output).all()

    # Vérification du passage des gradients
    loss = output.mean()
    loss.backward()

    assert attention.heads[0].wq.weight.grad is not None

    print("Tests réussis !")