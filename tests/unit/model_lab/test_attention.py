import torch.nn.functional as F
import torch 
from small_agentic_rag.model_lab.attention import MaskedSingleHeadSelfAttention ,AttentionHead 

def test_attention():

    attention = MaskedSingleHeadSelfAttention(
        d_model=64,
        d_k=64,
        d_v=64
    )

    x = torch.randn(2, 10, 64)

    output = attention(x, causal=True)

    print("Input :", x.shape)
    print("Output:", output.shape)

    assert output.shape == (2, 10, 64)
    assert torch.isfinite(output).all()

    print("Test réussi !")

def test_head_attention():
    
    head = AttentionHead(
    d_model=64,
    d_k=16,
    d_v=16
    )

    x = torch.randn(2, 10, 64)
    output = head(x, causal=True)

    print("Input :", x.shape)
    print("Output:", output.shape)

    assert output.shape == (2, 10, 16)
    assert torch.isfinite(output).all()

    print("Test réussi !")


