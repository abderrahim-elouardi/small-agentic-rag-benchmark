import torch 
from small_agentic_rag.model_lab.rope import apply_rope

def test_apply_rope():
        
    # Exemple : 3 tokens, embeddings de dimension 4
    x = torch.tensor([
        [1.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0, 0.0]
    ])

    positions = torch.arange(3)

    result = apply_rope(x, positions)

    print("Embeddings initiaux :")
    print(x)

    print("\nEmbeddings après RoPE :")
    print(result)
