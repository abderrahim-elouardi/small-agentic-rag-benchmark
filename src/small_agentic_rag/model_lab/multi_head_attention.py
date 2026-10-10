import torch.nn as nn 
from small_agentic_rag.model_lab.attention import AttentionHead
import torch 


class MaskedMultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model=64,
        num_heads=4,
        d_k=None,
        d_v=None
    ):
        super().__init__()

        assert d_model % num_heads == 0, (
            "d_model doit être divisible par num_heads"
        )

        self.d_model = d_model
        self.num_heads = num_heads

        # Dimension de chaque tête
        head_dim = d_model // num_heads

        self.d_k = head_dim if d_k is None else d_k
        self.d_v = head_dim if d_v is None else d_v

        # Création des têtes indépendantes
        self.heads = nn.ModuleList([
            AttentionHead(
                d_model=d_model,
                d_k=self.d_k,
                d_v=self.d_v
            )
            for _ in range(num_heads)
        ])

        # Projection après concaténation
        self.wo = nn.Linear(
            num_heads * self.d_v,
            d_model
        )

    def forward(self, x):

        # x : (B, L, d_model)

        # Calcul parallèle conceptuel : chaque tête traite x
        head_outputs = [
            head(x, causal=True)
            for head in self.heads
        ]

        # Concaténation sur la dernière dimension
        # Chaque sortie : (B, L, d_v)
        # Résultat : (B, L, num_heads * d_v)
        concatenated = torch.cat(
            head_outputs,
            dim=-1
        )

        # Projection finale
        output = self.wo(concatenated)

        return output





