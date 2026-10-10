
import torch
import torch.nn as nn

from small_agentic_rag.model_lab.multi_head_attention import (
    MaskedMultiHeadAttention,
)
from small_agentic_rag.model_lab.ffn import (
    FeedForwardNetwork,
)


class TransformerDecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.attention = MaskedMultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            # dropout=dropout,
        )

        self.ffn = FeedForwardNetwork(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Attention + connexion résiduelle + normalisation
        attention_output = self.attention(x)
        x = self.norm1(x + self.dropout1(attention_output))

        # 2. FFN + connexion résiduelle + normalisation
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))

        return x
