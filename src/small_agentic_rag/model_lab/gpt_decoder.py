
import torch
import torch.nn as nn

from small_agentic_rag.model_lab.transformer_block import (
    TransformerDecoderBlock,
)


class GPTDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int = 4,
        d_model: int = 64,
        num_heads: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        if vocab_size <= 0:
            raise ValueError("vocab_size doit être positif.")

        if num_layers <= 0:
            raise ValueError("num_layers doit être positif.")

        self.vocab_size = vocab_size
        self.d_model = d_model

        # Empilement des blocs Transformer
        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Normalisation finale
        self.final_norm = nn.LayerNorm(d_model)

        # Projection vers le vocabulaire
        self.lm_head = nn.Linear(
            d_model,
            vocab_size,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_hidden: bool = False,
    ):
        """
        x : embeddings de forme (B, L, d_model)

        Retour :
            logits : (B, L, vocab_size)
            hidden : (B, L, d_model), si demandé
        """
        for block in self.blocks:
            x = block(x)

        hidden = self.final_norm(x)
        logits = self.lm_head(hidden)

        if return_hidden:
            return logits, hidden

        return logits
