
import torch
import torch.nn as nn

from small_agentic_rag.model_lab.gpt_decoder import GPTDecoder


class GPTLanguageModel(nn.Module):
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

        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model,
        )

        self.decoder = GPTDecoder(
            vocab_size=vocab_size,
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
        )

    def forward(self, input_ids: torch.Tensor):
        # input_ids : (B, L)
        embeddings = self.token_embedding(input_ids)

        # embeddings : (B, L, d_model)
        logits = self.decoder(embeddings)

        # logits : (B, L, vocab_size)
        return logits
