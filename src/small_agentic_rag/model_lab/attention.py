import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


class Masked_Single_Head_Self_Attention(nn.Module):
    def __init__(self, d_model=64, d_k=64, d_v=64):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        self.wq = nn.Linear(d_model, d_k)
        self.wk = nn.Linear(d_model, d_k)
        self.wv = nn.Linear(d_model, d_v)

    def forward(self, x, causal=True):
        # x : (batch_size, seq_len, d_model)

        Q = self.wq(x)
        K = self.wk(x)
        V = self.wv(x)

        # Scores : (batch_size, seq_len, seq_len)
        scores = Q @ K.transpose(-2, -1)
        scores = scores / (self.d_k ** 0.5)

        # Masque causal
        if causal:
            seq_len = x.size(1)

            mask = torch.triu(
                torch.ones(
                    seq_len, seq_len,
                    device=x.device,
                    dtype=torch.bool
                ),
                diagonal=1
            )

            scores = scores.masked_fill(mask, float("-inf"))

        # Poids d'attention
        attn_weights = F.softmax(scores, dim=-1)

        # Combinaison pondérée des valeurs
        output = attn_weights @ V

        return output