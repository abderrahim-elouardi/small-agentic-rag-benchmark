
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from small_agentic_rag.model_lab.prepare_data import (
    all_ids,
    vocab_size,
)
from small_agentic_rag.model_lab.dataset import NextTokenDataset
from small_agentic_rag.model_lab.gpt_language_model import GPTLanguageModel


def train():
    # 1. Préparer le dataset
    block_size = 16
    batch_size = 4
    epochs = 50

    dataset = NextTokenDataset(
        token_ids=all_ids,
        block_size=block_size,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    # 2. Créer le modèle
    model = GPTLanguageModel(
        vocab_size=vocab_size,
        num_layers=2,
        d_model=64,
        num_heads=4,
        d_ff=256,
        dropout=0.1,
    )

    # 3. Définir la fonction de perte et l'optimiseur
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
    )

    # 4. Entraîner le modèle
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for input_ids, targets in dataloader:
            optimizer.zero_grad()

            logits = model(input_ids)

            # logits : (B, L, vocab_size)
            # targets : (B, L)
            loss = criterion(
                logits.reshape(-1, vocab_size),
                targets.reshape(-1),
            )

            loss.backward()

            # Stabiliser les gradients
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(dataloader)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1}/{epochs} "
                f"- Loss : {average_loss:.4f}"
            )

    # 5. Sauvegarder le modèle entraîné
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": vocab_size,
            "block_size": block_size,
            "config": {
                "num_layers": 2,
                "d_model": 64,
                "num_heads": 4,
                "d_ff": 256,
                "dropout": 0.1,
            },
        },
        "models/mini_gpt.pt",
    )

    print("Modèle sauvegardé dans mini_gpt.pt")


if __name__ == "__main__":
    train()
