from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from small_agentic_rag.model_lab.embeddings import (
    all_ids,
    vocab,
    vocab_size,
)
from small_agentic_rag.model_lab.dataset import NextTokenDataset
from small_agentic_rag.model_lab.gpt_language_model import GPTLanguageModel


# --------------------------------------------------
# 1. Configuration du projet
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = PROJECT_ROOT / "models" / "mini_gpt.pt"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURE_PATH = FIGURE_DIR / "training_curves.png"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# 2. Évaluation sur les données de validation
# --------------------------------------------------

@torch.no_grad()
def evaluate(model, data_loader, criterion, device):
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    for input_ids, targets in data_loader:
        input_ids = input_ids.to(device)
        targets = targets.to(device)

        logits = model(input_ids)

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )

        number_of_tokens = targets.numel()

        total_loss += loss.item() * number_of_tokens
        total_tokens += number_of_tokens

    if was_training:
        model.train()

    if total_tokens == 0:
        raise ValueError("Le dataset de validation est vide.")

    return total_loss / total_tokens


# --------------------------------------------------
# 3. Tracer et sauvegarder les courbes
# --------------------------------------------------

def save_training_curves(train_losses, val_losses):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        train_losses,
        label="Train Loss",
        linewidth=2,
    )

    plt.plot(
        epochs,
        val_losses,
        label="Validation Loss",
        linewidth=2,
    )

    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Évolution de la loss pendant l'entraînement")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        FIGURE_PATH,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Courbe sauvegardée : {FIGURE_PATH}")


# --------------------------------------------------
# 4. Entraînement du Mini-GPT
# --------------------------------------------------

def train():
    # Hyperparamètres
    block_size = 16
    batch_size = 4
    epochs = 50
    learning_rate = 3e-4

    num_layers = 2
    d_model = 64
    num_heads = 4
    d_ff = 256
    dropout = 0.1

    print(f"Device : {DEVICE}")
    print(f"Taille du vocabulaire : {vocab_size}")
    print(f"Nombre total de tokens : {len(all_ids)}")

    # ----------------------------------------------
    # Séparer les données avant de créer les fenêtres
    # ----------------------------------------------

    split_index = int(len(all_ids) * 0.9)

    train_ids = all_ids[:split_index]
    val_ids = all_ids[split_index:]

    train_dataset = NextTokenDataset(
        token_ids=train_ids,
        block_size=block_size,
    )

    val_dataset = NextTokenDataset(
        token_ids=val_ids,
        block_size=block_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    print(f"Tokens d'entraînement : {len(train_ids)}")
    print(f"Tokens de validation : {len(val_ids)}")

    # ----------------------------------------------
    # Créer le modèle
    # ----------------------------------------------

    model = GPTLanguageModel(
        vocab_size=vocab_size,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=dropout,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
    )

    # Historique des losses pour le graphique
    train_losses = []
    val_losses = []

    # ----------------------------------------------
    # Boucle d'entraînement
    # ----------------------------------------------
    best_val_loss = float("inf")
    patience = 5
    epochs_without_improvement = 0
    for epoch in range(epochs):
        model.train()

        total_loss = 0.0
        total_tokens = 0

        for input_ids, targets in train_loader:
            input_ids = input_ids.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()

            logits = model(input_ids)

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

            number_of_tokens = targets.numel()

            total_loss += loss.item() * number_of_tokens
            total_tokens += number_of_tokens

        # Loss moyenne sur les tokens d'entraînement
        train_loss = total_loss / total_tokens

        # Loss calculée sans mise à jour des poids
        val_loss = evaluate(
            model,
            val_loader,
            criterion,
            DEVICE,
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab_size": vocab_size,
                    "vocab": vocab,
                    "block_size": block_size,
                    "config": {
                        "num_layers": num_layers,
                        "d_model": d_model,
                        "num_heads": num_heads,
                        "d_ff": d_ff,
                        "dropout": dropout,
                    },
                    "training": {
                        "epochs": epoch + 1,
                        "learning_rate": learning_rate,
                        "batch_size": batch_size,
                    },
                    "best_val_loss": best_val_loss,
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                },
                MODEL_PATH,
            )

            print(
                f"Meilleur modèle sauvegardé à l'epoch {epoch + 1} "
                f"| Validation Loss : {best_val_loss:.4f}"
            )

        else:
            epochs_without_improvement += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1:02d}/{epochs} | "
                f"Train Loss : {train_loss:.4f} | "
                f"Validation Loss : {val_loss:.4f}"
            )

        if epochs_without_improvement >= patience:
            print(
                f"Arrêt anticipé à l'epoch {epoch + 1}. "
                f"Meilleure Validation Loss : {best_val_loss:.4f}"
            )
            break

   

    # ----------------------------------------------
    # Générer et sauvegarder le graphique
    # ----------------------------------------------
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_training_curves(train_losses, val_losses)


if __name__ == "__main__":
    train()
