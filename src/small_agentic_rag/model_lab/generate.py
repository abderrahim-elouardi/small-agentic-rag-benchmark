from pathlib import Path

import torch
from tokenizers import Tokenizer

from small_agentic_rag.model_lab.gpt_language_model import GPTLanguageModel


# ============================================================
# 1. Chemins et configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = PROJECT_ROOT / "models" / "mini_gpt.pt"
TOKENIZER_PATH = PROJECT_ROOT / "models" / "tokenizer.json"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 2. Chargement du tokenizer BPE
# ============================================================

def load_tokenizer():
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            f"Tokenizer introuvable : {TOKENIZER_PATH}\n"
            "Exécute d'abord train_tokenizer.py."
        )

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    return tokenizer


# ============================================================
# 3. Chargement du modèle entraîné
# ============================================================

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {MODEL_PATH}\n"
            "Entraîne d'abord ton modèle avec le tokenizer BPE."
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    config = checkpoint["config"]
    block_size = checkpoint["block_size"]
    vocab_size = checkpoint["vocab_size"]

    # Vérifier la compatibilité entre modèle et tokenizer
    tokenizer = load_tokenizer()
    tokenizer_vocab_size = tokenizer.get_vocab_size()

    if vocab_size != tokenizer_vocab_size:
        raise ValueError(
            f"Incompatibilité : le modèle attend {vocab_size} tokens, "
            f"mais le tokenizer en contient {tokenizer_vocab_size}.\n"
            "Réentraîne le modèle avec ce tokenizer BPE."
        )

    model = GPTLanguageModel(
        vocab_size=vocab_size,
        num_layers=config["num_layers"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, block_size, tokenizer


# ============================================================
# 4. Génération de texte
# ============================================================

@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    prompt,
    block_size,
    max_new_tokens=40,
    temperature=0.8,
    top_k=20,
):
    if temperature <= 0:
        raise ValueError(
            "La température doit être strictement positive."
        )

    if not prompt.strip():
        raise ValueError("Le prompt ne peut pas être vide.")

    # Encoder le prompt avec BPE
    encoded = tokenizer.encode(prompt)
    token_ids = encoded.ids

    if not token_ids:
        raise ValueError("Le prompt ne contient aucun token.")

    input_ids = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=DEVICE,
    )

    # Récupérer l'ID du token EOS s'il existe
    eos_id = tokenizer.token_to_id("[EOS]")

    for _ in range(max_new_tokens):

        # Garder les block_size derniers tokens
        context = input_ids[:, -block_size:]

        # Forme des logits : (batch, longueur, vocab_size)
        logits = model(context)

        # Logits du prochain token
        next_token_logits = logits[:, -1, :] / temperature

        # Limiter les candidats aux top_k tokens
        if top_k is not None:
            if top_k < 1:
                raise ValueError("top_k doit être supérieur ou égal à 1.")

            k = min(top_k, next_token_logits.size(-1))

            values, _ = torch.topk(next_token_logits, k)
            threshold = values[:, -1].unsqueeze(-1)

            next_token_logits = next_token_logits.masked_fill(
                next_token_logits < threshold,
                float("-inf"),
            )

        # Transformer les logits en probabilités
        probabilities = torch.softmax(
            next_token_logits,
            dim=-1,
        )

        # Échantillonner le prochain token
        next_id = torch.multinomial(
            probabilities,
            num_samples=1,
        )

        # Ajouter le token généré à la séquence
        input_ids = torch.cat(
            [input_ids, next_id],
            dim=1,
        )

        # Arrêter si le modèle génère EOS
        if eos_id is not None and next_id.item() == eos_id:
            break

    # Récupérer tous les IDs générés, prompt compris
    generated_ids = input_ids[0].tolist()

    # Retirer les tokens spéciaux de fin et inconnus
    special_ids = {
        token_id
        for token in ("[EOS]", "[UNK]", "[PAD]", "[BOS]")
        if (token_id := tokenizer.token_to_id(token)) is not None
    }

    generated_ids = [
        token_id
        for token_id in generated_ids
        if token_id not in special_ids
    ]

    # Reconstruire le texte avec le décodeur BPE
    return tokenizer.decode(generated_ids)


# ============================================================
# 5. Point d'entrée
# ============================================================

if __name__ == "__main__":

    model, block_size, tokenizer = load_model()

    prompt = "La science des données"

    generated = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        block_size=block_size,
        max_new_tokens=40,
        temperature=0.8,
        top_k=20,
    )

    print("\nPrompt :", prompt)
    print("\nTexte généré :")
    print(generated)
