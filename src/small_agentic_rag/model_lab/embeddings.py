
from collections import Counter
from pathlib import Path

# Charger le corpus
corpus_path = (
    Path("src")
    / "small_agentic_rag"
    / "data"
    / "corpus.txt"
)

with corpus_path.open("r", encoding="utf-8") as f:
    corpus = [line.strip().lower() for line in f if line.strip()]

# Construire le vocabulaire
all_tokens = []
for line in corpus:
    all_tokens.extend(line.split())

counter = Counter(all_tokens)

special_tokens = ["<unk>", "<eos>"]

vocab = {
    token: idx
    for idx, token in enumerate(special_tokens)
}

# Tri pour rendre la construction reproductible
for token, _ in sorted(counter.items()):
    if token not in vocab:
        vocab[token] = len(vocab)

vocab_size = len(vocab)

# Vocabulaire inverse : identifiant -> token
id_to_token = {
    idx: token for token, idx in vocab.items()
}


def encode(text: str) -> list[int]:
    tokens = text.lower().split()

    ids = [
        vocab.get(token, vocab["<unk>"])
        for token in tokens
    ]

    return ids


def decode(token_ids: list[int]) -> str:
    return " ".join(
        id_to_token.get(idx, "<unk>")
        for idx in token_ids
    )


# Ajouter EOS à chaque ligne pour marquer sa fin
all_ids = []

for line in corpus:
    all_ids.extend(encode(line))
    all_ids.append(vocab["<eos>"])

print("Taille du vocabulaire :", vocab_size)
print("Nombre total de tokens :", len(all_ids))
