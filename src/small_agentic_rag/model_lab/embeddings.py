
from pathlib import Path
from tokenizers import Tokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[3]

CORPUS_PATH = PROJECT_ROOT /"src"/"small_agentic_rag" /"data" / "corpus.txt"
TOKENIZER_PATH = PROJECT_ROOT / "models" / "tokenizer.json"

if not CORPUS_PATH.exists():
    raise FileNotFoundError(
        f"Corpus introuvable : {CORPUS_PATH}"
    )

if not TOKENIZER_PATH.exists():
    raise FileNotFoundError(
        "Tokenizer BPE introuvable. "
        "Exécute d'abord train_tokenizer.py."
    )

# Charger le corpus
text = CORPUS_PATH.read_text(encoding="utf-8")

if not text.strip():
    raise ValueError("Le corpus est vide.")

# Charger le tokenizer déjà entraîné
tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

# Construire le vocabulaire et les identifiants
vocab = tokenizer.get_vocab()
vocab_size = tokenizer.get_vocab_size()

# Encoder le corpus
all_ids = tokenizer.encode(text).ids

# Dictionnaire inverse utile pour l'inspection
id_to_token = {
    idx: token
    for token, idx in vocab.items()
}

print(f"Taille du vocabulaire BPE : {vocab_size:,}")
print(f"Nombre total de tokens : {len(all_ids):,}")
print(f"Exemple d'encodage : {all_ids[:30]}")
