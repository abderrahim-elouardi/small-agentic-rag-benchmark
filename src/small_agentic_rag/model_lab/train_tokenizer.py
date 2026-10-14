
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, normalizers
from tokenizers import pre_tokenizers, trainers


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CORPUS_PATH =  PROJECT_ROOT / "src" /"small_agentic_rag"/"data" / "corpus.txt"
TOKENIZER_PATH = (
    PROJECT_ROOT / "models" / "tokenizer.json"
)

VOCAB_SIZE = 16_000


def main():
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"Corpus introuvable : {CORPUS_PATH}"
        )

    TOKENIZER_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer = Tokenizer(
        models.BPE(unk_token="[UNK]")
    )

    # Normalisation Unicode
    tokenizer.normalizer = normalizers.NFC()

    # Découpage compatible avec BPE au niveau des octets
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False
    )

    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=[
            "[UNK]",
            "[PAD]",
            "[BOS]",
            "[EOS]",
        ],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    print("Entraînement du tokenizer BPE...")
    tokenizer.train(
        files=[str(CORPUS_PATH)],
        trainer=trainer,
    )

    tokenizer.save(str(TOKENIZER_PATH))

    print(f"Tokenizer sauvegardé : {TOKENIZER_PATH}")
    print(f"Taille réelle du vocabulaire : "
          f"{tokenizer.get_vocab_size()}")

    # Tester l'encodage et le décodage
    sample = "La science des données permet d'analyser le texte."
    encoded = tokenizer.encode(sample)

    print("\nTexte original :", sample)
    print("Tokens :", encoded.tokens)
    print("IDs :", encoded.ids)
    print("Texte décodé :", tokenizer.decode(encoded.ids))


if __name__ == "__main__":
    main()
