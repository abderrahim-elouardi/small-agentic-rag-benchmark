from pathlib import Path
from datasets import load_dataset


def main():
    print("Chargement du corpus français...")

    # Streaming : on ne télécharge pas tout le dataset
    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.fr",
        split="train",
        streaming=True,
    )

    paragraphs = []
    max_paragraphs = 2000

    for row in dataset:
        text = row.get("text", "").strip()

        if text:
            paragraphs.append(text)

        if len(paragraphs) >= max_paragraphs:
            break

    if not paragraphs:
        raise ValueError("Aucun texte n'a été récupéré.")

    corpus_text = "\n\n".join(paragraphs)

    project_root = Path(__file__).resolve().parents[3]
    output_path = project_root / "src" /"data" / "corpus.txt"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(corpus_text, encoding="utf-8")

    print(f"Nombre de paragraphes : {len(paragraphs)}")
    print(f"Nombre de caractères : {len(corpus_text):,}")
    print(f"Fichier créé : {output_path}")
    print("\nAperçu du corpus :\n")
    print(corpus_text[:1000])


if __name__ == "__main__":
    main()