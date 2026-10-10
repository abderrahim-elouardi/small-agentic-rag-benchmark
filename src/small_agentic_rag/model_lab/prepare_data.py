from torch.utils.data import DataLoader

from small_agentic_rag.model_lab.dataset import NextTokenDataset
from small_agentic_rag.model_lab.embeddings import vocab_size , all_ids
from small_agentic_rag.model_lab.gpt_language_model import GPTLanguageModel


def main():
    # all_ids est la liste des IDs construite depuis corpus.txt
    print("Nombre de tokens :", len(all_ids))
    print("Taille du vocabulaire :", vocab_size)

    # Taille du contexte : nombre de tokens vus à chaque prédiction
    block_size = 16

    dataset = NextTokenDataset(
        token_ids=all_ids,
        block_size=block_size,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
    )

    # Examiner un batch
    input_ids, targets = next(iter(dataloader))

    print("Input IDs :", input_ids)
    print("Targets :", targets)
    print("Shape des entrées :", input_ids.shape)
    print("Shape des cibles :", targets.shape)




    model = GPTLanguageModel(
        vocab_size=vocab_size,
        num_layers=2,
        d_model=64,
        num_heads=4,
        d_ff=256,
        dropout=0.1,
    )

    # Étape 1 : IDs -> embeddings
    embeddings = model.token_embedding(input_ids)

    print("Shape des embeddings :", embeddings.shape)

    # Étape 2 : passage dans le modèle complet
    logits = model(input_ids)

    print("Shape des logits :", logits.shape)


if __name__ == "__main__":
    main()