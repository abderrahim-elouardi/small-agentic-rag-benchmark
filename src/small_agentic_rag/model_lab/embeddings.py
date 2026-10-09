from collections import Counter
import torch.nn as nn
import torch 

corpus = [

]

with open("src\small_agentic_rag\data\corpus.txt", "r", encoding="utf-8") as f:
    for l in f:
        corpus.append(l.strip())


all_tokens = []
for l in corpus:
    tokens = l.lower().split()
    all_tokens.extend(tokens)

all_tokens = list(set(all_tokens))

counter = Counter(all_tokens)

special_tokens = ["<unk>"]

vocab = {
    token:idx
    for idx , token in enumerate(special_tokens)
}

for token in all_tokens:
    if token not in vocab:
        vocab[token] = len(vocab)

vocab_size = len(vocab)


class Token_not_in_vocab_Exception(Exception):
    def __init__(self):
        super()


def encode(text):
    text_tokens = text.split()
    text_tokens_ids = []
    for token in text_tokens:
        if token in vocab.keys():
            text_tokens_ids.append(vocab[token])
        
    return text_tokens_ids


def decode(text_tokens_ids):
    text_tokens = []
    for id in text_tokens_ids:
        text_tokens.append([value for key , value in enumerate(vocab) if key == id][0])
    return text_tokens


def ids_to_embedding(ids_list, d_model=64):
    embedding = nn.Embedding(vocab_size , d_model)
    token_ids_tensor = torch.tensor(ids_list)
    vector_embedding = embedding(token_ids_tensor)
    return vector_embedding



def text_to_embeddings(text , d_model=64):
    return ids_to_embedding(encode(text) , d_model)



