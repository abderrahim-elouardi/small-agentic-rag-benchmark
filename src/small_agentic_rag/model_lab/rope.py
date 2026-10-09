import torch 

def apply_rope(x , position):
    """
    x         : tenseur de forme (seq_len, d_model)
    positions : positions des tokens, de forme (seq_len,)
    """

    seq_len , d_model = x.shape

    # Une dimension paire est necessaire
    assert d_model % 2 == 0

    # Indices des paires : 0 , 2 , 4 , ...
    i = torch.arange(0,d_model , 2 , dtype=torch.float32)

    # Frequences de Rope : theta_i = 10000^(-2i/d_model)
    theta = 1.0 / ( 10000 ** (i / d_model))

    # Angle de rotation pour chaque position et chaque paire
    angles = position.float().unsqueeze(1) * theta.unsqueeze(0)

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    # Separer les deux composanres de chaque paire
    x1 = x[:, 0::2]
    x2 = x[:, 1::2]

    # Appliquer la rotation 
    rotated_x1 = x1 * cos - x2 *sin
    rotated_x2 = x1 * sin + x2 *cos


    # Reassembler les paires dnas l'ordre initial 
    result  = torch.stack(
        [rotated_x1 , rotated_x2] , dim = -1
    ).flatten(start_dim=1)

    return result

# Exemple : 3 tokens, embeddings de dimension 4
x = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],
    [1.0, 0.0, 1.0, 0.0],
    [1.0, 0.0, 1.0, 0.0]
])

positions = torch.arange(3)

result = apply_rope(x, positions)

print("Embeddings initiaux :")
print(x)

print("\nEmbeddings après RoPE :")
print(result)

        