
import torch


def apply_rope(x: torch.Tensor, position: torch.Tensor | None = None):
    """
    Applique Rotary Positional Embedding (RoPE).

    Args:
        x: tenseur de forme (L, d_k) ou (B, L, d_k).
        position: positions des tokens, de forme (L,).

    Returns:
        Un tenseur de même forme que x.
    """
    if x.ndim not in (2, 3):
        raise ValueError(
            "x doit avoir 2 dimensions (L, d_k) "
            "ou 3 dimensions (B, L, d_k)."
        )

    seq_len = x.shape[-2]
    d_k = x.shape[-1]

    if d_k % 2 != 0:
        raise ValueError("d_k doit être pair pour appliquer RoPE.")

    if position is None:
        position = torch.arange(seq_len, device=x.device)
    else:
        position = position.to(device=x.device)

    if position.ndim != 1 or position.shape[0] != seq_len:
        raise ValueError(
            "position doit être un vecteur de longueur égale à L."
        )

    # Calcul des fréquences pour chaque paire de dimensions
    i = torch.arange(0, d_k, 2, device=x.device, dtype=torch.float32)
    theta = 1.0 / (10000.0 ** (i / d_k))

    # Forme : (L, d_k // 2)
    angles = position.float().unsqueeze(1) * theta.unsqueeze(0)

    cos = angles.cos().to(dtype=x.dtype)
    sin = angles.sin().to(dtype=x.dtype)

    # Pour un batch, ajouter la dimension batch : (1, L, d_k // 2)
    if x.ndim == 3:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    # Séparer les composantes paires et impaires
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]

    # Appliquer la rotation à chaque paire
    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos

    # Reconstituer la dernière dimension
    result = torch.stack(
        (rotated_x1, rotated_x2), dim=-1
    ).flatten(start_dim=-2)

    return result
