import torch
from torch.utils.data import Dataset


class NextTokenDataset(Dataset):
    def __init__(
        self,
        token_ids: list[int],
        block_size: int = 16,
    ):
        if len(token_ids) <= block_size:
            raise ValueError(
                "Le corpus doit contenir au moins "
                "block_size + 1 tokens."
            )

        self.token_ids = torch.tensor(
            token_ids,
            dtype=torch.long,
        )
        self.block_size = block_size

    def __len__(self):
        return len(self.token_ids) - self.block_size

    def __getitem__(self, idx):
        chunk = self.token_ids[
            idx : idx + self.block_size + 1
        ]

        input_ids = chunk[:-1]
        targets = chunk[1:]

        return input_ids, targets
