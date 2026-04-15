import torch
from torch.utils.data import Dataset
import pandas as pd

from config import config


class CharDataset(Dataset):
    def __init__(self, df: pd.DataFrame, max_len: int):
        self.df = df
        self.max_len = max_len
        self.alphabet = config.alphabet

        self.char2idx = {char: idx + 2 for idx, char in enumerate(self.alphabet)}
        self.char2idx["PAD"] = 0  # пустота
        self.char2idx["UNK"] = 1  # не знаем

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        text = str(self.df.iloc[index]["text"]).lower()
        encoded = [self.char2idx.get(char, 1) for char in text]

        if len(encoded) > self.max_len:
            encoded = encoded[: self.max_len]
        else:
            encoded = encoded + [0] * (self.max_len - len(encoded))

        labels = self.df.iloc[index][["spam", "toxic", "obscenity"]].values.astype(
            float
        )
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(
            labels, dtype=torch.float
        )


# только для теста
# if __name__ == '__main__':
#     df = pd.read_csv('/home/financier/projects/robust-chat-filter/data/processed/train.csv', nrows=10)
#     dataset = CharDataset(df, 512)
#     x, y = dataset[0]

#     print(x.shape, y)
