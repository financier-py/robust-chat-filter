from dataclasses import dataclass


@dataclass
class Config:
    max_len: int = 512
    alphabet: str = "abcdefghijklmnopqrstuvwxyzабвгдеёжзийклмнопрстуфхцчшщъыьэюя0123456789 -,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "

    dropout: float = 0.4
    embed_dim: int = 16
    batch_size: int = 128
    lr: float = 1e-3
    epochs: int = 3

    seed: int = 67

    @property
    def vocab_size(self) -> int:
        return len(self.alphabet) + 2


config = Config()
