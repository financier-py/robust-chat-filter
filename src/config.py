from dataclasses import dataclass


@dataclass
class Config:
    max_len: int = 512
    alphabet: str = "abcdefghijklmnopqrstuvwxyzабвгдеёжзийклмнопрстуфхцчшщъыьэюя0123456789 -,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "

    dropout: float = 0.4
    embed_dim: int = 64
    batch_size: int = 1024
    lr: float = 5e-4
    epochs: int = 20

    seed: int = 67

    @property
    def vocab_size(self) -> int:
        return len(self.alphabet) + 2


config = Config()
