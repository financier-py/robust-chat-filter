from dataclasses import dataclass


@dataclass
class Config:
    max_len: int = 512
    alphabet: str = "abcdefghijklmnopqrstuvwxyzабвгдеёжзийклмнопрстуфхцчшщъыьэюя0123456789 -,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "
    vocab_size = len(alphabet)
    embed_dim = 16


config = Config()