import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return self.pool(x)


class CharNet(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int = 3):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.feature_make = nn.Sequential(
            ConvBlock(embed_dim, 64), ConvBlock(64, 128), ConvBlock(128, 256)
        )

        self.global_pool = nn.AdaptiveMaxPool1d(1)

        self.classifier = nn.Linear(256, num_classes)
    
    def forward(self, x: torch.Tensor):
        x = self.embedding(x)
        x = x.transpose(1, 2)

        x = self.feature_make(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)

        return self.classifier(x)
    
if __name__ == "__main__":
    # Тест на адекватность (Sanity check)
    vocab_size = 100 # Представим, что у нас 100 символов в алфавите
    model = CharNet(vocab_size=vocab_size, embed_dim=64)
    
    # Генерируем фейковый батч: 4 сообщения, каждое длиной 512 символов
    dummy_input = torch.randint(0, vocab_size, (4, 512))
    
    # Прогоняем через модель
    output = model(dummy_input)
    
    print("Вход:", dummy_input.shape)
    print("Выход:", output.shape) # Должно быть [4, 3]