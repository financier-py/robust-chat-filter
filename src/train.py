import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

from dataset import CharDataset
from model import CharNet
from config import config

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def get_class_weights(df: pd.DataFrame):
    num_samples = len(df)
    pos_weights = []
    for col in ["spam", "toxic", "obscenity"]:
        num_cls = df[col].sum()
        weight = np.log(num_samples / (num_cls + 1))
        pos_weights.append(weight)
    return torch.tensor(pos_weights, dtype=torch.float32)


def soft_bce_loss(y_pred, y_true, pos_w):
    bce = F.binary_cross_entropy_with_logits(
        y_pred, y_true, pos_weight=pos_w, reduction='none'
    )

    with torch.no_grad():
        probs = torch.sigmoid(y_pred)
    
    false_obs = (y_true[:, 2] == 0) & (y_true[:, 1] == 1) & (probs[:, 2] > 0.5)
    bce[false_obs, 2] *= 0.3
    return bce.mean()


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Проверка проверка {device}")

    df = pd.read_csv(PROCESSED_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DIR / "val.csv")
    print("датасет загружен!!")

    pos_weights = get_class_weights(df).to(device)
    print(f"Вот веса классов: {pos_weights}")

    dataset = CharDataset(df, max_len=config.max_len)
    val_dataset = CharDataset(val_df, max_len=config.max_len)

    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size * 2, shuffle=False
    )

    model = CharNet(config.vocab_size, config.embed_dim, config.dropout).to(device)

    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    optimizer = AdamW(params=model.parameters(), lr=config.lr)

    best_val_loss = float("inf")
    for epoch in range(config.epochs):
        model.train()
        tot_loss = 0.0

        progress_bar = tqdm(dataloader, desc=f"Эпоха {epoch + 1}/{config.epochs}")

        for batch_idx, (x, y_true) in enumerate(progress_bar):
            x = x.to(device)
            y_true = y_true.to(device)

            optimizer.zero_grad()
            y_pred = model(x)
            cur_loss = soft_bce_loss(y_pred, y_true, pos_weights)
            cur_loss.backward()
            optimizer.step()

            tot_loss += cur_loss.item()
            progress_bar.set_postfix(loss=cur_loss.item())

        avg_tr_loss = tot_loss / len(dataloader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Эпоха {epoch + 1}/{config.epochs} [VAL]")

            for x, y_true in val_bar:
                x = x.to(device)
                y_true = y_true.to(device)
                y_pred = model(x)
                cur_loss = soft_bce_loss(y_pred, y_true, pos_weights)
                val_loss += cur_loss.item()
                val_bar.set_postfix(loss=cur_loss.item())

        avg_val_loss = val_loss / len(val_loader)
        print(
            f"\nЭпоха {epoch + 1} | Train Loss: {avg_tr_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = BASE_DIR / "char_net_best.pth"
            torch.save(model.state_dict(), save_path)
            print("Моделька была сохранена :)")


if __name__ == "__main__":
    train()
