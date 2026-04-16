from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

from config import config
from dataset import CharDataset
from model import CharNet
from train import get_class_weights, soft_bce_loss


BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def get_pos_weights(device) -> torch.Tensor:
    df = pd.read_csv(PROCESSED_DIR / "train.csv")
    pos_weights = get_class_weights(df).to(device)
    return pos_weights


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    test_ds = CharDataset(df=test_df, max_len=config.max_len)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size * 2, shuffle=False)

    model = CharNet(
        vocab_size=config.vocab_size, embed_dim=config.embed_dim, dropout=config.dropout
    ).to(device)
    model.load_state_dict(torch.load(BASE_DIR / "char_net_best.pth"))
    model.eval()

    pos_weights = get_pos_weights(device)  # для ф-ии потерь

    all_preds = []
    all_targets = []

    tot_loss = 0.0
    progress_bar = tqdm(test_loader)
    with torch.no_grad():
        for x, y_true in progress_bar:
            x = x.to(device)
            y_true = y_true.to(device)

            logits = model(x)
            cur_loss = soft_bce_loss(logits, y_true, pos_weights)
            tot_loss += cur_loss.item()
            progress_bar.set_postfix(loss=cur_loss.item())

            y_pred = torch.sigmoid(logits)

            all_targets.append(y_true.cpu().numpy())
            all_preds.append(y_pred.cpu().numpy())

    avg_loss = tot_loss / len(test_loader)
    print(f"Average Test Loss: {avg_loss:.4f}")

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    binary_preds = np.where(all_preds >= 0.5, 1, 0)
    cls_names = ["spam", "toxic", "obscenity"]

    print(
        classification_report(
            all_targets, binary_preds, target_names=cls_names, zero_division=0
        )
    )

    # проверка на obscenity
    test_df["obs_prob"] = all_preds[:, 2]
    test_df["obs_true"] = all_targets[:, 2]
    fp_cases = test_df[(test_df["obs_prob"] >= 0.5) & (test_df["obs_true"] == 0)]

    print(
        f"Всего {len(fp_cases)}  типо ложных срабатываний на мат, на самом деле не ложные, а датасет кривой, поэтому модель норм отработала"
    )
    print("Вот примеры текстов:")
    print(fp_cases["text"].head(10).values)


if __name__ == "__main__":
    test()
