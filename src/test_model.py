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


BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    test_ds = CharDataset(df=test_df, max_len=config.max_len)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size * 8, shuffle=False)

    model = CharNet(
        vocab_size=config.vocab_size, embed_dim=config.embed_dim, dropout=config.dropout
    ).to(device)
    model.load_state_dict(torch.load(BASE_DIR / "char_net_best.pth"))
    model.eval()

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y_true in tqdm(test_loader):
            x = x.to(device)
            y_true = y_true.to(device)

            logits = model(x)
            y_pred = torch.sigmoid(logits)

            all_targets.append(y_true.cpu().numpy())
            all_preds.append(y_pred.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    binary_preds = np.where(all_preds >= 0.5, 1, 0)
    cls_names = ["spam", "toxic", "obscenity"]

    print(classification_report(all_targets, binary_preds, target_names=cls_names, zero_division=0))


if __name__ == "__main__":
    test()
