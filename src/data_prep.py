from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
URL_SPAM_HF = (
    "hf://datasets/ruSpamModels/russian-spam-detection/processed_combined.parquet"
)


def load_telegram_spam(filepath: Path) -> pd.DataFrame:
    df = pd.read_csv(filepath, usecols=["text"])
    return df.assign(spam=1, toxic=0, obscenity=0)
    # можно оптимальнее
    # df = pd.read_csv(filepath)
    # return pd.DataFrame({"text": df["text"], "spam": 1, "toxic": 0, "obscenity": 0})


def load_pikabu(filepath: Path) -> pd.DataFrame:
    df = (
        pd.read_csv(filepath)
        .rename(columns={"comment": "text"})
        .assign(spam=0, toxic=lambda x: x["toxic"].astype(int), obscenity=0)
    )
    return df


def load_ok_ru(filepath: Path) -> pd.DataFrame:
    texts, toxic, obscenity = [], [], []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            texts.append(parts[1])

            if "NORMAL" in parts[0]:
                toxic.append(0)
                obscenity.append(0)
            elif (
                "INSULT" in parts[0] or "THREAT" in parts[0]
            ) and "OBSCENITY" not in parts[0]:
                toxic.append(1)
                obscenity.append(0)
            elif "OBSCENITY" in parts[0] and (
                not ("INSULT" in parts[0] or "THREAT" in parts[0])
            ):
                toxic.append(0)
                obscenity.append(1)
            else:
                toxic.append(1)
                obscenity.append(1)
    return pd.DataFrame(
        {"text": texts, "spam": 0, "toxic": toxic, "obscenity": obscenity}
    )


# решил добавить еще, тк жуткий дисбаланс классов вышел
def load_hf_spam() -> pd.DataFrame:
    return (
        pd.read_parquet(URL_SPAM_HF)
        .query("label == 1")
        .rename(columns={"message": "text", "label": "spam"})
        .assign(toxic=0, obscenity=0)
        .reset_index(drop=True)
        .sample(25_000)
    )


if __name__ == "__main__":
    df_spam_tg = load_telegram_spam(RAW_DIR / "telegram_spam.csv")
    df_spam_hf = load_hf_spam()
    df_pikabu = load_pikabu(RAW_DIR / "toxic_russian_comments_pikabu.csv")
    df_ok = load_ok_ru(RAW_DIR / "toxic_russian_comments_ok_ru.txt")

    print(f"размер df_spam телега: {len(df_spam_tg)}")
    print(f"размер df_spam hugging face: {len(df_spam_hf)}")

    final_df = pd.concat([df_spam_tg, df_ok, df_pikabu, df_spam_hf], ignore_index=True)
    final_df = final_df.drop_duplicates(subset=["text"])
    final_df = final_df.dropna(subset=["text"])

    final_df.to_csv(PROCESSED_DIR / "train.csv", index=False)

    print("кол-во строк:", len(final_df))
    print("кол-во токсиков:", final_df["toxic"].sum())
    print("кол-во спама:", final_df["spam"].sum())
    print("кол-во похабщины:", final_df["obscenity"].sum())
