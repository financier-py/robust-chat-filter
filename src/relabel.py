import json
import ollama
import pandas as pd
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL = "mistral-nemo"
FILES_TO_PROCESS = ["train.csv", "val.csv"]


def get_labels(text: str):
    prompt = """Ты модератор. Оцени текст и выдай вероятности от 0.0 до 1.0.
    spam: реклама, бессмысленный мусор.
    toxic: агрессия, оскорбления, угрозы.
    obscenity: мат, нецензурная лексика.
    
    Отвечай СТРОГО в формате JSON. Пример: {"spam": 0.0, "toxic": 0.8, "obscenity": 0.0}"""

    try:
        response = ollama.chat(
            model=MODEL,
            format="json",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ],
            options={"temperature": 0},
        )
        return json.loads(response["message"]["content"])
    except Exception:
        return None


def process_file(filename: str):
    print("размечаем", filename)
    input_path = PROCESSED_DIR / filename
    out_path = PROCESSED_DIR / f"llm_{filename}"

    df = pd.read_csv(input_path)

    pd.DataFrame(columns=df.columns).to_csv(out_path, index=False)

    for index, row in tqdm(df.iterrows(), total=len(df)):
        text = str(row["text"])

        labels = get_labels(text)

        if labels:
            row["spam"] = labels.get("spam", row["spam"])
            row["toxic"] = labels.get("toxic", row["toxic"])
            row["obscenity"] = labels.get("obscenity", row["obscenity"])
        pd.DataFrame([row]).to_csv(out_path, mode="a", header=False, index=False)


if __name__ == "__main__":
    for file in FILES_TO_PROCESS:
        process_file(file)
    print("Everything has been done :)")
