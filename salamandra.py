import csv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from pathlib import Path

MODEL_ID = "BSC-LT/salamandraTA-2b"
SRC_LANG = "English"
TGT_LANG = "Polish"
FILE_PATH = "100_sentences.csv"
INPUT_CSV = Path(FILE_PATH).expanduser().resolve()
OUTPUT_CSV = INPUT_CSV.with_name(INPUT_CSV.stem + "_translated2.csv")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def translate(sentence: str) -> str:
    prompt = f"[{SRC_LANG}] {sentence} \n[{TGT_LANG}]"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids
        )
    return tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()


def read_csv(path: Path) -> pd.DataFrame:
    print(f"Reading from {path}")
    with open(path, "r", encoding="utf-8", newline="") as f:
        sample = f.read(2048)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample)
        delimiter = dialect.delimiter
        quotechar = dialect.quotechar
    return pd.read_csv(path, header=None, sep=delimiter, quotechar=quotechar, engine="python", dtype=str)

try:
    df = read_csv(INPUT_CSV)
except Exception as e:
    print("CSV read failed:", e)
    raise SystemExit(1)

missing_cols = 3 - df.shape[1]
if missing_cols > 0:
    for _ in range(missing_cols):
        df[df.shape[1]] = ""

print("Translating")

df.iloc[:, 2] = df.iloc[:, 0].apply(translate)

print(f"Saving to {OUTPUT_CSV}")

df.to_csv(OUTPUT_CSV, index=False, header=False)

print("Done.")
