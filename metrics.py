import csv
import sys
import pathlib
import pandas as pd
from sacrebleu import corpus_bleu

CSV_PATH = pathlib.Path("100_sentences_translated2.csv")
print(f"Loading {CSV_PATH}")

def read_csv(path: pathlib.Path) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample)
        delimiter = dialect.delimiter
        quotechar = dialect.quotechar
    return pd.read_csv(path, header=None, sep=delimiter, quotechar=quotechar, engine="python", dtype=str)

try:
    df = read_csv(CSV_PATH)
except Exception as e:
    sys.exit(f"CSV read failed: {e}")

src = df.iloc[:, 0].fillna("").astype(str).tolist()
ref = df.iloc[:, 1].fillna("").astype(str).tolist()
hyp = df.iloc[:, 2].fillna("").astype(str).tolist()

assert len(src) == len(ref) == len(hyp) > 0, "CSV empty or mismatched lengths"

# SacreBLEU
print("📊 Computing SacreBLEU…")
bleu = corpus_bleu(hyp, [ref])
print(f"SacreBLEU: {bleu.score:.2f}")

# COMET
try:
    from comet import download_model, load_from_checkpoint
    ckpt = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(ckpt)
    data = [{"src": s, "mt": m, "ref": r} for s, m, r in zip(src, hyp, ref)]
    comet_out = comet_model.predict(data, batch_size=8, gpus=1)
    print(f"COMET-DA: {comet_out.system_score:.3f}")
except Exception as e:
    print("COMET failed:", e)

for name, lines in zip(("src.txt", "hyp.txt", "ref.txt"), (src, hyp, ref)):
    pathlib.Path(name).write_text("\n".join(lines), encoding="utf-8")

print("Done")
