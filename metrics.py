import sys, pathlib, pandas as pd
from sacrebleu import corpus_bleu

XL_PATH = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("test.xlsx")
df = pd.read_excel(XL_PATH, header=None, usecols=[0, 1, 2])
src  = df.iloc[:, 0].astype(str).tolist()
hyp  = df.iloc[:, 1].astype(str).tolist()
ref  = df.iloc[:, 2].astype(str).tolist()

assert len(src) == len(hyp) == len(ref) > 0, "Excel empty"

bleu = corpus_bleu(hyp, [ref])
print(f"SacreBLEU: {bleu.score:.2f}")
try:
    from comet import download_model, load_from_checkpoint

    ckpt = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(ckpt)
    data = [{"src": s, "mt": m, "ref": r} for s, m, r in zip(src, hyp, ref)]
    comet_out = comet_model.predict(data, batch_size=8, gpus=1)
    print(f"COMET-DA: {comet_out.system_score:.3f}")
except Exception as e:
    print("COMET failed", e)

for name, lines in zip(("src.txt", "hyp.txt", "ref.txt"), (src, hyp, ref)):
    pathlib.Path(name).write_text("\n".join(lines), encoding="utf-8")
