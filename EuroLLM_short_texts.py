import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sacrebleu
from comet import download_model, load_from_checkpoint

model_id = "utter-project/EuroLLM-1.7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Zmiana na nowy plikk zawierajacy 100 zdan do przetlumaczenia
csv_path = "100_sentences.csv"
df = pd.read_csv(csv_path, sep=';')

if not {"English", "Polish"}.issubset(df.columns):
    raise ValueError("CSV must contain 'English' and 'Polish' columns.")

comet_model_path = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(comet_model_path)

translations = []
sacrebleu_refs = []
sacrebleu_preds = []
comet_data = []

for idx, row in df.iterrows():
    src = str(row["English"])
    ref = str(row["Polish"])

    prompt = (
        "<|im_start|>system\n<|im_end|>\n"
        "<|im_start|>user\n"
        "Translate the following English source text to Polish:\n"
        f"English:\n{src}\nPolish: <|im_end|>\n<|im_start|>assistant\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=1,
        top_p=0.9,
        top_k=50,
        num_beams=15,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    translated = decoded.split("Polish:")[-1].strip()

    translations.append(f"English: {src}\nPredicted: {translated}\nReference: {ref}\n\n")
    sacrebleu_preds.append(translated)
    sacrebleu_refs.append([ref])  # list of references
    comet_data.append({"src": src, "mt": translated, "ref": ref})

# SacreBLEU
bleu = sacrebleu.corpus_bleu(sacrebleu_preds, sacrebleu_refs)

# COMET
comet_score = comet_model.predict(comet_data, batch_size=8, gpus=1 if torch.cuda.is_available() else 0)
comet_avg = sum(comet_score.scores) / len(comet_score.scores)

with open("translation_100_with_parameters.txt", "w", encoding="utf-8") as f:
    f.writelines(translations)
    f.write(f"\n---\nSacreBLEU: {bleu.score:.2f}\nCOMET: {comet_avg:.4f}\n")

print(f"Translations complete with metrics! Output saved to 'translation_with_metrics_4.txt'")
