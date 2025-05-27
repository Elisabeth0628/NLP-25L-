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

csv_path = "random_english_2.csv"
df = pd.read_csv(csv_path, sep=';')

if not {"English", "Polish"}.issubset(df.columns):
    raise ValueError("CSV must contain 'English' and 'Polish' columns.")

comet_model_path = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(comet_model_path)

translations = []

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

    # SacreBLEU
    bleu = sacrebleu.sentence_bleu(translated, [ref])

    # COMET
    comet_input = [{"src": src, "mt": translated, "ref": ref}]
    comet_score = comet_model.predict(comet_input, batch_size=1, gpus=1 if torch.cuda.is_available() else 0)

    translations.append(
        f"English: {src}\n"
        f"Predicted: {translated}\n"
        f"Reference: {ref}\n"
        f"SacreBLEU: {bleu.score:.2f}\n"
        f"COMET: {comet_score.scores[0]:.4f}\n\n"
    )

with open("translation_with_metrics_individual.txt", "w", encoding="utf-8") as f:
    f.writelines(translations)

print("Translations with individual metrics complete! Output saved to 'translation_with_metrics_individual.txt'")
