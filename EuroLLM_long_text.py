from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sacrebleu
from comet import download_model, load_from_checkpoint

model_id = "utter-project/EuroLLM-1.7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

input_path = "poetry.txt"        # plik ze źródłem po angielsku
ref_path = "poetry_ref.txt"      # plik z referencyjnym tłumaczeniem (po polsku)
output_path = "translation_report_poetry_tongue.txt"

with open(input_path, "r", encoding="utf-8") as f:
    full_text = f.read()

with open(ref_path, "r", encoding="utf-8") as f:
    reference_text = f.read()

prompt = (
    "<|im_start|>system\n<|im_end|>\n"
    "<|im_start|>user\n"
    "Translate the following English source text to Polish:\n"
    f"English:\n{full_text}\nPolish: <|im_end|>\n<|im_start|>assistant\n"
)

inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
outputs = model.generate(
        **inputs,
        max_new_tokens=2048,
        temperature=1.9,
        top_p=0.1,
        top_k=50,
        num_beams=15,
        no_repeat_ngram_size=3,
        early_stopping=True
    )
decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

polish_translation = decoded_output.split("Polish:")[-1].strip()

with open(output_path, "w", encoding="utf-8") as f:
    f.write(polish_translation)

print(f"Full translation complete. Saved to '{output_path}'")

# SacreBLEU
bleu_score = sacrebleu.corpus_bleu([polish_translation], [[reference_text]])
print(f"SacreBLEU score: {bleu_score.score:.2f}")

# COMET
comet_model_path = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(comet_model_path)

comet_input = [{
    "src": full_text,
    "mt": polish_translation,
    "ref": reference_text
}]

comet_output = comet_model.predict(comet_input, batch_size=1, gpus=0)
print(f"COMET score: {comet_output.scores[0]:.4f}")
