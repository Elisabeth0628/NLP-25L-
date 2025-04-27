import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "utter-project/EuroLLM-1.7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

csv_path = "random_english.csv"  # Replace with your actual CSV file path
df = pd.read_csv(csv_path)

if "English" not in df.columns:
    raise ValueError("The CSV file must have a column named 'English'.")

translations = []

for idx, row in df.iterrows():
    english_text = str(row["English"])

    prompt = (
        "<|im_start|>system\n<|im_end|>\n"
        "<|im_start|>user\n"
        "Translate the following English source text to Polish:\n"
        f"English:\n{english_text}\nPolish: <|im_end|>\n<|im_start|>assistant\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=256)
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    polish_translation = decoded_output.split("Polish:")[-1].strip()
    translations.append(f"English: {english_text}\nPolish: {polish_translation}\n\n")

output_path = "translation_report_short_text.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.writelines(translations)

print(f"Translation complete! Saved to '{output_path}'")
