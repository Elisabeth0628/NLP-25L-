from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "utter-project/EuroLLM-1.7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

input_path = "harry_potter.txt"  # Replace with your input .txt file path
output_path = "translation_report_long_text.txt"

with open(input_path, "r", encoding="utf-8") as f:
    full_text = f.read()

prompt = (
    "<|im_start|>system\n<|im_end|>\n"
    "<|im_start|>user\n"
    "Translate the following English source text to Polish:\n"
    f"English:\n{full_text}\nPolish: <|im_end|>\n<|im_start|>assistant\n"
)

inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
outputs = model.generate(**inputs, max_new_tokens=2048)  # adjust if needed
decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

polish_translation = decoded_output.split("Polish:")[-1].strip()

with open(output_path, "w", encoding="utf-8") as f:
    f.write(polish_translation)

print(f"Full translation complete. Saved to '{output_path}'")
