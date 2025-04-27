from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import csv


def load_texts(filepath):
    texts = []
    with open(filepath, newline='', encoding='utf-8') as csv_file:
        rd = csv.reader(csv_file)
        for line in rd:
            if line:
                texts.append(line[0])
    return texts


texts = load_texts("yt_dataset.csv")
print(f"Got texts from file: \n{texts}")
model_id = "Unbabel/TowerInstruct-7B-v0.2"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16
)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

for i in texts:
    prompt = f"<|prompter|>Translate to Polish: {i}<|endoftext|><|assistant|>"
    result = generator(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)[0]["generated_text"]
    print("Translated:")
    translated = result.split("<|assistant|>")[-1].strip()
    print(translated)
