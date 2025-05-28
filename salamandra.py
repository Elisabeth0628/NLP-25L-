import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = 'BSC-LT/salamandraTA-2b'

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

src_lang_code = 'English'
tgt_lang_code = 'Polish'
sentence = "She got hiccups and couldn't get rid of them for three hours."

prompt = f'[{src_lang_code}] {sentence} \n[{tgt_lang_code}]'

input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
output_ids = model.generate(input_ids, num_beams=15, length_penalty=1.3, no_repeat_ngram_size=3, max_new_tokens=256, early_stopping=True)
input_length = input_ids.shape[1]

generated_text = tokenizer.decode(output_ids[0, input_length:], skip_special_tokens=True).strip()
print(generated_text)
