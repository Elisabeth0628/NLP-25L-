import pandas as pd
from transformers import AutoTokenizer, M2M100ForConditionalGeneration


def translate_texts(input_texts, source_lang="en", target_lang="pl"):
    tokenizer = AutoTokenizer.from_pretrained("facebook/m2m100_418M")
    model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")

    tokenizer.src_lang = source_lang
    translations = []

    for text in input_texts:
        if pd.isna(text) or not isinstance(text, str):
            translations.append("")
            continue

        model_inputs = tokenizer(text, return_tensors="pt", truncation=True)
        gen_tokens = model.generate(
            **model_inputs,
            forced_bos_token_id=tokenizer.get_lang_id(target_lang)
        )
        translated = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0]
        translations.append(translated)

    return translations


def csv_read(file_path):
    # Wczytaj CSV (zakładam, że ma jedną kolumnę, np. "text")
    df = pd.read_csv(file_path)  # np. "teksty.csv"
    if "English" not in df.columns:
        raise ValueError("Plik CSV musi zawierać kolumnę o nazwie 'English'.")

    #texts = df["English"].tolist()
    #print(texts)
    example = df.iloc[0, 0]
    print(example)
    translated_texts = translate_texts(example, source_lang="en", target_lang="pl")
    #print(translated_texts)

    print("Tekst:", translated_texts, " ", "Tłumaczenie: ", example)

    #for original, translated in zip(texts, translated_texts):
    #    print(f"\n Oryginał: {original}")
    #    print(f" Tłumaczenie: {translated}")


if __name__ == "__main__":
    file = "yt_dataset.csv"
    csv_read(file)
