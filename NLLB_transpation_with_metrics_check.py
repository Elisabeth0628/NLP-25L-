from transformers import AutoTokenizer, M2M100ForConditionalGeneration
import pandas as pd
from evaluate import load
import os
import sacrebleu
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM
from comet import download_model, load_from_checkpoint

tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
#model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
#tokenizer = AutoTokenizer.from_pretrained("facebook/m2m100_418M")


def test_translation(texts_to_translate):
    tokenizer.src_lang = "eng_Latn"
    translations = []

    for text in texts_to_translate:
        model_inputs = tokenizer(text, return_tensors="pt")
        gen_tokens = model.generate(
            **model_inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids("pol_Latn"),
            max_new_tokens=512,  # pozwala na dłuższe tłumaczenia
            temperature = 1.0,
            num_beams=15, # poprawia jakość
            no_repeat_ngram_size=3,  # usuwa powtórzenia
            #length_penalty=1.3,  # zachęca do pełniejszych tłumaczeń
            early_stopping=False
        )
        translated = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0]
        translations.append(translated)
        print(translated)

    return translations
    #assert translated is not None and len(translated[0]) > 0  # bardzo podstawowy test

def evaluate_comet(predictions, references, sources):
    comet = load("comet")
    result = comet.compute(
        predictions=predictions,
        references=references,
        sources=sources
    )
    print("\n COMET raw result:", result)

def sentence_bleu_scores(predictions, references):
    scores = []
    for pred, ref in zip(predictions, references):
        score = sacrebleu.sentence_bleu(pred, [ref]).score
        scores.append(score)
    return scores


def csv_read(file_path):
    df = pd.read_csv(file_path, sep=';')

    if not {"English", "Polish"}.issubset(df.columns):
        raise ValueError("CSV must contain 'English' and 'Polish' columns.")

    sources = df["English"].astype(str).str.strip().tolist()
    references = df["Polish"].astype(str).str.strip().tolist()

    predictions = test_translation(sources)

    if all(isinstance(p, str) for p in predictions):
        evaluate_comet(predictions, references, sources)
    else:
        print("Błąd: predykcje nie są typu `str`.")

    bleu_scores = sentence_bleu_scores(predictions, references)
    print("\nWyniki sacreBLEU dla każdego zdania:")
    for i, (src, ref, pred, score) in enumerate(zip(sources, references, predictions, bleu_scores), 1):
        print(f"{i:>2}. BLEU: {score:.2f}")
        print(f"    EN : {src}")
        print(f"    REF: {ref}")
        print(f"    PRED: {pred}\n")



def evaluate_translation_from_file(src_file, pred_file, ref_file):
    if not all(os.path.exists(p) for p in [src_file, pred_file, ref_file]):
        raise FileNotFoundError("Upewnij się, że wszystkie pliki istnieją: src, prediction, reference")

    with open(src_file, "r", encoding="utf-8") as f:
        source = f.read().strip()
    with open(pred_file, "r", encoding="utf-8") as f:
        prediction = f.read().strip()
    with open(ref_file, "r", encoding="utf-8") as f:
        reference = f.read().strip()

    print(" Tekst źródłowy:\n", source)
    print("\nTłumaczenie modelu:\n", prediction)
    print("\nTłumaczenie referencyjne:\n", reference)

    # COMET
    print("\n COMET:")
    comet = load("comet")
    comet_result = comet.compute(predictions=[prediction], references=[reference], sources=[source])
    print(" COMET raw result:", comet_result)

    # sacreBLEU
    print("\nsacreBLEU:")
    bleu_score = sacrebleu.sentence_bleu(prediction, [reference]).score
    print(f" sacreBLEU: {bleu_score:.2f}")


def txt_read(file_path, rfile):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Nie znaleziono pliku: {file_path}")


    with open(file_path, "r", encoding="utf-8") as f:
        full_text = f.read().strip()

    print(" Wczytany tekst:\n", full_text)

    translated_texts = test_translation([full_text])
    prediction = translated_texts[0]
    print("\n Oryginał:\n", full_text)
    print("Tłumaczenie:\n", translated_texts)

    if not os.path.exists(rfile):
        print("\nBrak pliku z tłumaczeniem referencyjnym:", rfile)
        return

    with open(rfile, "r", encoding="utf-8") as f:
        reference = f.read().strip()


    comet = load("comet")
    comet_result = comet.compute(predictions=[prediction], references=[reference], sources=[full_text])
    print(" COMET raw result:", comet_result)

    print("\nsacreBLEU:")
    bleu_score = sacrebleu.sentence_bleu(prediction, [reference]).score
    print(f" sacreBLEU: {bleu_score:.2f}")

    print("\nReferencja:\n", reference)



if __name__ == "__main__":
    file = "random_english_3.csv"
    textFile = "test.txt"
    reference_file = "reference_harry.txt"
    transtaleFile = "translate_harry.txt"
    #csv_read(file)
    evaluate_translation_from_file(textFile, transtaleFile, reference_file)
    #txt_read(textFile, reference_file)