import json

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from datasets import load_dataset
import evaluate
import torch
from tqdm import tqdm

DUETA_MODEL_PATH = "davron04/gemma-3-270m-dueta"
FLORES_DATASET_PATH = "openlanguagedata/flores_plus"
RESULTS_FILE = "bleu_result.json"
language_mapping = {
    "english": {
        "flores_code": "eng_Latn",
        "dueta_code": "<english>"
    },
    "uzbek": {
        "flores_code": "uzn_Latn",
        "dueta_code": "<uzbek>"
    }
}

def load_dueta_model():
    tokenizer = AutoTokenizer.from_pretrained(DUETA_MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(DUETA_MODEL_PATH, device_map="auto")
    model.eval()
    return tokenizer, model

def load_flores_dataset(lang, split):
    dataset = load_dataset(FLORES_DATASET_PATH, lang, split=split)
    return dataset

def load_bleu_metric():
    bleu = evaluate.load("sacrebleu")
    return bleu

def evaluate_for_lang_pair(model, tokenizer, metric, source_lang, target_lang):
    flores_src_code = language_mapping[source_lang]["flores_code"]
    flores_tgt_code = language_mapping[target_lang]["flores_code"]
    dueta_src_code = language_mapping[source_lang]["dueta_code"]
    dueta_tgt_code = language_mapping[target_lang]["dueta_code"]
    source_split = load_flores_dataset(flores_src_code, "devtest")["text"]
    target_split = load_flores_dataset(flores_tgt_code, "devtest")["text"]

    for src_text, tgt_text in tqdm(zip(source_split, target_split), total=len(source_split), desc=f"Evaluating {source_lang} to {target_lang}"):
        input_text = f"""{dueta_src_code}: {src_text}\n{dueta_tgt_code}"""
        inputs = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
        input_length = inputs.shape[1]
        
        with torch.no_grad():
            outputs = model.generate(inputs)
        
        predicted_token_ids = outputs[0][input_length:]
        predicted_text = tokenizer.decode(predicted_token_ids, skip_special_tokens=True)
        metric.add(
            predictions=predicted_text,
            references=tgt_text
        )

    bleu_score = metric.compute()
    return bleu_score

def save_results_to_json(results, model_name, src_lang, tgt_lang, bleu_result, filename=RESULTS_FILE):
    print(f"BLEU score of {model_name} for {src_lang} to {tgt_lang}: {bleu_result}")
    result_json = {
        "model": model_name,
        "source_language": src_lang,
        "target_language": tgt_lang,
        "bleu_result": bleu_result
    }
    results.append(result_json)
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)
    return results

if __name__ == "__main__":
    tokenizer, model = load_dueta_model()
    bleu_metric = load_bleu_metric()
    results = []

    source_language = "english"
    target_language = "uzbek"
    bleu_result = evaluate_for_lang_pair(model, tokenizer, bleu_metric, source_language, target_language)
    results = save_results_to_json(results, DUETA_MODEL_PATH, source_language, target_language, bleu_result)

    source_language = "uzbek"
    target_language = "english"
    bleu_result = evaluate_for_lang_pair(model, tokenizer, bleu_metric, source_language, target_language)
    results = save_results_to_json(results, DUETA_MODEL_PATH, source_language, target_language, bleu_result)

    print(f"Evaluation completed. Results saved to {RESULTS_FILE}.")