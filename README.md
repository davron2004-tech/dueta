# gemma-3-270m-dueta

**DUETA** — a **D**ecoder-only, **U**zbek–**E**nglish **T**ransformer-based model for machine tr**A**nslation.

This model is a fine-tuned version of [google/gemma-3-270m](https://huggingface.co/google/gemma-3-270m), adapted for bidirectional English ↔ Uzbek translation using a decoder-only architecture. The approach follows the methodology described in *DIETA: A Decoder-only transformer-based model for Italian–English machine TrAnslation*, applying the same decoder-only translation paradigm to the English–Uzbek language pair.

## Model Details

- **Base model:** [google/gemma-3-270m](https://huggingface.co/google/gemma-3-270m)
- **Architecture:** Decoder-only transformer
- **Languages:** English (`en`), Uzbek (`uz`)
- **Task:** Machine translation (En→Uz and Uz→En)
- **Reference paper:** DIETA: A Decoder-only transformer-based model for Italian–English machine TrAnslation

## Training Data

The model was fine-tuned on a combination of the following datasets:

| Dataset | Source |
|---|---|
| `wikimedia-v20230407` | [OPUS](https://opus.nlpl.eu/) |
| `MLDataScientist/SlimOrca-Dedup-English-Uzbek` | Hugging Face |
| `ML-Jonibek/English-Uzbek-Translation-1` | Hugging Face |
| `Jonibek21/English-Uzbek-Translation` | Hugging Face |

## How to Use

The model uses a simple prompt format with language tags (`<english>` / `<uzbek>`) to indicate the source and target languages, and generates a translation in a decoder-only, causal-LM fashion.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "davron04/gemma-3-270m-dueta"
ENGLISH_TAG = "<english>"
UZBEK_TAG = "<uzbek>"

def load_tokenizer_and_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    return tokenizer, model

def translate_text(source_text: str, source_lang: str, target_lang: str, tokenizer, model) -> str:
    model_input = f"{source_lang}: {source_text}\n{target_lang}:"
    token_ids = tokenizer.encode(model_input, return_tensors="pt").to(model.device)
    input_token_count = token_ids.shape[1]
    output = model.generate(token_ids)
    generated_token_ids = output[0][input_token_count:]
    translated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    return translated_text

tokenizer, model = load_tokenizer_and_model(MODEL)
source_text = "Hello, world! What can I do for you today?"
translated_text = translate_text(source_text, ENGLISH_TAG, UZBEK_TAG, tokenizer, model)
print(f"Translated text: {translated_text}")

"""Salom, dunyo! Bugun siz uchun nima qilishim mumkin?"""
```

To translate from Uzbek to English, simply swap the `source_lang` and `target_lang` arguments:

```python
translated_text = translate_text(source_text, UZBEK_TAG, ENGLISH_TAG, tokenizer, model)
```

## Limitations

- **Context length:** Avoid feeding the model very long context. Performance degrades with long inputs; it is recommended to **split long text into individual sentences** before translation and process them one at a time (or in short chunks) rather than passing entire paragraphs or documents at once.
- As with any machine translation model, performance may vary across domains, informal/colloquial text, and low-resource constructs not well represented in the training data.
- The model has not been evaluated for factual accuracy preservation in translation; it should not be used for translating sensitive or critical content without human review.