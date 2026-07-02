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

if __name__ == "__main__":
    tokenizer, model = load_tokenizer_and_model(MODEL)
    source_text = "Hello, world! What can I do for you today?"
    translated_text = translate_text(source_text, ENGLISH_TAG, UZBEK_TAG, tokenizer, model)
    print(f"Translated text: {translated_text}")



