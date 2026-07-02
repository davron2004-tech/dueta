from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from dotenv import load_dotenv

BASE_MODEL_PATH = "google/gemma-3-270m"
ADAPTER_MODEL_PATH = "davron04/gemma-3-270m-dueta-base-adapter"
REMOTE_REPO = "davron04/gemma-3-270m-dueta"

def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

def merge_adapter_with_base_model(base_model_path, adapter_model_path, dtype="bfloat16"):
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, dtype=dtype)
    adapter_model = PeftModel.from_pretrained(base_model, adapter_model_path, dtype=dtype)
    merged_model = adapter_model.merge_and_unload()
    return merged_model

def get_generation_config(pad_token_id, eos_token_id):
    generation_config = GenerationConfig(
        max_length=32768,
        do_sample=False,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=5,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id
    )
    return generation_config

if __name__ == "__main__":
    load_dotenv()

    tokenizer = load_tokenizer(BASE_MODEL_PATH)
    merged_model = merge_adapter_with_base_model(BASE_MODEL_PATH, ADAPTER_MODEL_PATH)
    generation_config = get_generation_config(tokenizer.pad_token_id, tokenizer.eos_token_id)

    tokenizer.push_to_hub(REMOTE_REPO)
    merged_model.push_to_hub(REMOTE_REPO)
    generation_config.save_pretrained("generation_config", push_to_hub=True, repo_id=REMOTE_REPO)