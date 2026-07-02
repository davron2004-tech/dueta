from dotenv import load_dotenv
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer

MODEL_NAME = "google/gemma-3-270m"
CORPUS_MAPPING = {
    "davron04/wikimedia_v20230407_en_uz": {
        "<english>": "english",
        "<uzbek>": "uzbek"
    },
    "MLDataScientist/SlimOrca-Dedup-English-Uzbek": {
        "<english>": "value",
        "<uzbek>": "translations"
    },
    "ML-Jonibek/English-Uzbek-Translation-1": {
        "<english>": "en",
        "<uzbek>": "uz"
    },
    "Jonibek21/English-Uzbek-Translation": {
        "<english>": "en",
        "<uzbek>": "uz"
    }
}
REMOTE_REPO = "davron04/dueta-tokenized"
ENGLISH_LANG = "<english>"
UZBEK_LANG = "<uzbek>"
TEST_SIZE = 0.01
NUM_PROCESSES = 4
SEED = 42

def tokenize_dataset(batch, tokenizer, src_lang, src_column, tgt_lang, tgt_column) -> None:
    texts = [
        f"{src_lang}: {s}\n{tgt_lang}: {t}"
        for s, t in zip(batch[src_column], batch[tgt_column])
        if s is not None and t is not None
    ]
    tokenized = tokenizer(texts, truncation=False, return_length=True)
    return {"input_ids": tokenized["input_ids"], "length": tokenized["length"]}

def tokenize_for_pair(tokenized_datasets, tokenizer, src_lang, tgt_lang):
    for dataset_name, column_mapping in CORPUS_MAPPING.items():
        src_column = column_mapping[src_lang]
        tgt_column = column_mapping[tgt_lang]
        dataset = load_dataset(dataset_name, split="train")
        if dataset_name == "MLDataScientist/SlimOrca-Dedup-English-Uzbek":
            print("Before SlimOrca filtering:", len(dataset))
            dataset = dataset.filter(lambda x: x["from"] != "system")
            print("After SlimOrca filtering:", len(dataset))
        tokenized_dataset = dataset.map(
            tokenize_dataset,
            fn_kwargs={
                "tokenizer": tokenizer,
                "src_lang": src_lang,
                "src_column": src_column,
                "tgt_lang": tgt_lang,
                "tgt_column": tgt_column
            },
            batched=True,
            num_proc=NUM_PROCESSES,
            remove_columns=dataset.column_names
        )
        tokenized_datasets.append(tokenized_dataset)
    return tokenized_datasets


if __name__ == "__main__":
    load_dotenv()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_bos_token = True
    tokenizer.add_eos_token = True

    tokenized_datasets = []
    print("Tokenizing English-Uzbek pairs...")
    tokenized_datasets = tokenize_for_pair(tokenized_datasets, tokenizer, ENGLISH_LANG, UZBEK_LANG)
    print("Tokenizing Uzbek-English pairs...")
    tokenized_datasets = tokenize_for_pair(tokenized_datasets, tokenizer, UZBEK_LANG, ENGLISH_LANG)

    merged_dataset = concatenate_datasets(tokenized_datasets)
    print("Merged dataset length:", len(merged_dataset))

    merged_dataset = merged_dataset.shuffle(seed=SEED)

    split_dataset = merged_dataset.train_test_split(test_size=TEST_SIZE, seed=SEED, shuffle=False)
    print(split_dataset)

    split_dataset.push_to_hub(REMOTE_REPO)