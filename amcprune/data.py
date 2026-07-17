import torch
from datasets import load_dataset


def _detect_text_column(dataset):
    if "text" in dataset.column_names:
        return "text"
    for column in dataset.column_names:
        if isinstance(dataset[0].get(column), str):
            return column
    raise ValueError("Could not find a text column in the dataset.")


def _group_texts(examples, block_size):
    concatenated = {key: sum(examples[key], []) for key in examples.keys()}
    first_key = next(iter(concatenated))
    total_length = len(concatenated[first_key])
    total_length = (total_length // block_size) * block_size
    if total_length == 0:
        return {key: [] for key in concatenated}
    result = {
        key: [
            values[index:index + block_size]
            for index in range(0, total_length, block_size)
        ]
        for key, values in concatenated.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def load_tokenized_text_dataset(
    tokenizer,
    dataset_name="wikitext",
    dataset_config="wikitext-2-raw-v1",
    split="test",
    max_samples=128,
    seq_len=128,
):
    dataset = load_dataset(dataset_name, dataset_config, split=split)
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    text_column = _detect_text_column(dataset)

    def tokenize_function(examples):
        return tokenizer(examples[text_column])

    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing text for causal LM",
    )
    grouped = tokenized.map(
        lambda examples: _group_texts(examples, seq_len),
        batched=True,
        desc=f"Grouping texts into chunks of {seq_len}",
    )
    if len(grouped) == 0:
        raise ValueError(
            "No token chunks were produced. Increase max_samples or lower seq_len."
        )

    input_ids = torch.tensor(grouped["input_ids"], dtype=torch.long)
    if "attention_mask" in grouped.column_names:
        attention_mask = torch.tensor(grouped["attention_mask"], dtype=torch.long)
    else:
        attention_mask = torch.ones_like(input_ids)
    return torch.utils.data.TensorDataset(input_ids, attention_mask)
