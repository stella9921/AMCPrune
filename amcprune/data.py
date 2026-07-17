import torch
from datasets import load_dataset


def load_tokenized_text_dataset(
    tokenizer,
    dataset_name="wikitext",
    dataset_config="wikitext-2-raw-v1",
    split="test",
    max_samples=128,
    seq_len=128,
):
    dataset = load_dataset(dataset_name, dataset_config, split=split)
    texts = [
        row["text"] for row in dataset
        if isinstance(row.get("text", None), str) and row["text"].strip()
    ]
    if max_samples:
        texts = texts[:max_samples]

    encoded = tokenizer(
        "\n\n".join(texts),
        return_tensors="pt",
        truncation=False,
    )
    input_ids = encoded["input_ids"][0]
    usable = (input_ids.numel() // seq_len) * seq_len
    input_ids = input_ids[:usable].view(-1, seq_len)
    attention_mask = torch.ones_like(input_ids)
    return torch.utils.data.TensorDataset(input_ids, attention_mask)

