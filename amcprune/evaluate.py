import math

import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_perplexity(model, dataset, device, batch_size=1, max_batches=None):
    loader = DataLoader(dataset, batch_size=batch_size)
    total_loss = 0.0
    total_tokens = 0
    for step, (input_ids, attention_mask) in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )
        tokens = int(attention_mask.sum().item())
        total_loss += float(outputs.loss.item()) * tokens
        total_tokens += tokens
    mean_loss = total_loss / max(total_tokens, 1)
    return {
        "loss": mean_loss,
        "perplexity": math.exp(mean_loss) if mean_loss < 50 else float("inf"),
        "tokens": total_tokens,
    }
