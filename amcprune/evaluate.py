import math

import torch
import torch.nn.functional as F
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


@torch.no_grad()
def evaluate_preservation(
    model,
    dataset,
    device,
    apply_pruning,
    batch_size=1,
    max_batches=8,
):
    loader = DataLoader(dataset, batch_size=batch_size)
    hidden_cosine_sum = 0.0
    logit_kl_sum = 0.0
    batches = 0

    for step, (input_ids, attention_mask) in enumerate(loader):
        if step >= max_batches:
            break
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        dense_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        with apply_pruning():
            pruned_outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        dense_hidden = dense_outputs.hidden_states[-1].detach().float()
        pruned_hidden = pruned_outputs.hidden_states[-1].detach().float()
        mask = attention_mask.bool()
        hidden_cosine = F.cosine_similarity(
            dense_hidden[mask],
            pruned_hidden[mask],
            dim=-1,
        ).mean()

        dense_logits = dense_outputs.logits[:, :-1, :].detach().float()
        pruned_logits = pruned_outputs.logits[:, :-1, :].detach().float()
        next_token_mask = attention_mask[:, 1:].bool()
        dense_log_probs = F.log_softmax(dense_logits[next_token_mask], dim=-1)
        pruned_log_probs = F.log_softmax(pruned_logits[next_token_mask], dim=-1)
        kl = F.kl_div(
            pruned_log_probs,
            dense_log_probs.exp(),
            reduction="batchmean",
            log_target=False,
        )

        hidden_cosine_sum += float(hidden_cosine.item())
        logit_kl_sum += float(kl.item())
        batches += 1

    return {
        "hidden_cosine_similarity": hidden_cosine_sum / max(batches, 1),
        "logit_kl_divergence": logit_kl_sum / max(batches, 1),
        "batches": batches,
    }
