import torch
from torch.utils.data import DataLoader

from amcprune.evaluate import evaluate_perplexity
from amcprune.pruning import temporary_block_skip


def _extract_hidden(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
        return output[0]
    return None


@torch.no_grad()
def score_blocks_by_activation(model, blocks, dataset, device, batch_size=1, max_batches=8):
    sums = torch.zeros(len(blocks), dtype=torch.float64)
    counts = torch.zeros(len(blocks), dtype=torch.float64)
    hooks = []

    def make_hook(index):
        def hook(_, __, output):
            hidden = _extract_hidden(output)
            if hidden is None:
                return
            values = hidden.detach().float().abs()
            sums[index] += values.mean().cpu().double()
            counts[index] += 1
        return hook

    for index, block in enumerate(blocks):
        hooks.append(block.register_forward_hook(make_hook(index)))

    loader = DataLoader(dataset, batch_size=batch_size)
    was_training = model.training
    model.eval()
    try:
        for step, (input_ids, attention_mask) in enumerate(loader):
            if step >= max_batches:
                break
            model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
            )
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)

    scores = []
    for index in range(len(blocks)):
        score = float((sums[index] / counts[index]).item()) if counts[index] else 0.0
        scores.append({
            "block": index,
            "activation_abs_mean": score,
            "score": score,
        })
    return scores


@torch.no_grad()
def score_blocks_by_loss_delta(
    model,
    blocks,
    block_path,
    dataset,
    device,
    batch_size=1,
    max_batches=8,
):
    baseline = evaluate_perplexity(
        model,
        dataset,
        device=device,
        batch_size=batch_size,
        max_batches=max_batches,
    )
    scores = []
    for index in range(len(blocks)):
        with temporary_block_skip(model, blocks, block_path, [index]):
            skipped = evaluate_perplexity(
                model,
                dataset,
                device=device,
                batch_size=batch_size,
                max_batches=max_batches,
            )
        loss_delta = skipped["loss"] - baseline["loss"]
        scores.append({
            "block": index,
            "baseline_loss": baseline["loss"],
            "skipped_loss": skipped["loss"],
            "loss_delta": loss_delta,
            "baseline_perplexity": baseline["perplexity"],
            "skipped_perplexity": skipped["perplexity"],
            "score": loss_delta,
        })
    return scores


def rank_blocks_by_scores(score_rows, descending=False):
    return [
        row["block"] for row in sorted(
            score_rows,
            key=lambda item: item["score"],
            reverse=descending,
        )
    ]
