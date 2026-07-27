import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from amcprune.evaluate import evaluate_perplexity
from amcprune.pruning import temporary_block_skip


def _extract_hidden(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
        return output[0]
    return None


def _block_weight_abs_mean(block):
    total = 0.0
    count = 0
    for parameter in block.parameters():
        values = parameter.detach().float().abs()
        total += float(values.sum().cpu().item())
        count += values.numel()
    return total / max(count, 1)


def _flatten_hidden(hidden):
    if hidden is None or not torch.is_tensor(hidden):
        return None
    if hidden.dim() == 2:
        hidden = hidden.unsqueeze(0)
    if hidden.dim() != 3:
        return None
    return hidden.detach().float().reshape(-1, hidden.shape[-1])


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
def score_blocks_by_activation_weight(
    model,
    blocks,
    dataset,
    device,
    batch_size=1,
    max_batches=8,
):
    activation_rows = score_blocks_by_activation(
        model=model,
        blocks=blocks,
        dataset=dataset,
        device=device,
        batch_size=batch_size,
        max_batches=max_batches,
    )
    scores = []
    for row, block in zip(activation_rows, blocks):
        weight_abs_mean = _block_weight_abs_mean(block)
        activation_abs_mean = row["activation_abs_mean"]
        score = activation_abs_mean * weight_abs_mean
        scores.append({
            "block": row["block"],
            "activation_abs_mean": activation_abs_mean,
            "weight_abs_mean": weight_abs_mean,
            "score": score,
        })
    return scores


@torch.no_grad()
def score_blocks_by_hidden_cosine(
    model,
    blocks,
    dataset,
    device,
    batch_size=1,
    max_batches=8,
):
    sums = torch.zeros(len(blocks), dtype=torch.float64)
    counts = torch.zeros(len(blocks), dtype=torch.float64)
    hooks = []

    def make_hook(index):
        def hook(_, inputs, output):
            if not inputs:
                return
            hidden_in = _flatten_hidden(inputs[0])
            hidden_out = _flatten_hidden(_extract_hidden(output))
            if hidden_in is None or hidden_out is None:
                return
            size = min(hidden_in.shape[0], hidden_out.shape[0])
            if size == 0:
                return
            cosine = F.cosine_similarity(hidden_in[:size], hidden_out[:size], dim=-1)
            sums[index] += cosine.mean().cpu().double()
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
        similarity = float((sums[index] / counts[index]).item()) if counts[index] else 0.0
        representation_delta = 1.0 - similarity
        scores.append({
            "block": index,
            "hidden_cosine_similarity": similarity,
            "representation_delta": representation_delta,
            "score": representation_delta,
        })
    return scores


@torch.no_grad()
def score_blocks_by_hidden_cosine_streamline(
    model,
    blocks,
    dataset,
    device,
    layer_intervals,
    batch_size=1,
    max_batches=8,
):
    """LLM-Streamline-style contiguous depth score.

    It compares hidden_states[j] and hidden_states[j + layer_intervals] and
    selects the interval with the highest token-wise cosine similarity.
    """
    num_blocks = len(blocks)
    interval = max(int(layer_intervals), 1)
    interval = min(interval, max(num_blocks - 1, 1))
    sums = torch.zeros(num_blocks - interval + 1, dtype=torch.float64)
    counts = torch.zeros(num_blocks - interval + 1, dtype=torch.float64)

    loader = DataLoader(dataset, batch_size=batch_size)
    was_training = model.training
    model.eval()
    try:
        for step, (input_ids, attention_mask) in enumerate(loader):
            if step >= max_batches:
                break
            outputs = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                output_hidden_states=True,
                use_cache=False,
            )
            hidden_states = outputs.hidden_states
            if hidden_states is None:
                continue
            for start in range(num_blocks - interval + 1):
                source = _flatten_hidden(hidden_states[start])
                target = _flatten_hidden(hidden_states[start + interval])
                if source is None or target is None:
                    continue
                size = min(source.shape[0], target.shape[0])
                if size == 0:
                    continue
                cosine = F.cosine_similarity(source[:size], target[:size], dim=-1)
                sums[start] += cosine.mean().cpu().double()
                counts[start] += 1
    finally:
        model.train(was_training)

    interval_rows = []
    best_start = 0
    best_similarity = float("-inf")
    for start in range(num_blocks - interval + 1):
        similarity = float((sums[start] / counts[start]).item()) if counts[start] else 0.0
        if similarity > best_similarity:
            best_similarity = similarity
            best_start = start
        interval_rows.append({
            "interval_start": start,
            "interval_end": start + interval,
            "interval_length": interval,
            "streamline_similarity": similarity,
        })

    prune_set = set(range(best_start + 1, best_start + interval))
    scores = []
    for index in range(num_blocks):
        selected = index in prune_set
        if index < best_start + 1:
            distance_to_interval = (best_start + 1) - index
        elif index > best_start + interval - 1:
            distance_to_interval = index - (best_start + interval - 1)
        else:
            distance_to_interval = 0
        ranking_score = 0.0 if selected else 1.0 + float(distance_to_interval)
        scores.append({
            "block": index,
            "hidden_cosine_similarity": best_similarity if selected else 0.0,
            "representation_delta": 1.0 - best_similarity if selected else 1.0,
            "streamline_interval_start": best_start,
            "streamline_interval_end": best_start + interval,
            "streamline_interval_length": interval,
            "streamline_similarity": best_similarity if selected else 0.0,
            "streamline_selected": selected,
            "distance_to_streamline_interval": distance_to_interval,
            "streamline_interval_scores": interval_rows if index == 0 else None,
            "score": ranking_score,
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
