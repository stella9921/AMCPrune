import csv
import math
import os

import torch


@torch.no_grad()
def _iter_batches(dataset, batch_size, device, max_batches):
    total = len(dataset["input_ids"])
    limit = total if max_batches is None else min(total, max_batches * batch_size)
    for start in range(0, limit, batch_size):
        end = min(start + batch_size, limit)
        yield {
            "input_ids": dataset["input_ids"][start:end].to(device),
            "attention_mask": dataset["attention_mask"][start:end].to(device),
            "labels": dataset["labels"][start:end].to(device),
        }


def _detach_hidden_states(value):
    if isinstance(value, tuple):
        value = value[0]
    if not torch.is_tensor(value):
        return None
    if value.dim() == 2:
        value = value.unsqueeze(0)
    if value.dim() != 3:
        return None
    return value.detach().float().reshape(-1, value.shape[-1]).cpu()


class OATSSecondMomentAccumulator:
    """OATS-style input second moment tracker for a pruning unit.

    OATS accumulates squared L2 input statistics per input dimension before
    scaling weights by sqrt(second moment). Here we store the same signal as an
    analysis metric instead of directly modifying the pruning score.
    """

    def __init__(self):
        self.sum_sq = None
        self.count = 0

    def add(self, hidden_states):
        hidden_states = _detach_hidden_states(hidden_states)
        if hidden_states is None or hidden_states.numel() == 0:
            return
        squared_sum = hidden_states.pow(2).sum(dim=0)
        if self.sum_sq is None:
            self.sum_sq = torch.zeros_like(squared_sum)
        self.sum_sq += squared_sum
        self.count += hidden_states.shape[0]

    def second_moment(self):
        if self.sum_sq is None or self.count == 0:
            return None
        return self.sum_sq / float(self.count)


def _summarize_second_moment(block_index, block_name, selected, values, outlier_k):
    if values is None or values.numel() == 0:
        return {
            "block": block_index,
            "block_name": block_name,
            "selected_block": selected,
            "num_dimensions": 0,
            "second_moment_mean": 0.0,
            "second_moment_std": 0.0,
            "second_moment_max": 0.0,
            "second_moment_top1pct_mean": 0.0,
            "second_moment_q95": 0.0,
            "second_moment_q99": 0.0,
            "outlier_threshold": 0.0,
            "outlier_ratio": 0.0,
            "outlier_count": 0,
        }
    values = values.float()
    mean = values.mean()
    std = values.std(unbiased=False)
    threshold = mean + float(outlier_k) * std
    outlier_mask = values > threshold
    topk = max(1, int(math.ceil(values.numel() * 0.01)))
    return {
        "block": block_index,
        "block_name": block_name,
        "selected_block": selected,
        "num_dimensions": int(values.numel()),
        "second_moment_mean": float(mean.item()),
        "second_moment_std": float(std.item()),
        "second_moment_max": float(values.max().item()),
        "second_moment_top1pct_mean": float(torch.topk(values, topk).values.mean().item()),
        "second_moment_q95": float(torch.quantile(values, 0.95).item()),
        "second_moment_q99": float(torch.quantile(values, 0.99).item()),
        "outlier_threshold": float(threshold.item()),
        "outlier_ratio": float(outlier_mask.float().mean().item()),
        "outlier_count": int(outlier_mask.sum().item()),
    }


def measure_block_outliers(
    *,
    model,
    blocks,
    block_path,
    dataset,
    device,
    batch_size,
    max_batches,
    selected_blocks=None,
    outlier_k=3.0,
):
    selected = set(selected_blocks or [])
    accumulators = [OATSSecondMomentAccumulator() for _ in blocks]
    handles = []

    def make_hook(index):
        def hook(_, inputs, __):
            if inputs:
                accumulators[index].add(inputs[0])
        return hook

    for index, block in enumerate(blocks):
        handles.append(block.register_forward_hook(make_hook(index)))

    was_training = model.training
    model.eval()
    try:
        for batch in _iter_batches(dataset, batch_size, device, max_batches):
            model(**batch)
    finally:
        for handle in handles:
            handle.remove()
        if was_training:
            model.train()

    rows = []
    for index, accumulator in enumerate(accumulators):
        rows.append(_summarize_second_moment(
            block_index=index,
            block_name=f"{block_path}.{index}",
            selected=index in selected,
            values=accumulator.second_moment(),
            outlier_k=outlier_k,
        ))
    return rows


def save_outlier_metrics_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "block",
        "block_name",
        "selected_block",
        "num_dimensions",
        "second_moment_mean",
        "second_moment_std",
        "second_moment_max",
        "second_moment_top1pct_mean",
        "second_moment_q95",
        "second_moment_q99",
        "outlier_threshold",
        "outlier_ratio",
        "outlier_count",
    ]
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path