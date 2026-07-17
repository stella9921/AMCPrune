from contextlib import contextmanager

import torch.nn as nn


class SkipBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, hidden_states, *args, **kwargs):
        output = self.block(hidden_states, *args, **kwargs)
        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states


def rank_blocks(num_blocks, score="block_index"):
    if score == "block_index":
        return list(range(num_blocks - 1, -1, -1))
    if score == "early_block":
        return list(range(num_blocks))
    raise ValueError(f"Unknown block score: {score}")


def select_blocks(num_blocks, pruning_ratio, score="block_index"):
    count = max(1, int(round(num_blocks * pruning_ratio)))
    count = min(count, max(num_blocks - 1, 1))
    return rank_blocks(num_blocks, score=score)[:count]


@contextmanager
def temporary_block_skip(model, blocks, block_path, selected_indices):
    parent = model
    parts = block_path.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    attr = parts[-1]
    original = getattr(parent, attr)
    wrapped = list(original)
    selected = set(selected_indices)
    for index in selected:
        wrapped[index] = SkipBlock(wrapped[index])
    try:
        if hasattr(original, "__class__") and original.__class__.__name__ == "ModuleList":
            setattr(parent, attr, nn.ModuleList(wrapped))
        else:
            setattr(parent, attr, wrapped)
        yield
    finally:
        setattr(parent, attr, original)
