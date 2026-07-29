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


def select_blocks_from_ranking(ranking, num_blocks, pruning_ratio):
    count = max(1, int(round(num_blocks * pruning_ratio)))
    count = min(count, max(num_blocks - 1, 1))
    return list(ranking)[:count]


def select_non_adjacent_blocks_from_ranking(ranking, num_blocks, pruning_ratio, min_gap=1):
    count = max(1, int(round(num_blocks * pruning_ratio)))
    count = min(count, max(num_blocks - 1, 1))
    min_gap = max(0, int(min_gap))

    ranking = [int(index) for index in ranking]
    selected = []
    for index in ranking:
        if len(selected) >= count:
            break
        if index < 0 or index >= num_blocks:
            continue
        if all(abs(index - other) > min_gap for other in selected):
            selected.append(index)

    if len(selected) < count:
        selected_set = set(selected)
        for index in ranking:
            if len(selected) >= count:
                break
            if index not in selected_set and 0 <= index < num_blocks:
                selected.append(index)
                selected_set.add(index)

    return sorted(selected)


def apply_block_skip(model, block_path, selected_indices):
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
    if hasattr(original, "__class__") and original.__class__.__name__ == "ModuleList":
        setattr(parent, attr, nn.ModuleList(wrapped))
    else:
        setattr(parent, attr, wrapped)
    return model


def _resolve_block_container(model, block_path):
    parent = model
    parts = block_path.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1], getattr(parent, parts[-1])


def _set_num_hidden_layers(model, num_layers):
    updated = []
    configs = [getattr(model, "config", None)]
    model_config = getattr(model, "config", None)
    if model_config is not None:
        configs.append(getattr(model_config, "text_config", None))
    for config in configs:
        if config is not None and hasattr(config, "num_hidden_layers"):
            config.num_hidden_layers = num_layers
            updated.append(type(config).__name__)
    return updated

def _repair_per_layer_config_lists(model, kept_indices, original_num_blocks):
    updated = []
    configs = [getattr(model, "config", None)]
    model_config = getattr(model, "config", None)
    if model_config is not None:
        configs.append(getattr(model_config, "text_config", None))
    seen = set()
    for config in configs:
        if config is None or id(config) in seen:
            continue
        seen.add(id(config))
        for name, value in list(vars(config).items()):
            if isinstance(value, list) and len(value) == original_num_blocks:
                setattr(config, name, [value[index] for index in kept_indices])
                updated.append(f"{type(config).__name__}.{name}")
    return updated


def _renumber_layer_indices(blocks):
    updated = []
    for new_index, block in enumerate(blocks):
        for module_name, module in block.named_modules():
            if hasattr(module, "layer_idx"):
                module.layer_idx = new_index
                updated.append(
                    f"{new_index}:{module_name or type(module).__name__}"
                )
    return updated


def remove_transformer_blocks(model, block_path, selected_indices):
    """Permanently remove transformer blocks and repair model topology metadata."""
    parent, attr, original = _resolve_block_container(model, block_path)
    num_blocks = len(original)
    selected = sorted(set(int(index) for index in selected_indices))
    invalid = [index for index in selected if index < 0 or index >= num_blocks]
    if invalid:
        raise IndexError(
            f"Block indices out of range for {block_path} ({num_blocks} blocks): {invalid}"
        )
    if len(selected) >= num_blocks:
        raise ValueError("Physical pruning must keep at least one transformer block.")

    selected_set = set(selected)
    kept_indices = [index for index in range(num_blocks) if index not in selected_set]
    kept_blocks = [original[index] for index in kept_indices]
    if isinstance(original, nn.ModuleList):
        new_container = nn.ModuleList(kept_blocks)
    else:
        new_container = original.__class__(kept_blocks)
    setattr(parent, attr, new_container)

    layer_indices_updated = _renumber_layer_indices(new_container)
    configs_updated = _set_num_hidden_layers(model, len(new_container))
    per_layer_configs_updated = _repair_per_layer_config_lists(
        model, kept_indices, num_blocks
    )
    return {
        "block_path": block_path,
        "original_num_blocks": num_blocks,
        "pruned_num_blocks": len(selected),
        "remaining_num_blocks": len(new_container),
        "removed_original_indices": selected,
        "kept_original_indices": kept_indices,
        "configs_updated": configs_updated,
        "per_layer_configs_updated": per_layer_configs_updated,
        "layer_indices_updated": layer_indices_updated,
    }


@contextmanager
def temporary_block_skip(model, blocks, block_path, selected_indices):
    parent = model
    parts = block_path.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    attr = parts[-1]
    original = getattr(parent, attr)
    try:
        apply_block_skip(model, block_path, selected_indices)
        yield
    finally:
        setattr(parent, attr, original)
