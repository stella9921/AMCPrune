import torch
import torch.nn as nn


def _linear(module, name):
    layer = getattr(module, name, None)
    weight = getattr(layer, "weight", None)
    return layer if weight is not None else None


def _find_attention(block):
    for name in ["self_attn", "attn", "attention", "self_attention"]:
        module = getattr(block, name, None)
        if module is not None:
            return module
    return None


def _find_mlp(block):
    for name in ["mlp", "feed_forward", "ffn", "MLP"]:
        module = getattr(block, name, None)
        if module is not None:
            return module
    return None


def _replace_linear(layer, *, keep_rows=None, keep_cols=None):
    if layer is None:
        return None, 0
    weight = layer.weight.detach()
    bias = layer.bias.detach() if getattr(layer, "bias", None) is not None else None
    if keep_rows is None:
        keep_rows = torch.arange(weight.shape[0], device=weight.device)
    else:
        keep_rows = torch.as_tensor(keep_rows, device=weight.device, dtype=torch.long)
    if keep_cols is None:
        keep_cols = torch.arange(weight.shape[1], device=weight.device)
    else:
        keep_cols = torch.as_tensor(keep_cols, device=weight.device, dtype=torch.long)

    new_weight = weight.index_select(0, keep_rows).index_select(1, keep_cols).contiguous()
    new_layer = nn.Linear(
        int(new_weight.shape[1]),
        int(new_weight.shape[0]),
        bias=bias is not None,
        device=weight.device,
        dtype=weight.dtype,
    )
    with torch.no_grad():
        new_layer.weight.copy_(new_weight)
        if bias is not None:
            new_layer.bias.copy_(bias.index_select(0, keep_rows).contiguous())
    removed = weight.numel() - new_weight.numel()
    if bias is not None:
        removed += bias.numel() - new_layer.bias.numel()
    return new_layer, int(removed)


def _set_linear(module, name, layer):
    if layer is not None and hasattr(module, name):
        setattr(module, name, layer)


def _zero_rows(layer, start, end):
    if layer is None:
        return 0
    count = 0
    with torch.no_grad():
        layer.weight[start:end, :] = 0
        count += layer.weight[start:end, :].numel()
        if getattr(layer, "bias", None) is not None:
            layer.bias[start:end] = 0
            count += layer.bias[start:end].numel()
    return count


def _zero_cols(layer, start, end):
    if layer is None:
        return 0
    with torch.no_grad():
        layer.weight[:, start:end] = 0
    return layer.weight[:, start:end].numel()


def _mask_attention_head(block, head_index):
    attn = _find_attention(block)
    if attn is None:
        return 0
    head_dim = getattr(attn, "head_dim", None)
    if not isinstance(head_dim, int):
        return 0
    start = int(head_index) * head_dim
    end = start + head_dim
    count = 0
    q_proj = _linear(attn, "q_proj") or _linear(attn, "c_attn")
    k_proj = _linear(attn, "k_proj")
    v_proj = _linear(attn, "v_proj")
    o_proj = _linear(attn, "o_proj") or _linear(attn, "c_proj")
    count += _zero_rows(q_proj, start, end)
    count += _zero_rows(k_proj, start, end)
    count += _zero_rows(v_proj, start, end)
    count += _zero_cols(o_proj, start, end)
    return count


def _mask_ffn_neuron(block, neuron_index):
    mlp = _find_mlp(block)
    if mlp is None:
        return 0
    start = int(neuron_index)
    end = start + 1
    count = 0
    for name in ["gate_proj", "up_proj", "fc1", "c_fc", "w1", "wi"]:
        count += _zero_rows(_linear(mlp, name), start, end)
    for name in ["down_proj", "fc2", "c_proj", "w2", "wo"]:
        count += _zero_cols(_linear(mlp, name), start, end)
    return count


def _physical_prune_ffn_neurons(block, neuron_indices):
    mlp = _find_mlp(block)
    if mlp is None or not neuron_indices:
        return 0, 0
    ffn_dim = None
    for name in ["gate_proj", "up_proj", "fc1", "c_fc", "w1", "wi"]:
        layer = _linear(mlp, name)
        if layer is not None:
            ffn_dim = int(layer.weight.shape[0])
            break
    if not ffn_dim:
        return 0, 0
    prune = sorted(set(int(index) for index in neuron_indices if 0 <= int(index) < ffn_dim))
    if len(prune) >= ffn_dim:
        prune = prune[:-1]
    keep = [index for index in range(ffn_dim) if index not in set(prune)]
    removed = 0
    for name in ["gate_proj", "up_proj", "fc1", "c_fc", "w1", "wi"]:
        layer = _linear(mlp, name)
        if layer is None:
            continue
        new_layer, removed_count = _replace_linear(layer, keep_rows=keep)
        _set_linear(mlp, name, new_layer)
        removed += removed_count
    for name in ["down_proj", "fc2", "c_proj", "w2", "wo"]:
        layer = _linear(mlp, name)
        if layer is None:
            continue
        new_layer, removed_count = _replace_linear(layer, keep_cols=keep)
        _set_linear(mlp, name, new_layer)
        removed += removed_count
    for attr in ["intermediate_size", "ffn_dim", "hidden_features"]:
        if hasattr(mlp, attr):
            try:
                setattr(mlp, attr, len(keep))
            except Exception:
                pass
    return int(removed), len(prune)


def apply_unit_mask_pruning(blocks, unit_plan):
    selected = [row for row in unit_plan.get("units", []) if row.get("selected")]
    masked_parameters = 0
    selected_by_type = {}
    for row in selected:
        block_index = int(row["block"])
        unit_type = row["unit_type"]
        unit_index = int(row["unit_index"])
        if block_index < 0 or block_index >= len(blocks):
            continue
        if unit_type == "attention_head":
            masked_parameters += _mask_attention_head(blocks[block_index], unit_index)
        elif unit_type == "ffn_neuron":
            masked_parameters += _mask_ffn_neuron(blocks[block_index], unit_index)
        selected_by_type[unit_type] = selected_by_type.get(unit_type, 0) + 1
    return {
        "pruning_mode": "unit_mask",
        "selected_units": len(selected),
        "selected_by_type": selected_by_type,
        "masked_parameter_entries": int(masked_parameters),
        "note": "Structured unit mask pruning zeros selected head/neuron slices. Unit cost follows the MCPrune resource-cost view; physical dimension removal is model-specific.",
    }


def apply_unit_physical_pruning(blocks, unit_plan):
    selected = [row for row in unit_plan.get("units", []) if row.get("selected")]
    by_block = {}
    for row in selected:
        by_block.setdefault(int(row["block"]), []).append(row)

    removed_parameter_entries = 0
    masked_parameter_entries = 0
    selected_by_type = {}
    physically_pruned_by_type = {}
    masked_fallback_by_type = {}
    for block_index, rows in by_block.items():
        if block_index < 0 or block_index >= len(blocks):
            continue
        ffn_indices = [
            int(row["unit_index"])
            for row in rows
            if row["unit_type"] == "ffn_neuron"
        ]
        removed, pruned_neurons = _physical_prune_ffn_neurons(
            blocks[block_index],
            ffn_indices,
        )
        removed_parameter_entries += removed
        if pruned_neurons:
            physically_pruned_by_type["ffn_neuron"] = (
                physically_pruned_by_type.get("ffn_neuron", 0) + pruned_neurons
            )

        for row in rows:
            unit_type = row["unit_type"]
            selected_by_type[unit_type] = selected_by_type.get(unit_type, 0) + 1
            if unit_type == "attention_head":
                masked_parameter_entries += _mask_attention_head(
                    blocks[block_index],
                    int(row["unit_index"]),
                )
                masked_fallback_by_type[unit_type] = (
                    masked_fallback_by_type.get(unit_type, 0) + 1
                )

    return {
        "pruning_mode": "unit_physical",
        "selected_units": len(selected),
        "selected_by_type": selected_by_type,
        "physically_pruned_by_type": physically_pruned_by_type,
        "masked_fallback_by_type": masked_fallback_by_type,
        "removed_parameter_entries": int(removed_parameter_entries),
        "masked_parameter_entries": int(masked_parameter_entries),
        "note": "FFN neurons are physically removed. Attention heads currently use mask fallback to avoid unsafe GQA/RoPE config changes.",
    }
