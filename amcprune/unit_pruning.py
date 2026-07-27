import torch
import torch.nn as nn
from contextlib import contextmanager


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


def _infer_attention_shape(attn):
    num_heads = getattr(attn, "num_heads", None)
    if not isinstance(num_heads, int):
        num_heads = getattr(attn, "num_attention_heads", None)
    num_key_value_heads = getattr(attn, "num_key_value_heads", None)
    if not isinstance(num_key_value_heads, int):
        config = getattr(attn, "config", None)
        num_key_value_heads = getattr(config, "num_key_value_heads", None)
    if not isinstance(num_key_value_heads, int):
        num_key_value_heads = num_heads
    head_dim = getattr(attn, "head_dim", None)
    if not isinstance(head_dim, int):
        hidden_size = getattr(attn, "hidden_size", None)
        if not isinstance(hidden_size, int):
            config = getattr(attn, "config", None)
            hidden_size = getattr(config, "hidden_size", None)
        if isinstance(hidden_size, int) and isinstance(num_heads, int) and num_heads > 0:
            head_dim = hidden_size // num_heads
    q_proj = _linear(attn, "q_proj") or _linear(attn, "c_attn")
    k_proj = _linear(attn, "k_proj")
    if not isinstance(num_heads, int):
        config = getattr(attn, "config", None)
        num_heads = getattr(config, "num_attention_heads", None)
    if not isinstance(head_dim, int) and q_proj is not None and isinstance(num_heads, int) and num_heads > 0:
        head_dim = int(q_proj.weight.shape[0]) // num_heads
    if (
        not isinstance(num_key_value_heads, int)
        and k_proj is not None
        and isinstance(head_dim, int)
        and head_dim > 0
    ):
        num_key_value_heads = int(k_proj.weight.shape[0]) // head_dim
    if not all(isinstance(value, int) and value > 0 for value in [num_heads, num_key_value_heads, head_dim]):
        return None
    return num_heads, num_key_value_heads, head_dim


def _row_head_indices(row):
    values = row.get("head_indices")
    if isinstance(values, (list, tuple)):
        return [int(value) for value in values]
    return [int(row["unit_index"])]


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


def _save_and_zero_rows(layer, start, end, backups):
    if layer is None:
        return
    backups.append((layer.weight, (slice(start, end), slice(None)), layer.weight[start:end, :].detach().clone()))
    with torch.no_grad():
        layer.weight[start:end, :] = 0
        if getattr(layer, "bias", None) is not None:
            backups.append((layer.bias, (slice(start, end),), layer.bias[start:end].detach().clone()))
            layer.bias[start:end] = 0


def _save_and_zero_cols(layer, start, end, backups):
    if layer is None:
        return
    backups.append((layer.weight, (slice(None), slice(start, end)), layer.weight[:, start:end].detach().clone()))
    with torch.no_grad():
        layer.weight[:, start:end] = 0


def _mask_attention_head(block, head_index):
    attn = _find_attention(block)
    if attn is None:
        return 0
    shape = _infer_attention_shape(attn)
    if shape is None:
        return 0
    _, _, head_dim = shape
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


def _temporary_mask_attention_head(block, head_index, backups):
    attn = _find_attention(block)
    if attn is None:
        return
    shape = _infer_attention_shape(attn)
    if shape is None:
        return
    _, _, head_dim = shape
    start = int(head_index) * head_dim
    end = start + head_dim
    q_proj = _linear(attn, "q_proj") or _linear(attn, "c_attn")
    k_proj = _linear(attn, "k_proj")
    v_proj = _linear(attn, "v_proj")
    o_proj = _linear(attn, "o_proj") or _linear(attn, "c_proj")
    _save_and_zero_rows(q_proj, start, end, backups)
    _save_and_zero_rows(k_proj, start, end, backups)
    _save_and_zero_rows(v_proj, start, end, backups)
    _save_and_zero_cols(o_proj, start, end, backups)


def _temporary_mask_ffn_neuron(block, neuron_index, backups):
    mlp = _find_mlp(block)
    if mlp is None:
        return
    start = int(neuron_index)
    end = start + 1
    for name in ["gate_proj", "up_proj", "fc1", "c_fc", "w1", "wi"]:
        _save_and_zero_rows(_linear(mlp, name), start, end, backups)
    for name in ["down_proj", "fc2", "c_proj", "w2", "wo"]:
        _save_and_zero_cols(_linear(mlp, name), start, end, backups)


@contextmanager
def temporary_unit_mask_pruning(blocks, unit_plan):
    backups = []
    selected = [row for row in unit_plan.get("units", []) if row.get("selected")]
    try:
        for row in selected:
            block_index = int(row["block"])
            if block_index < 0 or block_index >= len(blocks):
                continue
            if row["unit_type"] == "attention_head":
                for head_index in _row_head_indices(row):
                    _temporary_mask_attention_head(blocks[block_index], head_index, backups)
            elif row["unit_type"] == "ffn_neuron":
                _temporary_mask_ffn_neuron(blocks[block_index], int(row["unit_index"]), backups)
        yield
    finally:
        with torch.no_grad():
            for tensor, index, value in reversed(backups):
                tensor[index] = value.to(device=tensor.device, dtype=tensor.dtype)


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


def _physical_prune_attention_heads(block, head_indices):
    attn = _find_attention(block)
    if attn is None or not head_indices:
        return 0, 0, "no_attention"

    shape = _infer_attention_shape(attn)
    if shape is None:
        return 0, 0, "unknown_attention_shape"
    num_heads, num_key_value_heads, head_dim = shape

    q_proj = _linear(attn, "q_proj")
    k_proj = _linear(attn, "k_proj")
    v_proj = _linear(attn, "v_proj")
    o_proj = _linear(attn, "o_proj") or _linear(attn, "c_proj")
    if q_proj is None or o_proj is None:
        return 0, 0, "fused_attention_not_supported"

    prune = sorted(set(int(index) for index in head_indices if 0 <= int(index) < num_heads))
    if not prune:
        return 0, 0, "empty_selection"
    if len(prune) >= num_heads:
        prune = prune[:-1]
    keep_heads = [index for index in range(num_heads) if index not in set(prune)]

    if num_key_value_heads == num_heads:
        keep_q_dims = [dim for head in keep_heads for dim in range(head * head_dim, (head + 1) * head_dim)]
        removed = 0
        new_q, count = _replace_linear(q_proj, keep_rows=keep_q_dims)
        _set_linear(attn, "q_proj", new_q)
        removed += count
        if k_proj is not None:
            new_k, count = _replace_linear(k_proj, keep_rows=keep_q_dims)
            _set_linear(attn, "k_proj", new_k)
            removed += count
        if v_proj is not None:
            new_v, count = _replace_linear(v_proj, keep_rows=keep_q_dims)
            _set_linear(attn, "v_proj", new_v)
            removed += count
        new_o, count = _replace_linear(o_proj, keep_cols=keep_q_dims)
        if hasattr(attn, "o_proj"):
            _set_linear(attn, "o_proj", new_o)
        else:
            _set_linear(attn, "c_proj", new_o)
        removed += count
        try:
            attn.num_heads = len(keep_heads)
            attn.num_attention_heads = len(keep_heads)
            attn.num_key_value_heads = len(keep_heads)
            attn.num_key_value_groups = 1
        except Exception:
            pass
        return int(removed), len(prune), "physical_mha"

    if num_heads % num_key_value_heads != 0:
        return 0, 0, "gqa_non_divisible_heads"
    group_size = num_heads // num_key_value_heads
    prune_set = set(prune)
    prune_groups = []
    for group in range(num_key_value_heads):
        group_heads = set(range(group * group_size, (group + 1) * group_size))
        if group_heads and group_heads.issubset(prune_set):
            prune_groups.append(group)
    if not prune_groups or len(prune_groups) >= num_key_value_heads:
        return 0, 0, "gqa_requires_full_kv_group_selection"

    keep_groups = [group for group in range(num_key_value_heads) if group not in set(prune_groups)]
    keep_heads = [
        head
        for group in keep_groups
        for head in range(group * group_size, (group + 1) * group_size)
    ]
    keep_q_dims = [dim for head in keep_heads for dim in range(head * head_dim, (head + 1) * head_dim)]
    keep_kv_dims = [dim for group in keep_groups for dim in range(group * head_dim, (group + 1) * head_dim)]

    removed = 0
    new_q, count = _replace_linear(q_proj, keep_rows=keep_q_dims)
    _set_linear(attn, "q_proj", new_q)
    removed += count
    if k_proj is not None:
        new_k, count = _replace_linear(k_proj, keep_rows=keep_kv_dims)
        _set_linear(attn, "k_proj", new_k)
        removed += count
    if v_proj is not None:
        new_v, count = _replace_linear(v_proj, keep_rows=keep_kv_dims)
        _set_linear(attn, "v_proj", new_v)
        removed += count
    new_o, count = _replace_linear(o_proj, keep_cols=keep_q_dims)
    if hasattr(attn, "o_proj"):
        _set_linear(attn, "o_proj", new_o)
    else:
        _set_linear(attn, "c_proj", new_o)
    removed += count

    new_num_heads = len(keep_heads)
    new_num_kv_heads = len(keep_groups)
    try:
        attn.num_heads = new_num_heads
        attn.num_attention_heads = new_num_heads
        attn.num_key_value_heads = new_num_kv_heads
        attn.num_key_value_groups = new_num_heads // new_num_kv_heads
    except Exception:
        pass
    return int(removed), num_heads - new_num_heads, "physical_gqa_group"


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
            for head_index in _row_head_indices(row):
                masked_parameters += _mask_attention_head(blocks[block_index], head_index)
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

        attention_indices = [
            head_index
            for row in rows
            if row["unit_type"] == "attention_head"
            for head_index in _row_head_indices(row)
        ]
        if attention_indices:
            removed, pruned_heads, reason = _physical_prune_attention_heads(
                blocks[block_index],
                attention_indices,
            )
            removed_parameter_entries += removed
            if pruned_heads:
                physically_pruned_by_type["attention_head"] = (
                    physically_pruned_by_type.get("attention_head", 0) + pruned_heads
                )
            else:
                for index in attention_indices:
                    masked_parameter_entries += _mask_attention_head(blocks[block_index], index)
                masked_fallback_by_type["attention_head"] = (
                    masked_fallback_by_type.get("attention_head", 0) + len(attention_indices)
                )
                masked_fallback_by_type["attention_head_reason"] = reason

        for row in rows:
            unit_type = row["unit_type"]
            selected_by_type[unit_type] = selected_by_type.get(unit_type, 0) + 1

    return {
        "pruning_mode": "unit_physical",
        "selected_units": len(selected),
        "selected_by_type": selected_by_type,
        "physically_pruned_by_type": physically_pruned_by_type,
        "masked_fallback_by_type": masked_fallback_by_type,
        "removed_parameter_entries": int(removed_parameter_entries),
        "masked_parameter_entries": int(masked_parameter_entries),
        "note": "FFN neurons are physically removed. Attention heads are physically removed for shape-safe MHA/GQA selections; unsafe GQA partial groups use mask fallback.",
    }
