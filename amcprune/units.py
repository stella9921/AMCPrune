import csv
import os


def _get_attr_path(root, path):
    current = root
    for part in path.split("."):
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def _parameter_count(module):
    if module is None:
        return 0
    return sum(parameter.numel() for parameter in module.parameters())


def _infer_num_heads(block):
    candidates = [
        "self_attn.num_heads",
        "attn.num_heads",
        "attention.num_heads",
        "self_attention.num_heads",
    ]
    for path in candidates:
        value = _get_attr_path(block, path)
        if isinstance(value, int):
            return value
    for module in block.modules():
        value = getattr(module, "num_heads", None)
        if isinstance(value, int):
            return value
    return None


def _infer_head_dim(block):
    candidates = [
        "self_attn.head_dim",
        "attn.head_dim",
        "attention.head_dim",
        "self_attention.head_dim",
    ]
    for path in candidates:
        value = _get_attr_path(block, path)
        if isinstance(value, int):
            return value
    return None


def _find_attention_module(block):
    candidates = ["self_attn", "attn", "attention", "self_attention"]
    for path in candidates:
        module = _get_attr_path(block, path)
        if module is not None:
            return path, module
    return None, None


def _find_mlp_module(block):
    candidates = ["mlp", "feed_forward", "ffn", "MLP"]
    for path in candidates:
        module = _get_attr_path(block, path)
        if module is not None:
            return path, module
    return None, None


def _infer_ffn_dim(mlp_module):
    if mlp_module is None:
        return None
    candidates = ["gate_proj", "up_proj", "fc1", "c_fc", "w1", "wi"]
    for name in candidates:
        layer = getattr(mlp_module, name, None)
        weight = getattr(layer, "weight", None)
        if weight is not None and len(weight.shape) >= 2:
            return int(weight.shape[0])
    for module in mlp_module.modules():
        weight = getattr(module, "weight", None)
        if weight is not None and len(weight.shape) >= 2:
            return int(weight.shape[0])
    return None


def inspect_block_units(blocks, block_path, selected_blocks=None):
    selected = set(selected_blocks or [])
    rows = []
    for index, block in enumerate(blocks):
        attention_path, attention_module = _find_attention_module(block)
        mlp_path, mlp_module = _find_mlp_module(block)
        num_heads = _infer_num_heads(block)
        head_dim = _infer_head_dim(block)
        ffn_dim = _infer_ffn_dim(mlp_module)
        rows.append({
            "block": index,
            "block_name": f"{block_path}.{index}",
            "selected_block": index in selected,
            "attention_path": f"{block_path}.{index}.{attention_path}" if attention_path else None,
            "mlp_path": f"{block_path}.{index}.{mlp_path}" if mlp_path else None,
            "num_attention_heads": num_heads,
            "head_dim": head_dim,
            "ffn_intermediate_dim": ffn_dim,
            "attention_params": _parameter_count(attention_module),
            "mlp_params": _parameter_count(mlp_module),
            "block_params": _parameter_count(block),
        })
    return rows


def save_unit_inventory_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "block",
        "block_name",
        "selected_block",
        "attention_path",
        "mlp_path",
        "num_attention_heads",
        "head_dim",
        "ffn_intermediate_dim",
        "attention_params",
        "mlp_params",
        "block_params",
    ]
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path
