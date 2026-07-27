import csv
import math
import os

import torch
from torch.utils.data import DataLoader

from amcprune.hessian import SNOWSEngine, math_sdp_for_hvp


def _get_module(root, path):
    current = root
    for part in path.split("."):
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


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


def _linear(module, name):
    layer = getattr(module, name, None)
    weight = getattr(layer, "weight", None)
    return layer if weight is not None else None


def _infer_attention_shape(attn):
    num_heads = getattr(attn, "num_heads", None)
    if not isinstance(num_heads, int):
        num_heads = getattr(attn, "num_attention_heads", None)
    num_key_value_heads = getattr(attn, "num_key_value_heads", None)
    if not isinstance(num_key_value_heads, int):
        num_key_value_heads = num_heads
    head_dim = getattr(attn, "head_dim", None)
    if not isinstance(head_dim, int):
        hidden_size = getattr(attn, "hidden_size", None)
        if isinstance(hidden_size, int) and isinstance(num_heads, int) and num_heads > 0:
            head_dim = hidden_size // num_heads
    if not all(isinstance(value, int) and value > 0 for value in [num_heads, num_key_value_heads, head_dim]):
        return None
    return num_heads, num_key_value_heads, head_dim


def _infer_ffn_dim(mlp):
    for name in ["gate_proj", "up_proj", "fc1", "c_fc", "w1", "wi"]:
        layer = _linear(mlp, name)
        if layer is not None:
            return int(layer.weight.shape[0])
    return None


def _tensor_second_moment(value):
    if isinstance(value, tuple):
        value = value[0]
    if not torch.is_tensor(value):
        return None
    if value.dim() == 2:
        value = value.unsqueeze(0)
    if value.dim() != 3:
        return None
    flat = value.detach().float().reshape(-1, value.shape[-1])
    return flat.pow(2).mean(dim=0).cpu()


def _mean_slice(values, start, end):
    if values is None or values.numel() == 0:
        return 0.0
    start = max(0, min(start, values.numel()))
    end = max(start, min(end, values.numel()))
    if end <= start:
        return 0.0
    return float(values[start:end].mean().item())


def _score_parameter_slice(parameter, row_slice=None, col_slice=None):
    if parameter is None or parameter.grad is None:
        return 0.0, 0
    values = parameter.detach().float()
    grads = parameter.grad.detach().float()
    if row_slice is not None and values.dim() >= 1:
        values = values[row_slice]
        grads = grads[row_slice]
    if col_slice is not None and values.dim() >= 2:
        values = values[:, col_slice]
        grads = grads[:, col_slice]
    score = (values * grads).pow(2).sum()
    return float(score.cpu().item()), values.numel()


def _safe_float(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _attention_resource_cost(parameter_cost, head_dim, group_size, batch_size, seq_len):
    parameter_cost = float(parameter_cost)
    head_dim = float(max(head_dim, 1))
    group_size = float(max(group_size, 1))
    batch_size = float(max(batch_size, 1))
    seq_len = float(max(seq_len, 1))
    linear_flops = batch_size * seq_len * parameter_cost
    attention_flops = batch_size * 2.0 * (seq_len ** 2) * head_dim
    attention_compute_cost = linear_flops + attention_flops
    activation_cost = batch_size * seq_len * head_dim
    kv_cache_cost = batch_size * seq_len * (2.0 * head_dim / group_size)
    resource_cost = (
        parameter_cost
        + attention_compute_cost
        + activation_cost
        + kv_cache_cost
    )
    return {
        "parameter_cost": parameter_cost,
        "linear_flops": linear_flops,
        "attention_flops": attention_flops,
        "attention_compute_cost": attention_compute_cost,
        "activation_cost": activation_cost,
        "kv_cache_cost": kv_cache_cost,
        "resource_cost": resource_cost,
        "resource_cost_type": "attention_head_parameter_flops_activation_kvcache",
    }


def _ffn_resource_cost(parameter_cost, batch_size, seq_len):
    parameter_cost = float(parameter_cost)
    batch_size = float(max(batch_size, 1))
    seq_len = float(max(seq_len, 1))
    mlp_compute_cost = batch_size * seq_len * parameter_cost
    activation_cost = batch_size * seq_len
    resource_cost = parameter_cost + mlp_compute_cost + activation_cost
    return {
        "parameter_cost": parameter_cost,
        "linear_flops": mlp_compute_cost,
        "attention_flops": 0.0,
        "mlp_compute_cost": mlp_compute_cost,
        "activation_cost": activation_cost,
        "kv_cache_cost": 0.0,
        "resource_cost": resource_cost,
        "resource_cost_type": "ffn_neuron_parameter_flops_activation",
    }


def _hvp_parameter_slice(hv, row_slice=None, col_slice=None):
    if hv is None:
        return 0.0, 0
    values = torch.nan_to_num(
        hv.detach().float(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if row_slice is not None and values.dim() >= 1:
        values = values[row_slice]
    if col_slice is not None and values.dim() >= 2:
        values = values[:, col_slice]
    score = values.sum()
    return _safe_float(score.cpu().item()), values.numel()


def _add_linear_row_score(rows, layer, row_slice):
    score, count = _score_parameter_slice(layer.weight, row_slice=row_slice)
    bias_score, bias_count = _score_parameter_slice(getattr(layer, "bias", None), row_slice=row_slice)
    rows[0] += score + bias_score
    rows[1] += count + bias_count


def _add_linear_col_score(rows, layer, col_slice):
    score, count = _score_parameter_slice(layer.weight, col_slice=col_slice)
    rows[0] += score
    rows[1] += count


def _add_linear_row_hvp(rows, layer, hv_by_param_id, row_slice):
    if layer is None:
        return
    score, count = _hvp_parameter_slice(
        hv_by_param_id.get(id(layer.weight)),
        row_slice=row_slice,
    )
    rows[0] += score
    rows[1] += count


def _add_linear_col_hvp(rows, layer, hv_by_param_id, col_slice):
    if layer is None:
        return
    score, count = _hvp_parameter_slice(
        hv_by_param_id.get(id(layer.weight)),
        col_slice=col_slice,
    )
    rows[0] += score
    rows[1] += count


def _collect_unit_outliers(model, blocks, selected_blocks, dataset, device, batch_size, max_batches):
    selected = set(selected_blocks)
    block_second = {}
    ffn_second = {}
    hooks = []

    def make_block_hook(index):
        def hook(_, inputs, __):
            if inputs and index in selected:
                values = _tensor_second_moment(inputs[0])
                if values is not None:
                    current = block_second.get(index)
                    block_second[index] = values if current is None else current + values
        return hook

    def make_mlp_hook(index, mlp):
        def hook(_, inputs, __):
            if not inputs or index not in selected:
                return
            hidden = inputs[0]
            with torch.no_grad():
                gate = _linear(mlp, "gate_proj")
                up = _linear(mlp, "up_proj")
                if gate is not None and up is not None:
                    act_fn = getattr(mlp, "act_fn", None)
                    gate_out = gate(hidden)
                    if act_fn is not None:
                        gate_out = act_fn(gate_out)
                    values = gate_out * up(hidden)
                else:
                    fc1 = _linear(mlp, "fc1") or _linear(mlp, "c_fc")
                    if fc1 is None:
                        return
                    values = fc1(hidden)
                moment = _tensor_second_moment(values)
                if moment is not None:
                    current = ffn_second.get(index)
                    ffn_second[index] = moment if current is None else current + moment
        return hook

    for index, block in enumerate(blocks):
        if index not in selected:
            continue
        hooks.append(block.register_forward_hook(make_block_hook(index)))
        mlp = _find_mlp(block)
        if mlp is not None:
            hooks.append(mlp.register_forward_hook(make_mlp_hook(index, mlp)))

    counts = 0
    model_was_training = model.training
    model.eval()
    try:
        loader = DataLoader(dataset, batch_size=batch_size)
        for step, (input_ids, attention_mask) in enumerate(loader):
            if step >= max_batches:
                break
            model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
            )
            counts += 1
    finally:
        for hook in hooks:
            hook.remove()
        model.train(model_was_training)

    if counts:
        for key in list(block_second):
            block_second[key] = block_second[key] / float(counts)
        for key in list(ffn_second):
            ffn_second[key] = ffn_second[key] / float(counts)
    return block_second, ffn_second


def _selected_target_linears(blocks, selected_blocks):
    selected = set(selected_blocks)
    layers = []
    seen = set()
    for block_index, block in enumerate(blocks):
        if block_index not in selected:
            continue
        attn = _find_attention(block)
        mlp = _find_mlp(block)
        for module, names in [
            (attn, ["q_proj", "k_proj", "v_proj", "o_proj", "c_attn", "c_proj"]),
            (mlp, ["gate_proj", "up_proj", "down_proj", "fc1", "fc2", "c_fc", "c_proj", "w1", "w2", "wi", "wo"]),
        ]:
            if module is None:
                continue
            for name in names:
                layer = _linear(module, name)
                if layer is None:
                    continue
                param = layer.weight
                if id(param) in seen:
                    continue
                seen.add(id(param))
                layers.append((f"block{block_index}.{name}", layer))
    print(f"[Unit Scoring] selected target linears={len(layers)}", flush=True)
    return layers


def _compute_hvp_by_param_id(
    *,
    model,
    blocks,
    selected_blocks,
    dataset,
    device,
    batch_size,
    max_batches,
    k_horizon,
):
    target_layers = _selected_target_linears(blocks, selected_blocks)
    target_params = [layer.weight for _, layer in target_layers]
    if not target_params:
        return {}, 0, []

    hv_accumulators = {
        id(param): torch.zeros_like(param, dtype=torch.float32, device="cpu")
        for param in target_params
    }
    engine = SNOWSEngine()
    loader = DataLoader(dataset, batch_size=batch_size)
    used_batches = 0
    model_was_training = model.training
    model.eval()
    try:
        for step, (input_ids, attention_mask) in enumerate(loader):
            if step >= max_batches:
                break
            with math_sdp_for_hvp():
                outputs = model(
                    input_ids=input_ids.to(device),
                    attention_mask=attention_mask.to(device),
                    labels=input_ids.to(device),
                )
                hv_list = engine.get_k_step_hessian_selective(
                    outputs.loss,
                    target_params,
                    K_horizon=k_horizon,
                )
            for param, hv in zip(target_params, hv_list):
                safe_hv = torch.nan_to_num(
                    hv.detach().float(),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                hv_accumulators[id(param)] += safe_hv.cpu().pow(2)
            used_batches += 1
            model.zero_grad(set_to_none=True)
    finally:
        model.train(model_was_training)

    if used_batches:
        for key in list(hv_accumulators):
            hv_accumulators[key] = hv_accumulators[key] / float(used_batches)
    return hv_accumulators, used_batches, [name for name, _ in target_layers]


def score_candidate_units_by_hessian_proxy(
    *,
    model,
    blocks,
    block_path,
    selected_blocks,
    dataset,
    device,
    batch_size,
    max_batches,
    method="hessian_proxy",
    k_horizon=1,
    seq_len=None,
):
    """Score selected-block heads and FFN neurons with HVP or a proxy.

    method="hvp" uses the same selective HVP pattern as MCPrune. The proxy
    path is kept as a faster fallback for smoke tests.
    """
    selected = set(selected_blocks)
    resource_seq_len = int(seq_len or 1)
    print(
        "[Unit Scoring] enter "
        f"method={method} selected_blocks={sorted(selected)} "
        f"batch_size={batch_size} seq_len={resource_seq_len} max_batches={max_batches}",
        flush=True,
    )
    print("[Unit Scoring] collect unit outliers start", flush=True)
    block_second, ffn_second = _collect_unit_outliers(
        model, blocks, selected_blocks, dataset, device, batch_size, max_batches
    )
    print("[Unit Scoring] collect unit outliers done", flush=True)

    hv_by_param_id = {}
    hvp_target_names = []
    if method == "hvp":
        print("[Unit Scoring] HVP start", flush=True)
        hv_by_param_id, used_batches, hvp_target_names = _compute_hvp_by_param_id(
            model=model,
            blocks=blocks,
            selected_blocks=selected_blocks,
            dataset=dataset,
            device=device,
            batch_size=batch_size,
            max_batches=max_batches,
            k_horizon=k_horizon,
        )
        print(
            f"[Unit Scoring] HVP done batches={used_batches} "
            f"targets={len(hvp_target_names)}",
            flush=True,
        )
    else:
        print("[Unit Scoring] proxy backward start", flush=True)
        model.zero_grad(set_to_none=True)
        model_was_training = model.training
        model.eval()
        loader = DataLoader(dataset, batch_size=batch_size)
        used_batches = 0
        for step, (input_ids, attention_mask) in enumerate(loader):
            if step >= max_batches:
                break
            outputs = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                labels=input_ids.to(device),
            )
            (outputs.loss / float(max_batches)).backward()
            used_batches += 1
        print(f"[Unit Scoring] proxy backward done batches={used_batches}", flush=True)

    print("[Unit Scoring] build unit rows start", flush=True)
    rows = []
    for block_index in sorted(selected):
        before_block_rows = len(rows)
        print(f"[Unit Scoring] build rows block={block_index} start", flush=True)
        block = blocks[block_index]
        block_name = f"{block_path}.{block_index}"
        attn = _find_attention(block)
        mlp = _find_mlp(block)
        block_outlier = block_second.get(block_index)
        ffn_outlier = ffn_second.get(block_index)

        if attn is not None:
            attention_shape = _infer_attention_shape(attn)
            q_proj = _linear(attn, "q_proj") or _linear(attn, "c_attn")
            k_proj = _linear(attn, "k_proj")
            v_proj = _linear(attn, "v_proj")
            o_proj = _linear(attn, "o_proj") or _linear(attn, "c_proj")
            if attention_shape is not None:
                num_heads, num_key_value_heads, head_dim = attention_shape
                group_size = max(num_heads // max(num_key_value_heads, 1), 1)
                for head in range(num_heads):
                    start = head * head_dim
                    end = start + head_dim
                    kv_head = min(head // group_size, num_key_value_heads - 1)
                    kv_start = kv_head * head_dim
                    kv_end = kv_start + head_dim
                    acc = [0.0, 0]
                    if method == "hvp":
                        _add_linear_row_hvp(acc, q_proj, hv_by_param_id, slice(start, end))
                        _add_linear_row_hvp(acc, k_proj, hv_by_param_id, slice(kv_start, kv_end))
                        _add_linear_row_hvp(acc, v_proj, hv_by_param_id, slice(kv_start, kv_end))
                        _add_linear_col_hvp(acc, o_proj, hv_by_param_id, slice(start, end))
                    else:
                        if q_proj is not None:
                            _add_linear_row_score(acc, q_proj, slice(start, end))
                        if k_proj is not None:
                            _add_linear_row_score(acc, k_proj, slice(kv_start, kv_end))
                        if v_proj is not None:
                            _add_linear_row_score(acc, v_proj, slice(kv_start, kv_end))
                        if o_proj is not None:
                            _add_linear_col_score(acc, o_proj, slice(start, end))
                    raw_score, parameter_cost = acc
                    raw_score = _safe_float(raw_score)
                    sensitivity = raw_score / max(parameter_cost, 1)
                    cost_terms = _attention_resource_cost(
                        parameter_cost=parameter_cost,
                        head_dim=head_dim,
                        group_size=group_size,
                        batch_size=batch_size,
                        seq_len=resource_seq_len,
                    )
                    rows.append({
                        "block": block_index,
                        "block_name": block_name,
                        "unit_type": "attention_head",
                        "unit_index": head,
                        "unit_name": f"{block_name}.attention_head.{head}",
                        "hessian_score": sensitivity,
                        "hessian_proxy_score": sensitivity,
                        "unit_score_method": method,
                        "sensitivity_score": sensitivity,
                        "outlier_risk": _mean_slice(block_outlier, start, end),
                        "memory_cost": cost_terms["resource_cost"],
                        **cost_terms,
                        "num_attention_heads": num_heads,
                        "num_key_value_heads": num_key_value_heads,
                        "kv_group_size": group_size,
                        "kv_head_index": kv_head,
                        "score": sensitivity,
                    })

        if mlp is not None:
            ffn_dim = _infer_ffn_dim(mlp)
            gate_proj = _linear(mlp, "gate_proj")
            up_proj = _linear(mlp, "up_proj")
            down_proj = _linear(mlp, "down_proj")
            fc1 = _linear(mlp, "fc1") or _linear(mlp, "c_fc")
            fc2 = _linear(mlp, "fc2") or _linear(mlp, "c_proj")
            if ffn_dim:
                for neuron in range(ffn_dim):
                    acc = [0.0, 0]
                    row_slice = slice(neuron, neuron + 1)
                    col_slice = slice(neuron, neuron + 1)
                    if method == "hvp":
                        for layer in [gate_proj, up_proj, fc1]:
                            _add_linear_row_hvp(acc, layer, hv_by_param_id, row_slice)
                        for layer in [down_proj, fc2]:
                            _add_linear_col_hvp(acc, layer, hv_by_param_id, col_slice)
                    else:
                        for layer in [gate_proj, up_proj, fc1]:
                            if layer is not None:
                                _add_linear_row_score(acc, layer, row_slice)
                        for layer in [down_proj, fc2]:
                            if layer is not None:
                                _add_linear_col_score(acc, layer, col_slice)
                    raw_score, parameter_cost = acc
                    raw_score = _safe_float(raw_score)
                    sensitivity = raw_score / max(parameter_cost, 1)
                    cost_terms = _ffn_resource_cost(
                        parameter_cost=parameter_cost,
                        batch_size=batch_size,
                        seq_len=resource_seq_len,
                    )
                    rows.append({
                        "block": block_index,
                        "block_name": block_name,
                        "unit_type": "ffn_neuron",
                        "unit_index": neuron,
                        "unit_name": f"{block_name}.ffn_neuron.{neuron}",
                        "hessian_score": sensitivity,
                        "hessian_proxy_score": sensitivity,
                        "unit_score_method": method,
                        "sensitivity_score": sensitivity,
                        "outlier_risk": _mean_slice(ffn_outlier, neuron, neuron + 1),
                        "memory_cost": cost_terms["resource_cost"],
                        **cost_terms,
                        "score": sensitivity,
                    })

        print(
            f"[Unit Scoring] build rows block={block_index} "
            f"done added={len(rows) - before_block_rows} total={len(rows)}",
            flush=True,
        )

    model.zero_grad(set_to_none=True)
    if method != "hvp":
        model.train(model_was_training)
    for row in rows:
        row["score_batches"] = used_batches
        row["hvp_target_layers"] = ";".join(hvp_target_names)
    print(f"[Unit Scoring] build unit rows done rows={len(rows)}", flush=True)
    return rows


def save_unit_scores_csv(path, rows):
    if not rows:
        return None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path
