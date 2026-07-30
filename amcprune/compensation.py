import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _min_max_normalize(values, eps=1.0e-12):
    if values.numel() == 0:
        return values
    min_value = values.min()
    max_value = values.max()
    denom = max_value - min_value
    if float(denom.item()) <= eps:
        return torch.zeros_like(values)
    return (values - min_value) / (denom + eps)


class BoundaryAffineWrapper(nn.Module):
    def __init__(self, block, alpha, beta):
        super().__init__()
        self.block = block
        self.register_buffer("alpha", alpha.detach().clone().view(1, 1, -1))
        self.register_buffer("beta", beta.detach().clone().view(1, 1, -1))

    def forward(self, hidden_states, *args, **kwargs):
        hidden_states = hidden_states * self.alpha.to(dtype=hidden_states.dtype) + self.beta.to(
            dtype=hidden_states.dtype
        )
        return self.block(hidden_states, *args, **kwargs)


def _hidden_from_block_output(output):
    if isinstance(output, tuple):
        return output[0]
    return output


def _resolve_block_container(model, block_path):
    parent = model
    parts = block_path.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return getattr(parent, parts[-1])


def boundary_indices_for_depth_pruning(depth_pruned_blocks, num_blocks):
    if not depth_pruned_blocks:
        return None
    low = min(int(index) for index in depth_pruned_blocks)
    high = max(int(index) for index in depth_pruned_blocks)
    source_index = low - 1
    target_index = high + 1
    if source_index < 0 or target_index >= num_blocks:
        return None
    return source_index, target_index


@torch.no_grad()
def estimate_boundary_affine_compensation(
    *,
    model,
    blocks,
    dataset,
    device,
    batch_size,
    max_batches,
    depth_pruned_blocks,
    channel_ratio=1.0,
    outlier_weight=0.25,
    memory_weight=0.25,
    eps=1.0e-6,
):
    boundary = boundary_indices_for_depth_pruning(depth_pruned_blocks, len(blocks))
    if boundary is None:
        return {
            "enabled": False,
            "reason": "depth-pruned interval has no valid left/right boundary",
        }
    source_index, target_index = boundary

    source_sum = None
    source_sumsq = None
    target_sum = None
    target_sumsq = None
    token_count = 0
    captured = {}

    def source_hook(_module, _inputs, output):
        captured["source"] = _hidden_from_block_output(output).detach().float()

    def target_pre_hook(_module, inputs):
        if inputs:
            captured["target"] = inputs[0].detach().float()

    source_handle = blocks[source_index].register_forward_hook(source_hook)
    target_handle = blocks[target_index].register_forward_pre_hook(target_pre_hook)

    loader = DataLoader(dataset, batch_size=batch_size)
    try:
        for step, (input_ids, attention_mask) in enumerate(loader):
            if step >= max_batches:
                break
            captured.clear()
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            model(input_ids=input_ids, attention_mask=attention_mask)
            if "source" not in captured or "target" not in captured:
                continue

            source = captured["source"].reshape(-1, captured["source"].shape[-1])
            target = captured["target"].reshape(-1, captured["target"].shape[-1])
            if source.shape != target.shape:
                continue
            if source_sum is None:
                source_sum = torch.zeros(source.shape[-1], device=device, dtype=torch.float32)
                source_sumsq = torch.zeros_like(source_sum)
                target_sum = torch.zeros_like(source_sum)
                target_sumsq = torch.zeros_like(source_sum)
            source_sum += source.sum(dim=0)
            source_sumsq += (source * source).sum(dim=0)
            target_sum += target.sum(dim=0)
            target_sumsq += (target * target).sum(dim=0)
            token_count += int(source.shape[0])
    finally:
        source_handle.remove()
        target_handle.remove()

    if token_count <= 0 or source_sum is None:
        return {
            "enabled": False,
            "source_original_index": source_index,
            "target_original_index": target_index,
            "reason": "no calibration activations were captured",
        }

    source_mean = source_sum / token_count
    target_mean = target_sum / token_count
    source_var = torch.clamp(source_sumsq / token_count - source_mean * source_mean, min=0.0)
    target_var = torch.clamp(target_sumsq / token_count - target_mean * target_mean, min=0.0)
    source_std = torch.sqrt(source_var + eps)
    target_std = torch.sqrt(target_var + eps)
    alpha = target_std / source_std
    beta = target_mean - alpha * source_mean
    mismatch = torch.abs(target_mean - source_mean) + torch.abs(target_std - source_std)
    outlier_risk = torch.maximum(source_sumsq / token_count, target_sumsq / token_count)
    # Hidden-channel affine compensation has the same inference overhead per channel:
    # one scale and one bias. The cost term therefore controls the selected budget
    # through channel_ratio, while keeping the Lagrangian objective explicit.
    resource_cost = torch.ones_like(mismatch) * 2.0
    mismatch_norm = _min_max_normalize(mismatch)
    outlier_norm = _min_max_normalize(outlier_risk)
    resource_norm = _min_max_normalize(resource_cost)
    objective = mismatch_norm + float(outlier_weight) * outlier_norm - float(memory_weight) * resource_norm
    ratio = min(max(float(channel_ratio), 0.0), 1.0)
    if ratio >= 1.0:
        channel_mask = torch.ones_like(mismatch, dtype=torch.bool)
    elif ratio <= 0.0:
        channel_mask = torch.zeros_like(mismatch, dtype=torch.bool)
    else:
        selected_count = max(1, int(round(objective.numel() * ratio)))
        selected_indices = torch.topk(objective, k=selected_count, largest=True).indices
        channel_mask = torch.zeros_like(mismatch, dtype=torch.bool)
        channel_mask[selected_indices] = True

    identity_alpha = torch.ones_like(alpha)
    identity_beta = torch.zeros_like(beta)
    alpha = torch.where(channel_mask, alpha, identity_alpha)
    beta = torch.where(channel_mask, beta, identity_beta)

    return {
        "enabled": True,
        "mode": "boundary_affine_channelwise",
        "source_original_index": source_index,
        "target_original_index": target_index,
        "calibration_tokens": token_count,
        "selection_objective": "lagrangian_boundary_mismatch_outlier_resource",
        "channel_ratio": ratio,
        "selected_channels": int(channel_mask.sum().item()),
        "total_channels": int(channel_mask.numel()),
        "outlier_weight": float(outlier_weight),
        "memory_weight": float(memory_weight),
        "alpha": alpha.detach(),
        "beta": beta.detach(),
        "mismatch_mean": float(mismatch.mean().item()),
        "mismatch_std": float(mismatch.std(unbiased=False).item()),
        "selected_mismatch_mean": float(mismatch[channel_mask].mean().item()) if bool(channel_mask.any()) else 0.0,
        "outlier_risk_mean": float(outlier_risk.mean().item()),
        "outlier_risk_std": float(outlier_risk.std(unbiased=False).item()),
        "selected_outlier_risk_mean": float(outlier_risk[channel_mask].mean().item()) if bool(channel_mask.any()) else 0.0,
        "resource_cost_per_channel": 2.0,
        "resource_cost_total": float(resource_cost[channel_mask].sum().item()),
        "objective_mean": float(objective.mean().item()),
        "objective_std": float(objective.std(unbiased=False).item()),
        "selected_objective_mean": float(objective[channel_mask].mean().item()) if bool(channel_mask.any()) else 0.0,
        "alpha_mean": float(alpha.mean().item()),
        "alpha_std": float(alpha.std(unbiased=False).item()),
        "beta_mean": float(beta.mean().item()),
        "beta_std": float(beta.std(unbiased=False).item()),
    }


def apply_boundary_affine_compensation(model, block_path, kept_original_indices, compensation):
    if not compensation or not compensation.get("enabled"):
        return {
            "enabled": False,
            "reason": compensation.get("reason") if compensation else "no compensation",
        }
    target_original_index = int(compensation["target_original_index"])
    if target_original_index not in kept_original_indices:
        return {
            "enabled": False,
            "reason": f"target boundary block {target_original_index} was not kept",
        }
    target_new_index = list(kept_original_indices).index(target_original_index)
    blocks = _resolve_block_container(model, block_path)
    target_device = next(blocks[target_new_index].parameters()).device
    alpha = compensation["alpha"].to(device=target_device)
    beta = compensation["beta"].to(device=alpha.device)
    blocks[target_new_index] = BoundaryAffineWrapper(blocks[target_new_index], alpha, beta)
    return {
        "enabled": True,
        "mode": compensation.get("mode", "boundary_affine_channelwise"),
        "source_original_index": int(compensation["source_original_index"]),
        "target_original_index": target_original_index,
        "target_new_index": target_new_index,
        "calibration_tokens": int(compensation.get("calibration_tokens", 0)),
        "selection_objective": compensation.get("selection_objective", "boundary_mismatch_topk"),
        "channel_ratio": float(compensation.get("channel_ratio", 1.0)),
        "selected_channels": int(compensation.get("selected_channels", 0)),
        "total_channels": int(compensation.get("total_channels", 0)),
        "outlier_weight": float(compensation.get("outlier_weight", 0.0)),
        "memory_weight": float(compensation.get("memory_weight", 0.0)),
        "resource_cost_total": float(compensation.get("resource_cost_total", 0.0)),
        "alpha_mean": float(compensation.get("alpha_mean", 0.0)),
        "alpha_std": float(compensation.get("alpha_std", 0.0)),
        "beta_mean": float(compensation.get("beta_mean", 0.0)),
        "beta_std": float(compensation.get("beta_std", 0.0)),
    }


def apply_boundary_affine_compensation_from_metadata(
    model,
    block_path,
    kept_original_indices,
    compensation,
):
    if not compensation or not compensation.get("enabled"):
        return {
            "enabled": False,
            "reason": compensation.get("reason") if compensation else "no compensation",
        }
    target_original_index = int(compensation["target_original_index"])
    if target_original_index not in kept_original_indices:
        return {
            "enabled": False,
            "reason": f"target boundary block {target_original_index} was not kept",
        }
    target_new_index = list(kept_original_indices).index(target_original_index)
    blocks = _resolve_block_container(model, block_path)
    target_device = next(blocks[target_new_index].parameters()).device
    hidden_size = getattr(getattr(model, "config", None), "hidden_size", None)
    if not isinstance(hidden_size, int):
        hidden_size = int(next(blocks[target_new_index].parameters()).shape[-1])
    alpha = torch.ones(hidden_size, device=target_device, dtype=torch.float32)
    beta = torch.zeros(hidden_size, device=target_device, dtype=torch.float32)
    blocks[target_new_index] = BoundaryAffineWrapper(blocks[target_new_index], alpha, beta)
    applied = dict(compensation)
    applied.update({"target_new_index": target_new_index})
    return applied
