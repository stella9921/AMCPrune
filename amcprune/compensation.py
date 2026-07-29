import torch
import torch.nn as nn
from torch.utils.data import DataLoader


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

    return {
        "enabled": True,
        "mode": "boundary_affine",
        "source_original_index": source_index,
        "target_original_index": target_index,
        "calibration_tokens": token_count,
        "alpha": alpha.detach(),
        "beta": beta.detach(),
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
        "mode": "boundary_affine",
        "source_original_index": int(compensation["source_original_index"]),
        "target_original_index": target_original_index,
        "target_new_index": target_new_index,
        "calibration_tokens": int(compensation.get("calibration_tokens", 0)),
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
