import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_causal_lm(model_name, device=None, dtype="auto"):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = "auto"
    if dtype == "fp16":
        torch_dtype = torch.float16
    elif dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype == "fp32":
        torch_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
    )
    model.to(device)
    model.eval()
    return model, tokenizer, torch.device(device)


def get_transformer_blocks(model):
    candidates = [
        "transformer.h",
        "model.layers",
        "gpt_neox.layers",
        "decoder.layers",
    ]
    for path in candidates:
        current = model
        found = True
        for part in path.split("."):
            if not hasattr(current, part):
                found = False
                break
            current = getattr(current, part)
        if found:
            return list(current), path
    raise ValueError("Could not locate transformer blocks for this model.")

