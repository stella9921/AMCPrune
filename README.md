# AMCPrune

AMCPrune is a prototype repository for memory-aware structured pruning of
pretrained small language models.

Initial goal:

1. Load a pretrained causal language model.
2. Load a small text calibration/evaluation dataset.
3. Measure baseline perplexity.
4. Apply a simple block-level pruning mask.
5. Measure pruned perplexity and memory metrics.

The first prototype intentionally keeps the scoring simple so that the LLM
pruning pipeline can run end to end before adding Fisher/Hessian refinement.

## Smoke Test

```bash
python main.py \
  --model sshleifer/tiny-gpt2 \
  --dataset wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --split test \
  --max-samples 32 \
  --seq-len 128 \
  --pruning-ratio 0.25 \
  --score block_index \
  --output-dir exp/smoke_tiny_gpt2
```

