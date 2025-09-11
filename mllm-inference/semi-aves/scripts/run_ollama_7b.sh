#!/usr/bin/env bash
set -euo pipefail

# Wrapper for the local Ollama backend. It defines the Ollama model and host,
# then calls the generic run_qwen.sh launcher.
# Don't forget: chmod +x run_ollama_7b.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# >>> EDIT THESE LINES TO CONFIGURE YOUR EXPERIMENT <<<
# 1. Choose the prompt template.
#    e.g., top5-sci-with-confidence, zeroshot-explanation, top5-simple-with-confidence, top5-flat-with-confidence, zeroshot-all200-explanation.
#    e.g., top5-sci, zeroshot, top5-simple, top5-flat, zeroshot-all200.
PROMPT_TEMPLATE="top5-sci-with-confidence"

# 2. Configure the Ollama-specific model and host.
OLLAMA_MODEL="qwen2.5vl:7b"
OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}" # Uses env var or defaults to localhost

# 3. Define the base model for file naming and TOPK JSON path.
#    Choose from vitb32_openclip_laion400m, resnet50_imagenet_pretrained, resnet50_inat_pretrained, dinov2_vitb14_reg, dinov3_vitb16
BASE_MODEL="vitb32_openclip_laion400m"

# --- Define paths based on configuration ---
OUTPUT_CSV="${SCRIPT_DIR}/../mllm_output/ollama_${BASE_MODEL}_${OLLAMA_MODEL//:/_}_${PROMPT_TEMPLATE}.csv"
ERROR_LOG="${SCRIPT_DIR}/../mllm_output/ollama_${BASE_MODEL}_${OLLAMA_MODEL//:/_}_${PROMPT_TEMPLATE}_errors.txt"
TOPK_JSON="${SCRIPT_DIR}/../../../data/semi-aves/topk/fewshot_finetune_${BASE_MODEL}_semi-aves_16_1_topk_test_predictions.json"

# --- Sanity check: ensure Ollama server is reachable ---
if command -v curl >/dev/null 2>&1; then
  if ! curl -sSf -m 3 "${OLLAMA_HOST}/api/version" >/dev/null; then
    echo "ERROR: Can't reach Ollama at ${OLLAMA_HOST}. Is 'ollama serve' running?" >&2
    exit 1
  fi
  echo ">> Successfully connected to Ollama at ${OLLAMA_HOST}"
fi

# --- Call the generic runner script ---
# The 'exec' command replaces the current script process with the new one.
exec "${SCRIPT_DIR}/run_qwen.sh" \
  --backend ollama \
  --python "${SCRIPT_DIR}/../run_inference.py" \
  --prompt-template "$PROMPT_TEMPLATE" \
  --prompt-dir "${SCRIPT_DIR}/../prompt_templates" \
  --image-dir "${SCRIPT_DIR}/../../../datasets/semi-aves" \
  --image-paths "${SCRIPT_DIR}/../../../data/semi-aves/test.txt" \
  --topk-json "$TOPK_JSON" \
  --taxonomy-json "${SCRIPT_DIR}/../../../data/semi-aves/semi-aves_metrics-LAION400M-taxonomy-enriched.json" \
  --ollama-model "$OLLAMA_MODEL" \
  --ollama-host "$OLLAMA_HOST" \
  --output-csv "$OUTPUT_CSV" \
  --error-file "$ERROR_LOG"
  # Add --dry-run here to test your configuration