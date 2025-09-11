#!/usr/bin/env bash
set -euo pipefail

# Wrapper for the Hyperbolic API. It sets the correct API key, base URL, and model,
# then calls the generic run_qwen.sh launcher.
# Don't forget to run first: chmod +x run_hyperbolic_7b.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# >>> EDIT THESE LINES TO CONFIGURE YOUR EXPERIMENT <<<
# 1. Choose the prompt template.
#    e.g., top5-sci-with-confidence, zeroshot-explanation, top5-simple-with-confidence, top5-flat-with-confidence, zeroshot-all200-explanation.
#    e.g., top5-sci, zeroshot, top5-simple, top5-flat, zeroshot-all200.
PROMPT_TEMPLATE="top5-sci-with-confidence"

# 2. Configure the Hyperbolic-specific model and API base.
API_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
API_BASE="https://api.hyperbolic.xyz/v1"

# 3. Define the base model for file naming and TOPK JSON path.
#    Choose from vitb32_openclip_laion400m, resnet50_imagenet_pretrained, resnet50_inat_pretrained, dinov2_vitb14_reg, dinov3_vitb16
BASE_MODEL="vitb32_openclip_laion400m"

# --- Define paths based on configuration ---
OUTPUT_CSV="${SCRIPT_DIR}/../mllm_output/hyperbolic_${BASE_MODEL}_${API_MODEL//\//_}_${PROMPT_TEMPLATE}.csv"
ERROR_LOG="${SCRIPT_DIR}/../mllm_output/hyperbolic_${BASE_MODEL}_${API_MODEL//\//_}_${PROMPT_TEMPLATE}_errors.txt"
TOPK_JSON="${SCRIPT_DIR}/../../../data/semi-aves/topk/fewshot_finetune_${BASE_MODEL}_semi-aves_16_1_topk_test_predictions.json"
ENV_FILE="${SCRIPT_DIR}/../../../.env"

# --- Load environment and set API Key ---
# This wrapper requires HYPERBOLIC_API_KEY to be in your .env file.
# It exports it as OPENAI_API_KEY, which the downstream script expects.
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  echo ">> Loaded env from: $ENV_FILE"
fi
: "${HYPERBOLIC_API_KEY:?HYPERBOLIC_API_KEY not set in .env or exported}"
export OPENAI_API_KEY="$HYPERBOLIC_API_KEY"

# --- Call the generic runner script ---
"${SCRIPT_DIR}/run_qwen.sh" \
  --backend hyperbolic \
  --python "${SCRIPT_DIR}/../run_inference.py" \
  --prompt-template "$PROMPT_TEMPLATE" \
  --prompt-dir "${SCRIPT_DIR}/../prompt_templates" \
  --image-dir "${SCRIPT_DIR}/../../../datasets/semi-aves" \
  --image-paths "${SCRIPT_DIR}/../../../data/semi-aves/test.txt" \
  --topk-json "$TOPK_JSON" \
  --taxonomy-json "${SCRIPT_DIR}/../../../data/semi-aves/semi-aves_metrics-LAION400M-taxonomy-enriched.json" \
  --api-model "$API_MODEL" \
  --api-base "$API_BASE" \
  --env-file "$ENV_FILE" \
  --output-csv "$OUTPUT_CSV" \
  --error-file "$ERROR_LOG"
  # Add --dry-run here to test your configuration