#!/usr/bin/env bash
set -euo pipefail

# Wrapper: calls run_qwen.sh with a specific prompt template.
# This is the primary file you should edit to configure and run an experiment.
# It is compatible with ALL prompt templates.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ==============================================================================
# >>> EDIT THESE 2 SECTIONS TO CONFIGURE YOUR EXPERIMENT <<<
# ==============================================================================

# 1. Choose the prompt template you want to run.
#    e.g., top5-sci-with-confidence, zeroshot-explanation, top5-simple-with-confidence, top5-flat-with-confidence, zeroshot-all200-explanation.
#    e.g., top5-sci, zeroshot, top5-simple, top5-flat, zeroshot-all200.
PROMPT_TEMPLATE="top5-multimodal-16shot-with-confidence"

# 2. The base model name affects the TOPK_JSON path and output filenames.
#    Choose from vitb32_openclip_laion400m, resnet50_imagenet_pretrained, resnet50_inat_pretrained, dinov2_vitb14_reg, dinov3_vitb16
BASE_MODEL="vitb32_openclip_laion400m"

# ==============================================================================
# --- Define paths for the experiment ---
# ==============================================================================
OUTPUT_CSV="${SCRIPT_DIR}/../mllm_output/nebius_${BASE_MODEL}_${PROMPT_TEMPLATE}.csv"
ERROR_LOG="${SCRIPT_DIR}/../mllm_output/nebius_${BASE_MODEL}_${PROMPT_TEMPLATE}_errors.txt"
ENV_FILE="${SCRIPT_DIR}/../../../.env"

# These paths are now used by ALL top5 templates, including multimodal.
TOPK_JSON="${SCRIPT_DIR}/../../../data/semi-aves/topk/fewshot_finetune_${BASE_MODEL}_semi-aves_16_1_topk_test_predictions.json"
IMAGE_PATHS_LIST="${SCRIPT_DIR}/../../../data/semi-aves/test.txt"

# This path is ONLY used by the 'top5-multimodal-16shot' template.
REF_IMAGE_DIR_PATH="${SCRIPT_DIR}/../../../datasets/semi-aves/pregenerated_references_16shot"
TAXONOMY_JSON="${SCRIPT_DIR}/../../../data/semi-aves/semi-aves_metrics-LAION400M-taxonomy-enriched.json"

# ==============================================================================
# --- API Key Setup ---
# ==============================================================================
if [[ -f "$ENV_FILE" ]]; then
  source "$ENV_FILE"
fi
: "${NEBIUS_API_KEY:?ERROR: NEBIUS_API_KEY is not set in your .env file}"
export OPENAI_API_KEY="$NEBIUS_API_KEY"

# ==============================================================================
# --- Call the Generic Runner Script ---
# ==============================================================================
"${SCRIPT_DIR}/run_qwen.sh" \
  --backend nebius \
  --python "${SCRIPT_DIR}/../run_inference.py" \
  --prompt-template "$PROMPT_TEMPLATE" \
  --prompt-dir "${SCRIPT_DIR}/../prompt_templates" \
  --image-dir "${SCRIPT_DIR}/../../../datasets/semi-aves" \
  --image-paths "$IMAGE_PATHS_LIST" \
  --topk-json "$TOPK_JSON" \
  --taxonomy-json "$TAXONOMY_JSON" \
  --ref-image-dir "$REF_IMAGE_DIR_PATH" \
  --api-model "Qwen/Qwen2.5-VL-72B-Instruct" \
  --api-base "https://api.studio.nebius.com/v1/" \
  --env-file "$ENV_FILE" \
  --output-csv "$OUTPUT_CSV" \
  --error-file "$ERROR_LOG"
  # Add --dry-run above this line to test configuration
