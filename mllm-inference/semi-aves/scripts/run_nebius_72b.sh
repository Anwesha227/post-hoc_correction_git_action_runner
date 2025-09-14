#!/usr/bin/env bash
set -euo pipefail

# Universal Wrapper: calls run_qwen.sh for any Nebius experiment.
# This is the primary file you should edit to configure and run an experiment.
# It intelligently adds the correct arguments based on the chosen prompt template.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/../../.."
echo $PROJECT_ROOT

# ==============================================================================
# >>> EDIT THESE 2 SECTIONS TO CONFIGURE YOUR EXPERIMENT <<<
# ==============================================================================

# 1. Choose ANY prompt template you want to run.
#    - Standard: "top5-sci-with-confidence", "zeroshot-explanation", etc.
#    - Multimodal: "top5-multimodal-16shot", "top5-multimodal-16shot-with-confidence"
#    - Description-based: "top5_with_descriptions", "top5_with_descriptions-with-confidence"

PROMPT_TEMPLATE="top5-multimodal-16shot-with-confidence"


# 2. The base model name affects output filenames and data paths.
#    Choose from vitb32_openclip_laion400m, resnet50_imagenet_pretrained, resnet50_inat_pretrained, dinov2_vitb14_reg, dinov3_vitb16
BASE_MODEL="vitb32_openclip_laion400m"

# ==============================================================================
# --- Define paths for the experiment ---
# ==============================================================================

# --- Shared paths ---
OUTPUT_CSV="${SCRIPT_DIR}/../mllm_output/nebius_${BASE_MODEL}_${PROMPT_TEMPLATE}.csv"
ERROR_LOG="${SCRIPT_DIR}/../mllm_output/nebius_${BASE_MODEL}_${PROMPT_TEMPLATE}_errors.txt"
ENV_FILE="${PROJECT_ROOT}/.env"
TAXONOMY_JSON="${PROJECT_ROOT}/data/semi-aves/semi-aves_metrics-LAION400M-taxonomy-enriched.json"
IMAGE_PATHS_LIST="${PROJECT_ROOT}/data/semi-aves/test.txt"

# --- Paths for specific prompt types ---
TOPK_JSON="${PROJECT_ROOT}/data/semi-aves/topk/fewshot_finetune_${BASE_MODEL}_semi-aves_16_1_topk_test_predictions.json"
REF_IMAGE_DIR_PATH="${PROJECT_ROOT}/datasets/semi-aves/pregenerated_references_16shot"
DESCRIPTIONS_JSON="${PROJECT_ROOT}/data/semi-aves/knowledge_pipeline/species_descriptions_complete.json"

# ==============================================================================
# --- API Key Setup ---
# ==============================================================================
if [[ -f "$ENV_FILE" ]]; then
  source "$ENV_FILE"
fi
: "${NEBIUS_API_KEY:?ERROR: NEBIUS_API_KEY is not set in your .env file}"
export OPENAI_API_KEY="$NEBIUS_API_KEY"

# ==============================================================================
# --- Dynamic Execution Logic ---
# ==============================================================================

# Build the base command array with arguments common to all templates.
CMD=(
  "${SCRIPT_DIR}/run_qwen.sh"
  "--backend" "nebius"
  "--python" "${SCRIPT_DIR}/../run_inference.py"
  "--prompt-template" "$PROMPT_TEMPLATE"
  "--prompt-dir" "${SCRIPT_DIR}/../prompt_templates"
  "--image-dir" "${PROJECT_ROOT}/datasets/semi-aves"
  "--image-paths" "$IMAGE_PATHS_LIST"
  "--taxonomy-json" "$TAXONOMY_JSON"
  "--api-model" "Qwen/Qwen2.5-VL-72B-Instruct"
  "--api-base" "https://api.studio.nebius.com/v1/"
  "--env-file" "$ENV_FILE"
  "--output-csv" "$OUTPUT_CSV"
  "--error-file" "$ERROR_LOG"
)

# --- Intelligently add arguments based on the prompt template ---

# Add --topk-json if it's any kind of top5 prompt.
if [[ "$PROMPT_TEMPLATE" == top5* ]]; then
  CMD+=("--topk-json" "$TOPK_JSON")
fi

# Add --ref-image-dir if it's a multimodal prompt.
if [[ "$PROMPT_TEMPLATE" == *multimodal* ]]; then
  CMD+=("--ref-image-dir" "$REF_IMAGE_DIR_PATH")
fi

# Add --descriptions-json if it's a description-based prompt.
if [[ "$PROMPT_TEMPLATE" == *with_descriptions* ]]; then
  CMD+=("--descriptions-json" "$DESCRIPTIONS_JSON")
fi

# To add a dry-run for testing, uncomment the following line:
# CMD+=("--dry-run")

# --- Execute the Final Command ---
echo ">> Running experiment with template: $PROMPT_TEMPLATE"
# The "${CMD[@]}" syntax safely handles all arguments, even if they contain spaces.
exec "${CMD[@]}"