#!/usr/bin/env bash
set -euo pipefail

# Picks the Hyperbolic key from .env, exports OPENAI_API_KEY for the run,
# and passes the Hyperbolic base/model through to run_qwen.sh.
# Don't forget to run first : chmod +x run_hyperbolic_7b.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Load repo-level .env (same location pattern you use for 72B) ---
ENV_FILE="${SCRIPT_DIR}/../../../.env"
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  echo ">> Loaded env from: $ENV_FILE"
else
  echo ">> No .env found at $ENV_FILE (continuing)"
fi

# --- Force Hyperbolic provider for this wrapper ---
: "${HYPERBOLIC_API_KEY:?HYPERBOLIC_API_KEY not set in .env}"
export OPENAI_API_KEY="$HYPERBOLIC_API_KEY"
API_BASE="https://api.hyperbolic.xyz/v1"
API_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"

# --- Keep your BASE_MODEL naming/paths pattern ---
BASE_MODEL="vitb32_openclip_laion400m"

OUTPUT_CSV="${SCRIPT_DIR}/../mllm_output/hyperbolic_${BASE_MODEL}_Qwen_Qwen2.5-VL-7B-Instruct_qwen_zeroshot_all200_explanation_rerun.csv"
ERROR_LOG="${SCRIPT_DIR}/../mllm_output/hyperbolic_${BASE_MODEL}_Qwen_Qwen2.5-VL-7B-Instruct_qwen_zeroshot_all200_explanation_errors.txt"

TOPK_JSON="${SCRIPT_DIR}/../../../data/semi-aves/topk/fewshot_finetune_${BASE_MODEL}_semi-aves_16_1_topk_test_predictions.json"

# --- Handoff (identical structure to your 72B wrapper) ---
"${SCRIPT_DIR}/run_qwen.sh" \
  --backend nebius \
  --python "${SCRIPT_DIR}/../qwen_zeroshot_all200_explanation.py" \
  --image-dir "${SCRIPT_DIR}/../../../datasets/semi-aves" \
  --image-paths "${SCRIPT_DIR}/../mllm_output/hyperbolic_vitb32_openclip_laion400m_Qwen_Qwen2.5-VL-7B-Instruct_qwen_zeroshot_all200_explanation_errors.txt" \
  --topk-json "${TOPK_JSON}" \
  --taxonomy-json "${SCRIPT_DIR}/../../../data/semi-aves/semi-aves_metrics-LAION400M-taxonomy-enriched.json" \
  --api-model "${API_MODEL}" \
  --api-base "${API_BASE}" \
  --env-file "${ENV_FILE}" \
  --output-csv "$OUTPUT_CSV" \
  --error-file "$ERROR_LOG"
# --image-paths "${SCRIPT_DIR}/../../../data/semi-aves/test.txt" \