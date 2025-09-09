#!/usr/bin/env bash
set -euo pipefail

# Wrapper: calls run_qwen.sh with fixed arguments
# Edit BASE_MODEL below to change backbone tag in filenames and TOPK JSON
# Don't forget to run first : chmod +x run_nebius_72b.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# >>> EDIT THIS LINE AS NEEDED <<<
BASE_MODEL="vitb32_openclip_laion400m"   # or openclip, resnet, ...

# Output + error paths that include BASE_MODEL
OUTPUT_CSV="${SCRIPT_DIR}/../mllm_output/nebius_${BASE_MODEL}_Qwen_Qwen2.5-VL-72B-Instruct_qwen_top5_grouped_withconf.csv"
ERROR_LOG="${SCRIPT_DIR}/../mllm_output/nebius_${BASE_MODEL}_Qwen_Qwen2.5-VL-72B-Instruct_qwen_top5_grouped_withconf_errors.txt"

TOPK_JSON="${SCRIPT_DIR}/../../../data/semi-aves/topk/fewshot_finetune_vitb32_openclip_laion400m_semi-aves_16_1_topk_test_predictions.json"

"${SCRIPT_DIR}/run_qwen.sh" \
  --backend nebius \
  --python "${SCRIPT_DIR}/../qwen_top5_grouped_withconf.py" \
  --image-dir "${SCRIPT_DIR}/../../../datasets/semi-aves" \
  --image-paths "${SCRIPT_DIR}/../../../data/semi-aves/test.txt" \
  --topk-json "${TOPK_JSON}" \
  --taxonomy-json "${SCRIPT_DIR}/../../../data/semi-aves/semi-aves_metrics-LAION400M-taxonomy-enriched.json" \
  --api-model "Qwen/Qwen2.5-VL-72B-Instruct" \
  --api-base "https://api.studio.nebius.com/v1/" \
  --env-file "${SCRIPT_DIR}/../../../.env" \
  --output-csv "$OUTPUT_CSV" \
  --error-file "$ERROR_LOG"