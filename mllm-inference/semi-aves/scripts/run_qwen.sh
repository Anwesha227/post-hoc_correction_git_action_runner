#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# This script requires explicit flags. No hidden defaults.
# It derives the output directory as: <script>/../mllm_output
# CSV and error log filenames include: <backend>_<model>
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"   # e.g., .../mllm_inference/semi-aves
OUTPUT_DIR="${DATASET_DIR}/mllm_output"
mkdir -p "${OUTPUT_DIR}"

usage() {
  cat <<'EOF'
Required (common):
  --backend nebius|ollama
  --python PATH                 # run_qwen_inference_zeroshot_all200.py
  --image-dir PATH
  --image-paths PATH
  --topk-json PATH
  --taxonomy-json PATH

Required (backend=nebius):
  --api-model NAME
  --api-base URL
  [optional] --env-file PATH    # if provided, will be sourced; must contain NEBIUS_API_KEY or it must be exported in shell

Required (backend=ollama):
  --ollama-model NAME[:tag]
  --ollama-host URL

Optional:
  --output-csv PATH
  --error-file PATH
  --dry-run

Example:
  ./run_qwen_required.sh \
    --backend nebius \
    --python /abs/path/run_qwen_inference_zeroshot_all200.py \
    --image-dir /abs/datasets/semi-aves \
    --image-paths /abs/data/semi-aves/test.txt \
    --topk-json /abs/data/semi-aves/topk/preds.json \
    --taxonomy-json /abs/data/semi-aves/taxonomy.json \
    --api-model Qwen/Qwen2.5-VL-72B-Instruct \
    --api-base https://api.studio.nebius.com/v1/ \
    --env-file /abs/.env
EOF
}

sanitize_for_filename() {
  echo "$1" | sed 's/[^A-Za-z0-9._-]/_/g'
}

# -------------------- parse flags --------------------
BACKEND=""
PY_SCRIPT=""
IMAGE_DIR=""
IMAGE_PATHS=""
TOPK_JSON=""
TAXONOMY_JSON=""
API_MODEL=""
API_BASE=""
OLLAMA_MODEL=""
OLLAMA_HOST=""
ENV_FILE=""
OUTPUT_CSV=""
ERROR_FILE=""
DRY_RUN="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)        BACKEND="${2:-}"; shift 2 ;;
    --python)         PY_SCRIPT="${2:-}"; shift 2 ;;
    --image-dir)      IMAGE_DIR="${2:-}"; shift 2 ;;
    --image-paths)    IMAGE_PATHS="${2:-}"; shift 2 ;;
    --topk-json)      TOPK_JSON="${2:-}"; shift 2 ;;
    --taxonomy-json)  TAXONOMY_JSON="${2:-}"; shift 2 ;;
    --api-model)      API_MODEL="${2:-}"; shift 2 ;;
    --api-base)       API_BASE="${2:-}"; shift 2 ;;
    --ollama-model)   OLLAMA_MODEL="${2:-}"; shift 2 ;;
    --ollama-host)    OLLAMA_HOST="${2:-}"; shift 2 ;;
    --env-file)       ENV_FILE="${2:-}"; shift 2 ;;
    --output-csv)     OUTPUT_CSV="${2:-}"; shift 2 ;;
    --error-file)     ERROR_FILE="${2:-}"; shift 2 ;;
    --dry-run)        DRY_RUN="true"; shift ;;
    -h|--help)        usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

# -------------------- hard validation --------------------
# Common required
[[ -n "$BACKEND"      ]] || { echo "ERROR: --backend is required" >&2; usage; exit 1; }
[[ -n "$PY_SCRIPT"    ]] || { echo "ERROR: --python is required" >&2; usage; exit 1; }
[[ -n "$IMAGE_DIR"    ]] || { echo "ERROR: --image-dir is required" >&2; usage; exit 1; }
[[ -n "$IMAGE_PATHS"  ]] || { echo "ERROR: --image-paths is required" >&2; usage; exit 1; }
[[ -n "$TOPK_JSON"    ]] || { echo "ERROR: --topk-json is required" >&2; usage; exit 1; }
[[ -n "$TAXONOMY_JSON" ]] || { echo "ERROR: --taxonomy-json is required" >&2; usage; exit 1; }
[[ -f "$PY_SCRIPT"    ]] || { echo "ERROR: Python script not found: $PY_SCRIPT" >&2; exit 1; }

case "$BACKEND" in
  nebius)
    [[ -n "$API_MODEL" ]] || { echo "ERROR: --api-model is required for backend=nebius" >&2; exit 1; }
    [[ -n "$API_BASE"  ]] || { echo "ERROR: --api-base is required for backend=nebius"  >&2; exit 1; }
    if [[ -n "$ENV_FILE" ]]; then
      [[ -f "$ENV_FILE" ]] || { echo "ERROR: --env-file not found: $ENV_FILE" >&2; exit 1; }
      # shellcheck disable=SC1090
      source "$ENV_FILE"
      echo ">> Loaded env from: $ENV_FILE"
    fi
    : "${NEBIUS_API_KEY:=}"
    [[ -n "$NEBIUS_API_KEY" ]] || { echo "ERROR: NEBIUS_API_KEY not set/exported" >&2; exit 1; }
    MODEL_FOR_NAME="$API_MODEL"
    ;;
  ollama)
    [[ -n "$OLLAMA_MODEL" ]] || { echo "ERROR: --ollama-model is required for backend=ollama" >&2; exit 1; }
    [[ -n "$OLLAMA_HOST"  ]] || { echo "ERROR: --ollama-host is required for backend=ollama"  >&2; exit 1; }
    command -v curl >/dev/null 2>&1 || { echo "ERROR: curl is required to probe Ollama" >&2; exit 1; }
    curl -sSf -m 3 "${OLLAMA_HOST}/api/version" >/dev/null || {
      echo "ERROR: Can't reach Ollama at ${OLLAMA_HOST} (is 'ollama serve' running?)" >&2; exit 1; }
    MODEL_FOR_NAME="$OLLAMA_MODEL"
    ;;
  *) echo "ERROR: --backend must be 'nebius' or 'ollama'"; exit 1 ;;
esac

MODEL_SAFE="$(sanitize_for_filename "${MODEL_FOR_NAME}")"

# -------------------- outputs (derived unless overridden) --------------------
# Use the Python script's filename (without .py) as the task name suffix
BASE_PY="$(basename "$PY_SCRIPT")"            # e.g., qwen_zeroshot_all200_explanation.py
BASE_NOEXT="${BASE_PY%.py}"                   # e.g., qwen_zeroshot_all200_explanation
BASE_SAFE="$(sanitize_for_filename "$BASE_NOEXT")"

# model-safe already depends on backend; we compute it above or now:
# (MODEL_FOR_NAME is set in the backend case-switch)
MODEL_SAFE="$(sanitize_for_filename "${MODEL_FOR_NAME}")"

if [[ -z "$OUTPUT_CSV" ]]; then
  OUTPUT_CSV="${OUTPUT_DIR}/${BACKEND}_${MODEL_SAFE}_${BASE_SAFE}.csv"
fi
if [[ -z "$ERROR_FILE" ]]; then
  ERROR_FILE="${OUTPUT_DIR}/${BACKEND}_${MODEL_SAFE}_${BASE_SAFE}_errors.txt"
fi

mkdir -p "$(dirname "$OUTPUT_CSV")" "$(dirname "$ERROR_FILE")"

# -------------------- build + run --------------------
CMD=( python3 "$PY_SCRIPT"
  --backend "$BACKEND"
  --api-model "$API_MODEL"
  --api-base "${API_BASE:-}"
  --ollama-model "$OLLAMA_MODEL"
  --ollama-host "$OLLAMA_HOST"
  --image-dir "$IMAGE_DIR"
  --image-paths "$IMAGE_PATHS"
  --topk-json "$TOPK_JSON"
  --taxonomy-json "$TAXONOMY_JSON"
  --output-csv "$OUTPUT_CSV"
  --error-file "$ERROR_FILE"
)

echo ">> Backend       : $BACKEND"
if [[ "$BACKEND" == "nebius" ]]; then
  echo ">> API model     : $API_MODEL"
  echo ">> API base      : $API_BASE"
else
  echo ">> Ollama model  : $OLLAMA_MODEL"
  echo ">> Ollama host   : $OLLAMA_HOST"
fi
echo ">> Image dir     : $IMAGE_DIR"
echo ">> test paths    : $IMAGE_PATHS"
echo ">> TOPK JSON     : $TOPK_JSON"
echo ">> Taxonomy JSON : $TAXONOMY_JSON"
echo ">> Output CSV    : $OUTPUT_CSV"
echo ">> Error log     : $ERROR_FILE"
echo ">> Python script : $PY_SCRIPT"

if [[ "$DRY_RUN" == "true" ]]; then
  printf 'DRY RUN: %q ' "${CMD[@]}"; echo
  exit 0
fi

exec "${CMD[@]}"
