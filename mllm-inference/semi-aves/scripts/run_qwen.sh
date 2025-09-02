#!/usr/bin/env bash
set -euo pipefail

# ---------- locate things based on your layout ----------
# repo structure:
# <repo root>/.env
# <repo root>/mllm_inference/semi-aves/run_qwen_inference_zeroshot_all200.py
# <repo root>/mllm_inference/semi-aves/scripts/run_qwen.sh  (this file)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEMIAVES_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"                  # .../mllm_inference/semi-aves
MMLM_DIR="$(cd "${SEMIAVES_DIR}/.." && pwd)"                   # .../mllm_inference
REPO_ROOT="$(cd "${MMLM_DIR}/.." && pwd)"                      # repo root

PY_SCRIPT="${SEMIAVES_DIR}/run_qwen_inference_zeroshot_all200.py"
ROOT_ENV="${REPO_ROOT}/.env"
SCRIPTS_ENV="${SCRIPT_DIR}/.env"

# ---------- defaults (override with flags) ----------
BACKEND="nebius"                                      # nebius | ollama
API_MODEL="Qwen/Qwen2.5-VL-72B-Instruct"
API_BASE="https://api.studio.nebius.com/v1/"
OLLAMA_MODEL="qwen2.5vl:7b"
OLLAMA_HOST="http://127.0.0.1:11434"
DRY_RUN="false"
ENV_FILE="$ROOT_ENV" ; [[ -f "$ENV_FILE" ]] || ENV_FILE="$SCRIPTS_ENV"

# dataset defaults 
IMAGE_DIR="${REPO_ROOT}/datasets/semi-aves/"
IMAGE_PATHS="${REPO_ROOT}/data/semi-aves/test.txt"
TOPK_JSON="${REPO_ROOT}/data/semi-aves/topk/swift_stage3_vitb32_openclip_laion400m_semi-aves_16_1_topk_test_predictions.json"

# We'll compute OUTPUT_CSV AFTER parsing flags so it reflects the final --backend
OUTPUT_CSV_DEFAULT_TEMPLATE='${MMLM_DIR}/mllm_output/${BACKEND}_qwen_zeroshot_all200_explanation.csv'
OUTPUT_CSV=""
OUTPUT_OVERRIDE="false"

ERROR_FILE="${MMLM_DIR}/error_logs/qwen_error_log_zeroshot_all200_explanation.txt"
TAXONOMY_JSON="${REPO_ROOT}/data/semi-aves/semi-aves_metrics-LAION400M-taxonomy-enriched.json"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Backends:
  --backend nebius|ollama
  --api-model NAME
  --api-base URL
  --ollama-model NAME[:tag]
  --ollama-host URL

Dataset paths:
  --image-dir PATH
  --image-paths PATH
  --topk-json PATH
  --output-csv PATH            (default auto-includes backend in filename)
  --error-file PATH
  --taxonomy-json PATH

Other:
  --env-file PATH              (default: repo-root .env, falls back to scripts/.env)
  --python PATH                (default: ${PY_SCRIPT})
  --dry-run
  -h, --help
EOF
}

# ---------- parse flags ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)        BACKEND="${2:-}"; shift 2 ;;
    --api-model)      API_MODEL="${2:-}"; shift 2 ;;
    --api-base)       API_BASE="${2:-}"; shift 2 ;;
    --ollama-model)   OLLAMA_MODEL="${2:-}"; shift 2 ;;
    --ollama-host)    OLLAMA_HOST="${2:-}"; shift 2 ;;
    --image-dir)      IMAGE_DIR="${2:-}"; shift 2 ;;
    --image-paths)    IMAGE_PATHS="${2:-}"; shift 2 ;;
    --topk-json)      TOPK_JSON="${2:-}"; shift 2 ;;
    --output-csv)     OUTPUT_CSV="${2:-}"; OUTPUT_OVERRIDE="true"; shift 2 ;;
    --error-file)     ERROR_FILE="${2:-}"; shift 2 ;;
    --taxonomy-json)  TAXONOMY_JSON="${2:-}"; shift 2 ;;
    --env-file)       ENV_FILE="${2:-}"; shift 2 ;;
    --python)         PY_SCRIPT="${2:-}"; shift 2 ;;
    --dry-run)        DRY_RUN="true"; shift ;;
    -h|--help)        usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

# ---------- finalize derived defaults after parsing ----------
# Set OUTPUT_CSV if user didn't override, ensuring the BACKEND chosen is reflected in the filename
if [[ "$OUTPUT_OVERRIDE" != "true" ]]; then
  eval "OUTPUT_CSV=${OUTPUT_CSV_DEFAULT_TEMPLATE}"
fi

# ---------- load env (prefers root .env, falls back to scripts/.env) ----------
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  echo ">> Loaded env from: $ENV_FILE"
else
  echo ">> No .env found (looked for ${ROOT_ENV} then ${SCRIPTS_ENV})"
fi

# ---------- sanity checks ----------
[[ -f "$PY_SCRIPT" ]] || { echo "ERROR: Python not found at $PY_SCRIPT" >&2; exit 1; }

case "$BACKEND" in
  nebius)
    : "${NEBIUS_API_KEY:=}"
    [[ -n "${NEBIUS_API_KEY}" ]] || { echo "ERROR: NEBIUS_API_KEY missing (set it in $ENV_FILE or export it)" >&2; exit 1; }
    ;;
  ollama)
    command -v curl >/dev/null 2>&1 || { echo "ERROR: curl required to probe Ollama" >&2; exit 1; }
    curl -sSf -m 3 "${OLLAMA_HOST}/api/version" >/dev/null || {
      echo "ERROR: Can't reach Ollama at ${OLLAMA_HOST} (is 'ollama serve' running?)" >&2; exit 1; }
    ;;
  *) echo "ERROR: invalid --backend '$BACKEND' (use nebius|ollama)" >&2; exit 1 ;;
esac

# ---------- build python command ----------
CMD=( python3 "$PY_SCRIPT"
  --backend "$BACKEND"
  --api-model "$API_MODEL"
  --api-base "$API_BASE"
  --ollama-model "$OLLAMA_MODEL"
  --ollama-host "$OLLAMA_HOST"
  --image-dir "$IMAGE_DIR"
  --image-paths "$IMAGE_PATHS"
  --topk-json "$TOPK_JSON"
  --output-csv "$OUTPUT_CSV"
  --error-file "$ERROR_FILE"
  --taxonomy-json "$TAXONOMY_JSON"
)

echo ">> Repo root    : $REPO_ROOT"
echo ">> Semi-aves dir: $SEMIAVES_DIR"
echo ">> Backend      : $BACKEND"
if [[ "$BACKEND" == "nebius" ]]; then
  echo ">> API model    : $API_MODEL"
  echo ">> API base     : $API_BASE"
else
  echo ">> Ollama model : $OLLAMA_MODEL"
  echo ">> Ollama host  : $OLLAMA_HOST"
fi
echo ">> Image dir    : $IMAGE_DIR"
echo ">> test.txt     : $IMAGE_PATHS"
echo ">> TOPK JSON    : $TOPK_JSON"
echo ">> Output CSV   : $OUTPUT_CSV"
echo ">> Error file   : $ERROR_FILE"
echo ">> Taxonomy JSON: $TAXONOMY_JSON"
echo ">> Python script: $PY_SCRIPT"

[[ "$DRY_RUN" == "true" ]] && { printf 'DRY RUN: %q ' "${CMD[@]}"; echo; exit 0; }

exec "${CMD[@]}"
