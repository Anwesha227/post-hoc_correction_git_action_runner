#!/usr/bin/env bash
set -euo pipefail

# This script runs Phase 2.5 of the Knowledge Extraction Pipeline:
# the VLM-powered data quality enricher.

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/../.."

# --- Define Paths ---
PYTHON_SCRIPT="${SCRIPT_DIR}/2.5_enrich_descriptions.py"
ENV_FILE="${PROJECT_ROOT}/.env"
SCHEMA_FILE="${PROJECT_ROOT}/data/semi-aves/knowledge_pipeline/bird_schema.yml"
INPUT_JSON="${PROJECT_ROOT}/data/semi-aves/knowledge_pipeline/species_descriptions.json"
REF_IMAGE_DIR="${PROJECT_ROOT}/datasets/semi-aves/pregenerated_references_16shot"
OUTPUT_JSON="${PROJECT_ROOT}/data/semi-aves/knowledge_pipeline/species_descriptions_final.json"
LOG_FILE="${PROJECT_ROOT}/data/semi-aves/knowledge_pipeline/enrichment_log.txt"

# --- API Configuration ---
# CRITICAL: This phase MUST use a Vision-Language Model (VL).
API_MODEL="Qwen/Qwen2.5-VL-72B-Instruct"
API_BASE="https://api.studio.nebius.com/v1/"

# --- Pre-flight Checks ---
if [[ ! -f "$PYTHON_SCRIPT" ]]; then echo "ERROR: Python script not found: $PYTHON_SCRIPT" >&2; exit 1; fi
if [[ ! -f "$SCHEMA_FILE" ]]; then echo "ERROR: Schema file not found: $SCHEMA_FILE" >&2; exit 1; fi
if [[ ! -f "$INPUT_JSON" ]]; then echo "ERROR: Input JSON from Phase 2 not found: $INPUT_JSON" >&2; exit 1; fi
if [[ ! -d "$REF_IMAGE_DIR" ]]; then echo "ERROR: Reference image directory not found: $REF_IMAGE_DIR" >&2; exit 1; fi

# --- API Key Setup ---
# Load the .env file to make API keys available.
if [[ -f "$ENV_FILE" ]]; then
  source "$ENV_FILE"
  echo ">> Loaded .env file from: $ENV_FILE"
else
  echo "ERROR: .env file not found at: $ENV_FILE" >&2; exit 1;
fi

# Explicitly check for the NEBIUS key and export it as the generic OPENAI_API_KEY
# that the Python script will look for.
: "${NEBIUS_API_KEY:?ERROR: NEBIUS_API_KEY is not set in your .env file}"
export OPENAI_API_KEY="$NEBIUS_API_KEY"
echo ">> DEBUG: Exporting API Key starting with '${OPENAI_API_KEY:0:4}'..."


# --- Execute the Enricher ---
echo "================================================="
echo "Starting Phase 2.5: VLM Data Enricher"
echo "================================================="
echo "Schema source    : $SCHEMA_FILE"
echo "Input JSON       : $INPUT_JSON"
echo "Reference Images : $REF_IMAGE_DIR"
echo "Output JSON      : $OUTPUT_JSON"
echo "Log file         : $LOG_FILE"
echo "API Model        : $API_MODEL"
echo "-------------------------------------------------"

python3 "$PYTHON_SCRIPT" \
  --schema-file "$SCHEMA_FILE" \
  --input-json "$INPUT_JSON" \
  --ref-image-dir "$REF_IMAGE_DIR" \
  --output-json "$OUTPUT_JSON" \
  --log-file "$LOG_FILE" \
  --api-model "$API_MODEL" \
  --api-base "$API_BASE"

echo "-------------------------------------------------"
echo "Phase 2.5 complete."
echo "================================================="