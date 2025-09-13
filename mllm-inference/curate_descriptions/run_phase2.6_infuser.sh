#!/usr/bin/env bash
set -euo pipefail

# This script runs Phase 2.6 of the Knowledge Extraction Pipeline:
# the final knowledge infusion step for sparse entries.
# It provides all necessary arguments to the Python script.

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/../.." # The root of the project

# --- Define Paths ---
PYTHON_SCRIPT="${SCRIPT_DIR}/2.6_knowledge_infusion.py"
ENV_FILE="${PROJECT_ROOT}/.env"

# Input: The schema defining what to extract.
SCHEMA_FILE="${PROJECT_ROOT}/data/semi-aves/knowledge_pipeline/bird_schema.yml"

# Input: The JSON file from Phase 2.5, which may still have some gaps.
INPUT_JSON="${PROJECT_ROOT}/data/semi-aves/knowledge_pipeline/species_descriptions_final.json"

# Input: The main taxonomy file, needed to get the common and scientific names for the prompt.
TAXONOMY_JSON="${PROJECT_ROOT}/data/semi-aves/semi-aves_metrics-LAION400M-taxonomy-enriched.json"

# Output: The final, complete knowledge base.
OUTPUT_JSON="${PROJECT_ROOT}/data/semi-aves/knowledge_pipeline/species_descriptions_complete.json"

# Output: The log file for this infusion phase.
LOG_FILE="${PROJECT_ROOT}/data/semi-aves/knowledge_pipeline/infusion_log.txt"

# --- API Configuration ---
# This phase uses a powerful TEXT-ONLY model.
API_MODEL="Qwen/Qwen2.5-72B-Instruct"
API_BASE="https://api.studio.nebius.com/v1/"

# --- Pre-flight Checks ---
if [[ ! -f "$PYTHON_SCRIPT" ]]; then echo "ERROR: Python script not found: $PYTHON_SCRIPT" >&2; exit 1; fi
if [[ ! -f "$SCHEMA_FILE" ]]; then echo "ERROR: Schema file not found: $SCHEMA_FILE" >&2; exit 1; fi
if [[ ! -f "$INPUT_JSON" ]]; then echo "ERROR: Input JSON from Phase 2.5 not found: $INPUT_JSON" >&2; exit 1; fi
if [[ ! -f "$TAXONOMY_JSON" ]]; then echo "ERROR: Taxonomy JSON not found: $TAXONOMY_JSON" >&2; exit 1; fi

# --- API Key Setup ---
if [[ -f "$ENV_FILE" ]]; then
  source "$ENV_FILE"
  echo ">> Loaded .env file from: $ENV_FILE"
else
  echo "ERROR: .env file not found at: $ENV_FILE" >&2; exit 1;
fi

: "${NEBIUS_API_KEY:?ERROR: NEBIUS_API_KEY is not set in your .env file}"
export OPENAI_API_KEY="$NEBIUS_API_KEY"
echo ">> DEBUG: Exporting API Key starting with '${OPENAI_API_KEY:0:4}'..."


# --- Execute the Infuser ---
echo "================================================="
echo "Starting Phase 2.6: Knowledge Infuser"
echo "================================================="
echo "Schema source    : $SCHEMA_FILE"
echo "Input JSON       : $INPUT_JSON"
echo "Taxonomy source  : $TAXONOMY_JSON"
echo "Output JSON      : $OUTPUT_JSON"
echo "Log file         : $LOG_FILE"
echo "API Model        : $API_MODEL"
echo "-------------------------------------------------"

# This command now passes all the arguments that 2.6_knowledge_infusion.py requires.
python3 "$PYTHON_SCRIPT" \
  --schema-file "$SCHEMA_FILE" \
  --input-json "$INPUT_JSON" \
  --taxonomy-json "$TAXONOMY_JSON" \
  --output-json "$OUTPUT_JSON" \
  --log-file "$LOG_FILE" \
  --api-model "$API_MODEL" \
  --api-base "$API_BASE"

echo "-------------------------------------------------"
echo "Phase 2.6 complete. Knowledge base is finalized."
echo "================================================="