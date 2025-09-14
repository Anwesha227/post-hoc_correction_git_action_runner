#!/usr/bin/env python3
"""
Phase 2.6: Knowledge Infusion for the Knowledge Extraction Pipeline.

This is the final fallback step. This script reads the output from Phase 2.5,
identifies any entries that are still sparse, and uses a text-only LLM's
pre-trained knowledge to fill in the remaining gaps.
"""

import os
import sys
import json
import time
import argparse
import yaml  # Requires PyYAML: pip install PyYAML
from pathlib import Path
from typing import Dict, Any

from openai import OpenAI as OpenAIClient
from tqdm import tqdm

# --- CONFIGURATION ---
MIN_QUALITY_SCORE = 25

def calculate_quality_score(species_data: Dict[str, Any]) -> int:
    """Calculates a simple quality score (number of non-empty string fields)."""
    score = 0
    def count_filled(data):
        nonlocal score
        if isinstance(data, dict):
            for value in data.values():
                count_filled(value)
        elif isinstance(data, list):
            for item in data:
                count_filled(item)
        elif isinstance(data, str) and data.strip():
            score += 1
    count_filled(species_data)
    return score

def clean_llm_json_response(response_text: str) -> Dict[str, Any]:
    """Cleans and parses the LLM's response to ensure it's valid JSON."""
    start_brace = response_text.find('{')
    end_brace = response_text.rfind('}')
    if start_brace == -1 or end_brace == -1:
        raise ValueError(f"No valid JSON object found in LLM response: {response_text[:500]}...")
    json_str = response_text[start_brace : end_brace + 1]
    return json.loads(json_str)

def merge_data(original_data: Dict, new_data: Dict) -> Dict:
    """Recursively merges new data into original data, only filling empty fields."""
    for key, new_value in new_data.items():
        if key in original_data:
            original_value = original_data[key]
            if isinstance(original_value, dict) and isinstance(new_value, dict):
                original_data[key] = merge_data(original_value, new_value)
            elif (isinstance(original_value, str) and not original_value.strip()) and \
                 (isinstance(new_value, str) and new_value.strip()):
                original_data[key] = new_value
    return original_data

def build_knowledge_infusion_prompt(common_name: str, scientific_name: str, schema_str: str) -> str:
    """Builds the text-only prompt for the LLM to use its internal knowledge."""
    
    # --- PROMPT TEXT ---
    prompt = prompt = f"""
    You are an expert ornithologist and data architect, acting as a factual database.

    **TASK:**
    I will provide you with the name of a bird species and a YAML schema. Your task is to populate the schema with concise, factual descriptions for this species by recalling information from your pre-trained knowledge base.

    **SPECIES TO DESCRIBE:**
    Common Name: {common_name}
    Scientific Name: {scientific_name}

    **CRITICAL INSTRUCTIONS:**
    1.  **Recall, Do Not Create:** Use your internal knowledge to fill in the visual attributes for the species named above.
    2.  **Accuracy is Paramount:** DO NOT MAKE ANYTHING UP. You must only provide information that is widely accepted and scientifically documented. If you are not highly confident about a specific feature, you MUST leave the value as an empty string "". Your primary goal is accuracy and the avoidance of hallucination.
    3.  **Be Concise:** Provide short, descriptive phrases (e.g., "bright red" or "streaked with brown").
    4.  **Strictly Adhere to the Schema:** Your final output must be ONLY a single, valid JSON object that perfectly follows the provided schema. Do not add any conversational text, explanations, markdown formatting like ```json, or any fields not present in the schema.

    **SCHEMA TO POPULATE:**
        ```yaml
        {schema_str}```

    **FINAL JSON OUTPUT:**
    """
    
    return prompt

def call_text_llm_api(client: OpenAIClient, model: str, prompt: str) -> str:
    """Makes a text-only API call."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            top_p=1.0,
        )
        
        return (response.choices[0].message.content or "").strip()
    except Exception as e:
        raise e

def main():
    parser = argparse.ArgumentParser(description="Phase 2.6: Infuse knowledge from an LLM for sparse entries.")
    parser.add_argument("--schema-file", required=True)
    parser.add_argument("--input-json", required=True, help="Path to the JSON file from Phase 2.5.")
    parser.add_argument("--taxonomy-json", required=True, help="Path to the main taxonomy file.")
    parser.add_argument("--output-json", required=True, help="Path to save the final, complete JSON output.")
    parser.add_argument("--log-file", default="./infusion_log.txt")
    parser.add_argument("--api-model", required=True, help="Name of the TEXT-ONLY model.")
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--throttle-sec", type=float, default=1.0)
    args = parser.parse_args()

    # --- Setup ---
    schema_path = Path(args.schema_file)
    input_json_path = Path(args.input_json)
    taxonomy_path = Path(args.taxonomy_json)
    output_path = Path(args.output_json)
    log_file_path = Path(args.log_file)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("NEBIUS_API_KEY") or os.environ.get("HYPERBOLIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key: sys.exit("ERROR: An API key must be set.")

    client = OpenAIClient(base_url=args.api_base, api_key=api_key)
    
    # Load all necessary data
    print("Loading data...")
    with schema_path.open('r', encoding='utf-8') as f: schema = yaml.safe_load(f)
    schema_str = yaml.dump(schema, default_flow_style=False)
    
    with input_json_path.open('r', encoding='utf-8') as f: species_data_to_process = json.load(f)
    with taxonomy_path.open('r', encoding='utf-8') as f: taxonomy_data = json.load(f)

    # --- Resume Logic ---
    final_results: Dict[str, Any] = {}
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"Found existing output file at {output_path}. Resuming...")
        try:
            with output_path.open('r', encoding='utf-8') as f:
                final_results = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse existing output file. Starting fresh.")
            final_results = {}
    
    done_ids = set(final_results.keys())
    print(f"Found {len(done_ids)} already completed entries in the output file. Will skip them.")

    # --- Main Infusion Loop ---
    with log_file_path.open("a", encoding="utf-8") as log_f:
        items_to_process = list(species_data_to_process.items())
        
        for class_id_str, species_entry in tqdm(items_to_process, desc="Infusing Knowledge"):
            # Check against the set of already completed IDs from the output file
            if not class_id_str.isdigit() or class_id_str in done_ids:
                continue

            try:
                quality_score = calculate_quality_score(species_entry)
                if quality_score >= MIN_QUALITY_SCORE:
                    log_f.write(f"SKIP | ID: {class_id_str} | Reason: Quality score ({quality_score}) is sufficient.\n")
                    final_results[class_id_str] = species_entry
                else:
                    tqdm.write(f"  Infusing knowledge for ID: {class_id_str} (Score: {quality_score})")
                    
                    species_info = taxonomy_data.get(class_id_str, {})
                    common_name = species_info.get("most_common_name")
                    scientific_name = species_info.get("name")
                    
                    if not common_name or not scientific_name:
                        raise ValueError(f"Common or scientific name not found for ID {class_id_str}.")

                    prompt = build_knowledge_infusion_prompt(common_name, scientific_name, schema_str)
                    llm_response_str = call_text_llm_api(client, args.api_model, prompt)
                    knowledge_data = clean_llm_json_response(llm_response_str)
                    
                    final_entry = merge_data(species_entry, knowledge_data)
                    final_results[class_id_str] = final_entry
                    
                    log_f.write(f"SUCCESS | ID: {class_id_str} | New Score: {calculate_quality_score(final_entry)}\n")

                # Save the updated 'final_results' dictionary after every successful operation
                with output_path.open("w", encoding="utf-8") as out_f:
                    json.dump(final_results, out_f, indent=2)

                time.sleep(args.throttle_sec)

            except Exception as e:
                error_message = str(e).replace('\n', ' ')
                log_f.write(f"FAIL | ID: {class_id_str} | Reason: {error_message}\n")
                continue

    print(f"\nKnowledge infusion complete. Final data saved to '{output_path}'.")