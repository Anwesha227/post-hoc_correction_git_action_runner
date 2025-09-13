#!/usr/bin/env python3
"""
Phase 2.5: Data Quality Enhancer for the Knowledge Extraction Pipeline.

This script reads the output JSON from Phase 2, measures the data quality
of each species entry, and enriches "low-quality" entries by using a
Vision-Language Model (VLM) to generate descriptions from example images.

It intelligently merges the new data, filling in only the empty fields and
preserving any information that was already successfully extracted from text.
"""

import os
import sys
import json
import time
import argparse
import base64
import yaml  # Requires PyYAML: pip install PyYAML
from pathlib import Path
from typing import Dict, Any

from openai import OpenAI as OpenAIClient
from tqdm import tqdm

# --- CONFIGURATION ---
# Any species with fewer filled fields than this will be enriched.
MIN_QUALITY_SCORE = 25

def load_image_b64(image_path: Path) -> str:
    """Encodes an image to base64 for API calls."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def calculate_quality_score(species_data: Dict[str, Any]) -> int:
    """Calculates a simple quality score (number of non-empty string fields)."""
    score = 0
    # Use a recursive helper function to handle nested dictionaries
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

def build_visual_enrichment_prompt(schema_str: str) -> str:
    """Builds the prompt text for the VLM, asking it to fill the schema from an image."""
    
    # --- PROMPT TEXT ---
    prompt = f"""
    You are an expert ornithologist tasked with describing a bird from an image.

    **TASK:**
    Carefully examine the bird in the provided image. Based ONLY on what you can see in the image, fill in the following YAML schema with concise, factual descriptions.

    **INSTRUCTIONS:**
    1.  Describe only the most prominent bird in the image if there are multiple.
    2.  If you cannot determine a feature from the image (e.g., the legs are not visible), you MUST leave the value as an empty string "".
    3.  Do not invent or infer information not present in the image (e.g., do not guess the habitat if it's a plain background).
    4.  Your final output must be ONLY a single, valid JSON object that follows the schema. Do not include any conversational text, explanations, or markdown formatting like ```json.

    **SCHEMA TO POPULATE:**
    ```yaml
    {schema_str}```

    **FINAL JSON OUTPUT:**
    """
    
    return prompt

def clean_llm_json_response(response_text: str) -> Dict[str, Any]:
    """Cleans and parses the LLM's response to ensure it's valid JSON."""
    start_brace = response_text.find('{')
    end_brace = response_text.rfind('}')
    if start_brace == -1 or end_brace == -1:
        raise ValueError(f"No valid JSON object found in LLM response: {response_text[:500]}...")
    json_str = response_text[start_brace : end_brace + 1]
    return json.loads(json_str)

def call_vlm_api(client: OpenAIClient, model: str, prompt_text: str, image_b64: str) -> str:
    """Makes a multimodal API call to an OpenAI-compatible endpoint."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]
            }],
            temperature=0.0,
            top_p=1.0,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e:
        raise e

def merge_data(original_data: Dict, new_data: Dict) -> Dict:
    """
    Recursively merges the new data into the original data.
    It only fills fields in 'original_data' that are empty.
    """
    for key, new_value in new_data.items():
        if key in original_data:
            original_value = original_data[key]
            if isinstance(original_value, dict) and isinstance(new_value, dict):
                original_data[key] = merge_data(original_value, new_value)
            elif (isinstance(original_value, str) and not original_value.strip()) and \
                 (isinstance(new_value, str) and new_value.strip()):
                original_data[key] = new_value
    return original_data

def main():
    parser = argparse.ArgumentParser(description="Phase 2.5: Enrich extracted data with VLM.")
    parser.add_argument("--schema-file", required=True, help="Path to the YAML schema file.")
    parser.add_argument("--input-json", required=True, help="Path to the JSON file from Phase 2.")
    parser.add_argument("--ref-image-dir", required=True, help="Path to directory with reference images (named like '0.jpg').")
    parser.add_argument("--output-json", required=True, help="Path to save the final, enriched JSON output.")
    parser.add_argument("--log-file", default="./enrichment_log.txt", help="Path to the log file.")
    parser.add_argument("--api-model", required=True, help="Name of the Vision-Language Model to use (e.g., Qwen/Qwen2.5-VL-72B-Instruct).")
    parser.add_argument("--api-base", required=True, help="Base URL for the OpenAI-compatible API.")
    parser.add_argument("--throttle-sec", type=float, default=1.0, help="Seconds to wait between API calls.")
    args = parser.parse_args()

    # --- Setup ---
    schema_path = Path(args.schema_file)
    input_json_path = Path(args.input_json)
    ref_image_dir = Path(args.ref_image_dir)
    output_path = Path(args.output_json)
    log_file_path = Path(args.log_file)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("NEBIUS_API_KEY") or os.environ.get("HYPERBOLIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("ERROR: An API key must be set as an environment variable.")

    client = OpenAIClient(base_url=args.api_base, api_key=api_key)
    
    print(f"Loading schema from {schema_path}...")
    with schema_path.open('r', encoding='utf-8') as f:
        schema = yaml.safe_load(f)
    schema_str = yaml.dump(schema, default_flow_style=False)
    
    print(f"Loading existing data from {input_json_path}...")
    with input_json_path.open('r', encoding='utf-8') as f:
        species_data: Dict[str, Any] = json.load(f)

    # Prepare the prompt text once, as it's the same for all images
    prompt_text = build_visual_enrichment_prompt(schema_str)

    # --- Main Enrichment Loop ---
    with log_file_path.open("a", encoding="utf-8") as log_f:
        items_to_process = list(species_data.items())
        for class_id_str, species_entry in tqdm(items_to_process, desc="Validating & Enriching"):
            if not class_id_str.isdigit():
                continue

            try:
                quality_score = calculate_quality_score(species_entry)

                if quality_score >= MIN_QUALITY_SCORE:
                    log_f.write(f"SKIP | ID: {class_id_str} | Reason: Quality score ({quality_score}) meets threshold (>= {MIN_QUALITY_SCORE}).\n")
                    continue
                
                tqdm.write(f"  Enriching ID: {class_id_str} (Quality Score: {quality_score})")
                
                image_path = ref_image_dir / f"{class_id_str}.jpg"
                if not image_path.exists():
                    raise FileNotFoundError(f"Reference image not found: {image_path}")

                image_b64 = load_image_b64(image_path)
                
                # Call the VLM API
                llm_response_str = call_vlm_api(client, args.api_model, prompt_text, image_b64)
                
                # Clean the response from the VLM
                vlm_extracted_data = clean_llm_json_response(llm_response_str)
                
                # Merge the new data intelligently
                enriched_entry = merge_data(species_entry, vlm_extracted_data)
                species_data[class_id_str] = enriched_entry
                
                log_f.write(f"SUCCESS | ID: {class_id_str} | New Score: {calculate_quality_score(enriched_entry)}\n")
                
                # Save progress incrementally
                with output_path.open("w", encoding="utf-8") as out_f:
                    json.dump(species_data, out_f, indent=2)

                time.sleep(args.throttle_sec)

            except Exception as e:
                error_message = str(e).replace('\n', ' ')
                log_f.write(f"FAIL | ID: {class_id_str} | Reason: {error_message}\n")
                continue

    print(f"\nEnrichment complete. Final data saved to '{output_path}'.")
    print(f"Log file saved to '{log_file_path}'.")

if __name__ == "__main__":
    main()