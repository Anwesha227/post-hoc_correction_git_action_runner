#!/usr/bin/env python3
"""
Phase 2: Information Extractor for the Knowledge Extraction Pipeline.

This script reads a YAML schema and a directory of scraped text files. For each
text file, it uses a Large Language Model (LLM) to extract information according
to the schema and populates a structured JSON object.

This script is non-visual and uses a text-only model endpoint. It saves
progress incrementally after each successful API call.
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
import re

def load_and_prepare_schema(schema_path: Path) -> str:
    """Loads the YAML schema and formats it as a string for the prompt."""
    with schema_path.open('r', encoding='utf-8') as f:
        schema = yaml.safe_load(f)
    # Convert to a YAML string which is often more readable in prompts
    return yaml.dump(schema, default_flow_style=False)

def build_extraction_prompt(schema_str: str, scraped_text: str) -> str:
    """Builds the master prompt for the LLM to perform extraction."""
    return f"""
You are an expert data architect and ornithologist's assistant. Your task is to extract structured information from the provided text based on a given schema.

**INSTRUCTIONS:**
1.  Carefully read the YAML schema provided below.
2.  Read the "WIKIPEDIA ARTICLE TEXT" that follows.
3.  Your goal is to fill in the values for every field in the schema using ONLY information found in the article text.
4.  Do NOT invent or infer any information that is not explicitly mentioned in the text.
5.  If a piece of information for a specific field cannot be found, you MUST use an empty string "" as the value for that field. Do not use "null" or "N/A".
6.  Your final output must be ONLY the completed schema as a single, valid JSON object. Do not include any conversational text, explanations, or markdown formatting like ```json.

**SCHEMA TO POPULATE:**
```yaml
{schema_str}```

**WIKIPEDIA ARTICLE TEXT:**
{scraped_text}
**FINAL JSON OUTPUT:**
"""

def clean_llm_json_response(response_text: str) -> Dict[str, Any]:
    """
    Cleans and parses the LLM's response to ensure it's valid JSON.
    It uses a regular expression to find the JSON block, which is more
    robust against surrounding text and markdown.
    """
    # This regex finds a JSON object that starts with { and ends with }
    # It handles nested braces correctly.
    match = re.search(r'\{.*\}', response_text, re.DOTALL)
    
    if not match:
        raise ValueError(f"No valid JSON object found in the LLM response. Response text: {response_text[:500]}...")
        
    json_str = match.group(0)
    
    # Parse the cleaned string into a Python dictionary
    return json.loads(json_str)

def call_llm_api(client: OpenAIClient, model: str, prompt: str) -> str:
    """Makes an API call to an OpenAI-compatible text completion endpoint."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,  # Set to 0 for deterministic, factual extraction
            top_p=1.0,
        )
        # --- THIS IS THE CORRECTED LINE ---
        # We must access the first element of the 'choices' list.
        return (response.choices[0].message.content or "").strip()
    except Exception as e:
        raise e

def main():
    parser = argparse.ArgumentParser(description="Phase 2: Extract structured data from text using an LLM.")
    parser.add_argument("--schema-file", required=True, help="Path to the YAML schema file.")
    parser.add_argument("--scraped-dir", required=True, help="Directory containing the scraped .txt files from Phase 1.")
    parser.add_argument("--output-json", required=True, help="Path to save the final aggregated JSON output.")
    parser.add_argument("--log-file", default="./extraction_log.txt", help="Path to the log file.")
    parser.add_argument("--api-model", required=True, help="Name of the text model to use (e.g., Qwen/Qwen2.5-72B-Instruct).")
    parser.add_argument("--api-base", required=True, help="Base URL for the OpenAI-compatible API.")
    parser.add_argument("--throttle-sec", type=float, default=1.0, help="Seconds to wait between API calls.")
    args = parser.parse_args()

    # --- Setup ---
    schema_path = Path(args.schema_file)
    scraped_dir = Path(args.scraped_dir)
    output_path = Path(args.output_json)
    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("NEBIUS_API_KEY") or os.environ.get("HYPERBOLIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("ERROR: An API key (e.g., NEBIUS_API_KEY) must be set as an environment variable.")

    client = OpenAIClient(base_url=args.api_base, api_key=api_key)
    
    print(f"Loading schema from {schema_path}...")
    schema_str = load_and_prepare_schema(schema_path)

    # --- Load or initialize results (Resume Logic) ---
    results: Dict[str, Any] = {}
    if output_path.exists():
        print(f"Found existing output file at {output_path}. Resuming...")
        try:
            with output_path.open('r', encoding='utf-8') as f:
                # Handle empty file case
                if output_path.stat().st_size > 0:
                    results = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse existing output file '{output_path}'. Starting fresh.")
            results = {}

    # --- Main Extraction Loop ---
    all_text_files = sorted([p for p in scraped_dir.glob("*.txt") if p.stem.isdigit()], key=lambda p: int(p.stem))
    
    with log_path.open("a", encoding="utf-8") as log_f:
        for text_file in tqdm(all_text_files, desc="Extracting Information"):
            class_id_str = text_file.stem
            if class_id_str in results:
                continue  # Skip if we already have a result for this ID

            try:
                scraped_text = text_file.read_text(encoding="utf-8")
                
                if not scraped_text.strip():
                    log_f.write(f"SKIP | ID: {class_id_str} | Reason: Scraped text file is empty.\n")
                    continue
                
                prompt = build_extraction_prompt(schema_str, scraped_text)
                llm_response_str = call_llm_api(client, args.api_model, prompt)
                extracted_data = clean_llm_json_response(llm_response_str)
                
                results[class_id_str] = extracted_data
                log_f.write(f"SUCCESS | ID: {class_id_str}\n")
                
                # --- MODIFIED: Save progress incrementally after each success ---
                with output_path.open("w", encoding="utf-8") as out_f:
                    json.dump(results, out_f, indent=2)

                time.sleep(args.throttle_sec)

            except Exception as e:
                error_message = str(e).replace('\n', ' ')
                log_f.write(f"FAIL | ID: {class_id_str} | Reason: {error_message}\n")
                continue
                
    print(f"\nExtraction complete. Final structured data saved to '{output_path}'.")
    print(f"Log file saved to '{log_path}'.")


if __name__ == "__main__":
    main()