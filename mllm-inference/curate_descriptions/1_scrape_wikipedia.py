#!/usr/bin/env python3
"""
Phase 1: Wikipedia Scraper for the Knowledge Extraction Pipeline.

This script takes a taxonomy file as input, iterates through each species,
and scrapes the text content of the corresponding Wikipedia page.

It intelligently extracts BOTH the structured data from the species infobox
and the unstructured text from the main article sections to create a
comprehensive source document for each species.

Input:
- A taxonomy JSON file (e.g., semi-aves_metrics-LAION400M-taxonomy-enriched.json)
  which contains scientific names for each class ID.

Output:
- A directory of .txt files, where each file is named with a class_id
  (e.g., '13.txt') and contains the cleaned text from that species' article.
- A log file detailing successes and failures.
"""

import os
import sys
import json
import time
import argparse
import requests
from pathlib import Path

import wikipediaapi
from bs4 import BeautifulSoup
from tqdm import tqdm

# Sections to exclude from the final text to reduce noise
SECTIONS_TO_EXCLUDE = [
    "see also", "references", "external links", "gallery", "notes",
    "further reading", "citations", "etymology", "taxonomy"
]

def clean_section_title(title: str) -> str:
    """Prepares a section title for comparison by making it lowercase."""
    return title.strip().lower()

def get_infobox_data(page: wikipediaapi.WikipediaPage) -> str:
    """
    Makes a direct HTML request to get the page and parses the 'infobox' table
    to extract structured key-value data. This is crucial for facts like
    conservation status, mass, length, etc.
    """
    try:
        response = requests.get(page.fullurl, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # The main species infobox usually has the class 'infobox biota'
        infobox = soup.find('table', class_='infobox biota')
        if not infobox:
            return ""

        infobox_data = []
        infobox_data.append("[INFOBOX DATA]")
        
        # Find all table rows `<tr>` within the infobox
        for row in infobox.find_all('tr'):
            # Get header (th) and data (td) cells
            header = row.find('th')
            data = row.find('td')
            
            if header and data:
                # Clean up text by removing extra spaces, newlines, and citations like [1]
                key = ' '.join(header.get_text(strip=True).split())
                
                # Remove citation brackets (e.g., [1], [2], [a]) before getting text
                for citation in data.find_all('sup', class_='reference'):
                    citation.decompose()
                
                value = ' '.join(data.get_text(strip=True).split())
                
                if key and value:
                    infobox_data.append(f"{key}: {value}")
        
        return "\n".join(infobox_data)
    except Exception:
        # If anything goes wrong with the HTML request or parsing, fail gracefully
        return ""

def get_relevant_text(page: wikipediaapi.WikipediaPage) -> str:
    """
    Parses a page to get the infobox data, the summary, and the main text sections,
    combining them into a single comprehensive text block.
    """
    full_text = []

    # --- Step 1: Get structured data from the infobox first ---
    infobox_content = get_infobox_data(page)
    if infobox_content:
        full_text.append(infobox_content)
        full_text.append("\n" + "="*20 + "\n")
    
    # --- Step 2: Get unstructured text from summary and sections ---
    if page.summary:
        full_text.append("[SUMMARY]")
        full_text.append(page.summary)
        full_text.append("\n" + "="*20 + "\n")

    def process_sections(sections, level=0):
        for s in sections:
            section_title_clean = clean_section_title(s.title)
            if section_title_clean not in SECTIONS_TO_EXCLUDE:
                full_text.append(f"[{s.title.upper()}]")
                full_text.append(s.text)
                full_text.append("\n" + "="*20 + "\n")
                # Recursively process subsections
                process_sections(s.sections, level + 1)

    process_sections(page.sections)
    return "\n".join(full_text).strip()

def main():
    parser = argparse.ArgumentParser(description="Phase 1: Scrape Wikipedia articles for a list of species.")
    parser.add_argument("--taxonomy-json", required=True, help="Path to the taxonomy JSON file.")
    parser.add_argument("--output-dir", required=True, help="Directory to save the scraped .txt files.")
    parser.add_argument("--log-file", default="./scraping_log.txt", help="Path to the log file for successes and failures.")
    parser.add_argument("--language", default="en", help="Wikipedia language to use (e.g., 'en', 'de').")
    parser.add_argument("--throttle-sec", type=float, default=0.5, help="Seconds to wait between requests to be polite.")
    args = parser.parse_args()

    # --- Setup ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = Path(args.log_file)
    
    # A user agent is required for polite scraping.
    # Replace the placeholder with your project's info.
    user_agent = "KnowledgeBaseBuilder/1.0 (https://github.com/your-username/your-repo; your-email@example.com)"
    print(f"Connecting to Wikipedia ({args.language}) with User-Agent: {user_agent}...")
    wiki_wiki = wikipediaapi.Wikipedia(user_agent=user_agent, language=args.language)
    
    print(f"Loading taxonomy from {args.taxonomy_json}...")
    with open(args.taxonomy_json, 'r') as f:
        taxonomy_data = json.load(f)

    # --- Find already scraped files to support resuming ---
    done_ids = {int(p.stem) for p in output_dir.glob("*.txt") if p.stem.isdigit()}
    print(f"Found {len(done_ids)} already scraped articles. Will skip them.")
    
    # --- Main Scraping Loop ---
    with log_file_path.open("a", encoding="utf-8") as log_f:
        # We sort by class ID to ensure deterministic processing order
        sorted_items = sorted(taxonomy_data.items(), key=lambda item: int(item[0]))

        for class_id_str, species_info in tqdm(sorted_items, desc="Scraping Wikipedia"):
            try:
                class_id = int(class_id_str)
                if class_id in done_ids:
                    continue

                # Get the scientific name from the taxonomy file
                scientific_name = species_info.get("name") or species_info.get("scientific_name")
                if not scientific_name:
                    log_message = f"FAIL | ID: {class_id} | Reason: No 'name' or 'scientific_name' key found in taxonomy.\n"
                    log_f.write(log_message)
                    continue
                
                # Query Wikipedia API
                page = wiki_wiki.page(scientific_name)

                if not page.exists():
                    log_message = f"FAIL | ID: {class_id} | Name: {scientific_name} | Reason: Page does not exist.\n"
                    log_f.write(log_message)
                    continue

                # Extract and clean text from both infobox and main article
                content = get_relevant_text(page)
                if not content:
                    log_message = f"FAIL | ID: {class_id} | Name: {scientific_name} | Reason: Page exists but contains no extractable text.\n"
                    log_f.write(log_message)
                    continue

                # Save the text to a file
                output_path = output_dir / f"{class_id}.txt"
                with output_path.open("w", encoding="utf-8") as out_f:
                    out_f.write(content)
                
                log_message = f"SUCCESS | ID: {class_id} | Name: {scientific_name} | URL: {page.fullurl}\n"
                log_f.write(log_message)
                done_ids.add(class_id)
                
                # Be polite to Wikipedia's servers
                time.sleep(args.throttle_sec)

            except Exception as e:
                log_message = f"FAIL | ID: {class_id_str} | Reason: An unexpected error occurred - {e}\n"
                log_f.write(log_message)
                continue

    print(f"\nScraping complete. Results saved in '{output_dir}'.")
    print(f"Log file saved to '{log_file_path}'.")

if __name__ == "__main__":
    main()