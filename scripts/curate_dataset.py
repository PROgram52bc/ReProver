# Script to curate the dataset of failed tactics and their ground truth.

import json
import argparse
from pathlib import Path
from loguru import logger

def main():
    parser = argparse.ArgumentParser(
        description="Curate a dataset of failed tactics for error correction."
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        required=True,
        help="Path to the JSONL log file containing failed tactic data.",
    )
    parser.add_argument(
        "--leandojo-dataset-path",
        type=Path,
        required=True,
        help="Path to the LeanDojo dataset (e.g., data/leandojo_benchmark_4/random/val.json)",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Path to save the curated dataset.",
    )
    args = parser.parse_args()

    failed_tactics_data = []
    with open(args.log_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                # Handle both raw JSON lines and lines with "FAILED_TACTIC_DATA:" prefix
                if "FAILED_TACTIC_DATA:" in line:
                    json_str = line.split("FAILED_TACTIC_DATA:", 1)[1].strip()
                else:
                    json_str = line
                failed_tactics_data.append(json.loads(json_str))
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON from log line: {line} - {e}")

    logger.info(f"Loaded {len(failed_tactics_data)} failed tactic entries from logs.")

    leandojo_data = json.loads(args.leandojo_dataset_path.read_text())
    logger.info(f"Loaded {len(leandojo_data)} theorems from LeanDojo dataset.")

    curated_dataset = []

    # Create a mapping from theorem full_name to its traced_tactics for efficient lookup
    theorem_to_traced_tactics = {}
    for theorem_entry in leandojo_data:
        theorem_to_traced_tactics[theorem_entry["full_name"]] = theorem_entry["traced_tactics"]

    for failed_entry in failed_tactics_data:
        # Support both old and new key names
        theorem_name = failed_entry.get("theorem") or failed_entry.get("theorem_name")
        failed_proof_state = failed_entry.get("state") or failed_entry.get("proof_state")
        failed_tactic_str = failed_entry.get("tactic") or failed_entry.get("failed_tactic")
        error_message = failed_entry.get("error") or failed_entry.get("error_message")

        if not (theorem_name and failed_proof_state):
            continue

        correct_tactic = None
        if theorem_name in theorem_to_traced_tactics:
            traced_tactics = theorem_to_traced_tactics[theorem_name]
            for traced_tac_entry in traced_tactics:
                if traced_tac_entry["state_before"] == failed_proof_state:
                    correct_tactic = traced_tac_entry["tactic"]
                    break
        
        if correct_tactic:
            formatted_input = (
                f"State: {failed_proof_state} | "
                f"Bad Tactic: {failed_tactic_str} | "
                f"Error: {error_message}"
            )
            curated_dataset.append({"input": formatted_input, "target": correct_tactic})
        # else:
        #    logger.warning(
        #        f"Could not find ground truth for theorem {theorem_name} with proof state: {failed_proof_state}"
        #    )
    
    logger.info(f"Curated {len(curated_dataset)} entries.")

    with open(args.output_file, "w") as f:
        json.dump(curated_dataset, f, indent=4)
    logger.info(f"Curated dataset saved to {args.output_file}")

if __name__ == "__main__":
    main()
