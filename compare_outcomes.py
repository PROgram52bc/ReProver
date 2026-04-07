import re
import csv
import argparse
import os
from collections import defaultdict

def parse_theorem_outcomes(log_file):
    """
    Parses a ReProver log file and extracts the final outcome for each theorem.
    Optimized for robustness across different log versions and Ray actor formats.
    """
    outcomes = {}
    thm_name_pattern = re.compile(r"Proving (?:Theorem\(.*full_name=['\"]([^'\"]+)['\"]|(\S+))")
    
    success_patterns = [
        re.compile(r"Found a proof!"),
        re.compile(r"status=(?:Status\.PROVED|<Status\.PROVED: 'Proved'>)"),
        re.compile(r"Tactic finished proof")
    ]
    
    timeout_pattern = re.compile(r"Hit the resource limit")
    failed_pattern = re.compile(r"Failed early!")

    current_thm = None
    
    if not os.path.exists(log_file):
        print(f"Warning: File {log_file} not found.")
        return outcomes

    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            m_name = thm_name_pattern.search(line)
            if m_name:
                name = m_name.group(1) or m_name.group(2)
                if name:
                    current_thm = name
                    if current_thm not in outcomes:
                        outcomes[current_thm] = 'Failed'
                continue
            
            if not current_thm:
                continue
                
            if any(p.search(line) for p in success_patterns):
                outcomes[current_thm] = 'Proved'
                continue
            
            if timeout_pattern.search(line) and outcomes[current_thm] != 'Proved':
                outcomes[current_thm] = 'Timeout'
                
            if failed_pattern.search(line) and outcomes[current_thm] != 'Proved':
                outcomes[current_thm] = 'Failed'

    return outcomes

def main():
    parser = argparse.ArgumentParser(description="Compare theorem outcomes across multiple ReProver logs.")
    parser.add_argument("logs", nargs='+', help="Paths to one or more log files.")
    parser.add_argument("--output", default="outcome_comparison.csv", help="Output CSV path.")
    args = parser.parse_args()

    all_outcomes = []
    log_names = []
    
    for log_path in args.logs:
        name = os.path.basename(log_path)
        print(f"Parsing Log: {name}")
        outcomes = parse_theorem_outcomes(log_path)
        all_outcomes.append(outcomes)
        log_names.append(name)

    # Get sorted union of all theorems found in ANY log
    all_thms = set()
    for outcomes in all_outcomes:
        all_thms.update(outcomes.keys())
    all_thms = sorted(list(all_thms))
    
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Header: Theorem, Log1, Log2, ...
        writer.writerow(["Theorem"] + log_names)
        
        for thm in all_thms:
            row = [thm]
            for outcomes in all_outcomes:
                row.append(outcomes.get(thm, "N/A"))
            writer.writerow(row)

    print("\n" + "="*50)
    print("             COMPARISON SUMMARY")
    print("="*50)
    print(f"Total Theorems Attempted : {len(all_thms)}")
    
    for i, name in enumerate(log_names):
        proved_count = sum(1 for res in all_outcomes[i].values() if res == 'Proved')
        print(f"Proved in {name:20}: {proved_count}")
    
    print("="*50)
    print(f"Detailed breakdown saved to: {args.output}\n")

if __name__ == "__main__":
    main()
