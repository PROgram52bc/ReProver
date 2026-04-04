import re
import csv
import argparse
from collections import defaultdict

def parse_log(file_path):
    """
    Parses a ReProver log file.
    Aligns repairs with their original failing tactics and captures full compiler output.
    """
    theorems = defaultdict(lambda: {'outcome': 'Unknown', 'steps': []})
    current_thm = None
    
    thm_pattern = re.compile(r"Proving Theorem\(.*full_name='([^']+)'\)")
    node_pattern = re.compile(r"Expanding node: .*?state=TacticState\(pp='(.*?)'\)")
    sug_pattern = re.compile(r"Tactic suggestions: \[(.*)\]")
    
    # New patterns for comprehensive logging
    tac_success_pattern = re.compile(r"(\[REPAIR\] )?Tactic succeeded: (.*?) \| New State: (.*)")
    tac_finished_pattern = re.compile(r"(\[REPAIR\] )?Tactic finished proof: (.*)")
    tac_failed_pattern = re.compile(r"(\[REPAIR\] )?Tactic failed: (.*?) \| Error: (.*)")
    
    repair_link_pattern = re.compile(r"Fixed tactic: (.*?) -> (.*)")
    full_output_start = re.compile(r"Full Repair Model Output:")
    
    solved_pattern = re.compile(r"Found a proof!|status=.*?PROVED")
    failed_pattern = re.compile(r"Hit the resource limit|Failed early!|status=.*?OPEN")

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # 1. Theorem Detection
        m = thm_pattern.search(line)
        if m:
            current_thm = m.group(1).strip()
            i += 1
            continue
        if not current_thm:
            i += 1
            continue
                
        # 2. Outcome Detection
        if solved_pattern.search(line):
            theorems[current_thm]['outcome'] = 'Solved'
        elif failed_pattern.search(line):
            if theorems[current_thm]['outcome'] != 'Solved':
                theorems[current_thm]['outcome'] = 'Failed'

        # 3. State Detection
        m_node = node_pattern.search(line)
        if m_node:
            current_state = m_node.group(1).replace("\\n", "\n").strip()
            theorems[current_thm]['steps'].append({'state': current_state, 'tactics': {}})
            i += 1
            continue
            
        if not theorems[current_thm]['steps']:
            i += 1
            continue
        current_tactics = theorems[current_thm]['steps'][-1]['tactics']

        # 4. Suggestions Detection
        m_sug = sug_pattern.search(line)
        if m_sug:
            sugs = re.findall(r"\('(.*?)', ([-\d.e]+)\)", m_sug.group(1))
            for tac, _ in sugs:
                if tac not in current_tactics:
                    current_tactics[tac] = {
                        'result': 'Attempted', 
                        'repair_tac': None, 
                        'repair_res': None, 
                        'full_repair_output': ''
                    }
            i += 1
            continue

        # 5. Tactic Outcome (Success/Finished/Failed)
        m_s = tac_success_pattern.search(line)
        m_f = tac_finished_pattern.search(line)
        m_e = tac_failed_pattern.search(line)
        
        match = m_s or m_f or m_e
        if match:
            is_repair = bool(match.group(1))
            tac = match.group(2).strip()
            
            if m_s: res = f"Success: {match.group(3).strip()}"
            elif m_f: res = "Success: Finished Proof"
            else: res = f"Failed: {match.group(3).strip()}"
            
            if is_repair:
                # Find which original tactic this repair belongs to
                for t_info in current_tactics.values():
                    if t_info['repair_tac'] == tac:
                        t_info['repair_res'] = res
                        break
            else:
                if tac in current_tactics:
                    current_tactics[tac]['result'] = res
            i += 1
            continue

        # 6. Repair Detection (Link original to fixed)
        m_repair = repair_link_pattern.search(line)
        if m_repair:
            old_tac, new_tac = m_repair.group(1).strip(), m_repair.group(2).strip()
            if old_tac in current_tactics:
                current_tactics[old_tac]['repair_tac'] = new_tac
                current_tactics[old_tac]['repair_res'] = 'Attempted'
            i += 1
            continue

        # 7. Full Model Explanation (Multiline)
        if full_output_start.search(line):
            output_lines = []
            i += 1
            while i < len(lines):
                next_line = lines[i]
                if re.match(r"^\d{4}-\d{2}-\d{2}", next_line) or re.search(r"\| (INFO|DEBUG|ERROR) \|", next_line):
                    break
                output_lines.append(next_line.strip())
                i += 1
            
            full_text = "\n".join(output_lines).strip()
            for t in current_tactics.values():
                if t.get('repair_tac') and not t.get('full_repair_output'):
                    t['full_repair_output'] = full_text
                    break
            continue
            
        i += 1
                    
    return theorems

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_log")
    parser.add_argument("repair_log")
    parser.add_argument("--output", default="comparison_report.csv")
    args = parser.parse_args()

    print(f"Parsing Baseline: {args.baseline_log}")
    baseline = parse_log(args.baseline_log)
    print(f"Parsing Repair Run: {args.repair_log}")
    repair = parse_log(args.repair_log)
    all_thms = sorted(set(baseline.keys()) | set(repair.keys()))
    
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Theorem", "State", "Original Tactic", 
            "Baseline Result (Compiler Output)", "Repair Run: Orig Result (Compiler Output)", 
            "Repair Model Explanation", "Repair Model Suggestion (Fixed Tactic)", 
            "Repair Tactic Result (Compiler Output)",
            "Thm Baseline Outcome", "Thm Repair Outcome"
        ])
        
        for thm in all_thms:
            b_thm = baseline.get(thm, {'outcome': 'N/A', 'steps': []})
            r_thm = repair.get(thm, {'outcome': 'N/A', 'steps': []})
            
            def normalize_state(s): return re.sub(r'\s+', ' ', s).strip()

            aligned = defaultdict(lambda: {'base': None, 'repair': None})
            for step in b_thm['steps']:
                s = normalize_state(step['state'])
                for tac, info in step['tactics'].items():
                    aligned[(s, tac)]['base'] = info
            for step in r_thm['steps']:
                s = normalize_state(step['state'])
                for tac, info in step['tactics'].items():
                    aligned[(s, tac)]['repair'] = info
            
            for (state, tactic), info in sorted(aligned.items()):
                b_info = info['base'] or {}
                r_info = info['repair'] or {}
                
                writer.writerow([
                    thm, state, tactic,
                    b_info.get('result', 'N/A'),
                    r_info.get('result', 'N/A'),
                    r_info.get('full_repair_output', ''),
                    r_info.get('repair_tac', ''),
                    r_info.get('repair_res', ''),
                    b_thm['outcome'],
                    r_thm['outcome']
                ])
    print(f"Comparison report saved to {args.output}")

if __name__ == "__main__":
    main()
