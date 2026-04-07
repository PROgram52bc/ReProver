import re
import argparse
import sys

def parse_tree_logs(log_file):
    nodes = {} # id -> state
    edges = [] # (id, src, dst, tactic, type)
    
    # regexes
    node_pattern = re.compile(r"\[TREE_NODE\] ID: (.*?) \| State: (.*)")
    edge_pattern = re.compile(r"\[TREE_EDGE\] ID: (\d+) \| Src: (.*?) \| Dst: (.*?) \| Tactic: (.*?) \| Type: (.*)")
    
    # Process multiple theorems separately if they exist in the same log
    # For simplicity, this script will plot the first theorem found or a specific one if requested
    
    with open(log_file, 'r') as f:
        for line in f:
            line = line.strip()
            
            m_node = node_pattern.search(line)
            if m_node:
                node_id = m_node.group(1).strip()
                state = m_node.group(2).strip()
                nodes[node_id] = state
                
            m_edge = edge_pattern.search(line)
            if m_edge:
                edge_id = m_edge.group(1)
                src = m_edge.group(2).strip()
                dst = m_edge.group(3).strip()
                tac = m_edge.group(4).strip()
                etype = m_edge.group(5).strip()
                edges.append((edge_id, src, dst, tac, etype))
                
    return nodes, edges

def generate_mermaid(nodes, edges):
    mermaid = ["graph LR"]
    
    # Define nodes with truncated labels for readability
    for nid, state in nodes.items():
        # Truncate long states
        label = (state[:50] + '...') if len(state) > 50 else state
        # Escape quotes for Mermaid
        label = label.replace('"', "'").replace("\n", " ")
        
        # Color coding
        if "FINISH" in nid:
            mermaid.append(f'    {nid}["{label}"]')
            mermaid.append(f'    style {nid} fill:#9f9,stroke:#333,stroke-width:2px')
        elif "ERROR" in nid:
            mermaid.append(f'    {nid}["{label}"]')
            mermaid.append(f'    style {nid} fill:#f99,stroke:#333,stroke-width:1px')
        else:
            mermaid.append(f'    {nid}["Node {nid}: {label}"]')

    # Define edges
    for eid, src, dst, tac, etype in edges:
        # Style repair edges differently
        if etype == "Repair":
            mermaid.append(f'    {src} -- "{tac} (REPAIR)" --> {dst}')
            # Use linkStyle to make repair lines dashed (requires index calculation usually, but we can use IDs)
        else:
            mermaid.append(f'    {src} -- "{tac}" --> {dst}')
            
    return "\n".join(mermaid)

def main():
    parser = argparse.ArgumentParser(description="Generate Mermaid plot from ReProver tree logs.")
    parser.add_argument("log_file", help="The log file containing [TREE_NODE] and [TREE_EDGE] entries.")
    parser.add_argument("--output", help="Output file for the mermaid code (defaults to stdout).")
    args = parser.parse_args()
    
    nodes, edges = parse_tree_logs(args.log_file)
    
    if not nodes and not edges:
        print("No tree data found in log file.")
        return

    mermaid_code = generate_mermaid(nodes, edges)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(mermaid_code)
        print(f"Mermaid code saved to {args.output}")
    else:
        print(mermaid_code)

if __name__ == "__main__":
    main()
