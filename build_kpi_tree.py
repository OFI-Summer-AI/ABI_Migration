import json
import re
import os

def build_kpi_hierarchy():
    input_file = os.path.join('data', 'raw', 'extracted_kpis.json')
    output_file = os.path.join('data', 'raw', 'kpi_hierarchy.json')
    
    if not os.path.exists(input_file):
        print(f"Error: Could not find {input_file}")
        return
        
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        kpis = json.load(f)
        
    kpi_dict = {k['kpi_id']: k for k in kpis}
    
    # Pre-process backward compatibility:
    # If standard newly-extracted JSON doesn't have 'dependencies' array yet
    # because it was extracted before our updates, parse it now.
    all_dependencies = set()
    for kpi in kpis:
        if 'dependencies' not in kpi:
            formula = kpi.get('pql_formula', '')
            deps = re.findall(r'KPI\([\'"]([^\'"]+)[\'"]\)', formula)
            kpi['dependencies'] = deps
        all_dependencies.update(kpi['dependencies'])
        
    print(f"Found {len(kpis)} total metrics/attributes.")

    # Recursive function to build deep tree
    def get_tree(kpi_id, visited=None):
        if visited is None:
            visited = set()
            
        if kpi_id in visited:
            return {"kpi_id": kpi_id, "error": "Circular dependency detected"}
            
        visited.add(kpi_id)
        
        node = kpi_dict.get(kpi_id)
        if not node:
            return {"kpi_id": kpi_id, "warning": "Definition not found in JSON"}
            
        # Base node structure
        result = {
            "kpi_id": kpi_id,
            "name": node.get("name", kpi_id),
            "pql_formula": node.get("pql_formula", "")
        }
        
        deps = node.get('dependencies', [])
        if deps:
            # Recursion for nested dependencies
            result['depends_on'] = [get_tree(d, visited.copy()) for d in deps]
            
        return result

    # To build the macro-hierarchy, we find ROOT nodes.
    # Root nodes are those that NO OTHER KPI depends on.
    root_kpis = [kpi['kpi_id'] for kpi in kpis if kpi['kpi_id'] not in all_dependencies]
    print(f"Identified {len(root_kpis)} root metrics.")
    
    hierarchy = []
    for root in root_kpis:
        tree = get_tree(root)
        # We might only care about complex KPIs, but let's save everything
        hierarchy.append(tree)
        
    print(f"Saving deep hierarchy to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(hierarchy, f, indent=2)
        
    # Extra: print a quick sample for "invoice_value" to show the user
    print("\n--- Summary for 'invoice_value' abstract ---")
    sample_tree = get_tree("invoice_value")
    print(json.dumps(sample_tree, indent=2))

if __name__ == "__main__":
    build_kpi_hierarchy()
