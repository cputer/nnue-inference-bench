#!/usr/bin/env python3
"""
Validates that README claims match actual benchmark results.
Run this in CI to ensure honesty.
"""

import json
import re
import sys
from pathlib import Path

def load_benchmark_results():
    """Load all verified benchmark results."""
    results_dir = Path(__file__).parent.parent / "bench" / "results"
    results = {}
    
    # Load main results
    latest = results_dir / "LATEST_NNUE.json"
    if latest.exists():
        with open(latest) as f:
            data = json.load(f)
            if "cpu_results" in data:
                results["Python reference"] = {
                    "throughput": data["cpu_results"]["throughput_pos_per_s"],
                    "checksum": data["cpu_results"]["checksum"]
                }
            if "gpu_results" in data:
                results["CUDA DLL"] = {
                    "throughput": data["gpu_results"]["throughput_pos_per_s"],
                    "checksum": data["gpu_results"]["checksum"]
                }
    
    # Load C++ results
    cpp_results = results_dir / "cpp_baseline.json"
    if cpp_results.exists():
        with open(cpp_results) as f:
            data = json.load(f)
            results["C++ baseline"] = {
                "throughput": data["throughput_pos_per_s"],
                "checksum": data["checksum"]
            }
    
    return results

def parse_readme_claims():
    """Parse claims from README.md benchmark table."""
    readme = Path(__file__).parent.parent / "README.md"
    with open(readme, encoding='utf-8') as f:
        content = f.read()
    
    # Find benchmark table between markers
    match = re.search(r'<!-- AUTO:BENCHMARK_START -->(.*?)<!-- AUTO:BENCHMARK_END -->', 
                      content, re.DOTALL)
    if not match:
        return []
    
    table = match.group(1)
    claims = []
    
    # Parse table rows
    for line in table.strip().split('\n'):
        if '|' not in line or '---' in line or 'Implementation' in line:
            continue
        
        parts = [p.strip() for p in line.split('|')[1:-1]]
        if len(parts) >= 6:
            impl = parts[0].replace('**', '').strip()
            throughput_str = parts[4]
            checksum = parts[5]
            
            # Parse throughput (e.g., "81,687 pos/s")
            throughput_match = re.search(r'([\d,]+)\s*pos/s', throughput_str)
            if throughput_match:
                throughput = float(throughput_match.group(1).replace(',', ''))
                claims.append({
                    "implementation": impl,
                    "throughput": throughput,
                    "checksum": checksum
                })
    
    return claims

def validate():
    """Validate README claims against actual results."""
    results = load_benchmark_results()
    claims = parse_readme_claims()
    
    errors = []
    reference_checksum = None
    
    print("Validating benchmark claims...")
    print()
    
    for claim in claims:
        impl = claim["implementation"]
        claimed_throughput = claim["throughput"]
        claimed_checksum = claim["checksum"]
        
        # Set reference checksum from Python reference
        if "Python" in impl:
            reference_checksum = claimed_checksum
        
        print(f"  {impl}:")
        print(f"    Claimed: {claimed_throughput:,.0f} pos/s, {claimed_checksum}")
        
        # Check if we have results for this implementation
        matched = None
        for name, data in results.items():
            if name.lower() in impl.lower() or impl.lower() in name.lower():
                matched = data
                break
        
        if matched:
            actual_throughput = matched["throughput"]
            actual_checksum = matched["checksum"]
            print(f"    Actual:  {actual_throughput:,.0f} pos/s, {actual_checksum}")
            
            # Check checksum matches
            if actual_checksum != claimed_checksum:
                errors.append(f"{impl}: Checksum mismatch (claimed {claimed_checksum}, actual {actual_checksum})")
            
            # Check throughput is within 20% (reasonable variance)
            ratio = claimed_throughput / actual_throughput
            if ratio < 0.8 or ratio > 1.2:
                errors.append(f"{impl}: Throughput mismatch (claimed {claimed_throughput:.0f}, actual {actual_throughput:.0f})")
        else:
            print(f"    Actual:  [no benchmark data found]")
            # Only error if it's not a planned implementation
            if "Mind" not in impl:
                errors.append(f"{impl}: No benchmark results found to verify claim")
        
        print()
    
    # Check all checksums match reference
    if reference_checksum:
        print(f"Reference checksum: {reference_checksum}")
        for claim in claims:
            if claim["checksum"] != reference_checksum:
                errors.append(f"{claim['implementation']}: Checksum doesn't match reference ({claim['checksum']} != {reference_checksum})")
    
    print()
    if errors:
        print("ERRORS:")
        for e in errors:
            print(f"  - {e}")
        return 1
    else:
        print("All claims validated successfully!")
        return 0

if __name__ == "__main__":
    sys.exit(validate())
