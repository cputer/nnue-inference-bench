#!/usr/bin/env python3
"""Update README.md benchmark section from LATEST_NNUE.json."""

from __future__ import annotations
import json
import re
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).parent.parent
    readme_path = repo_root / "README.md"
    json_path = repo_root / "bench" / "results" / "LATEST_NNUE.json"

    if not json_path.exists():
        print(f"Error: {json_path} not found. Run benchmark first.")
        return 1

    data = json.loads(json_path.read_text(encoding="utf-8"))
    results = data["results"]
    config = data["config"]
    meta = data["meta"]

    throughput = f"{results['throughput_pos_per_s']:,.0f}"
    model_name = Path(meta["model_path"]).name
    timestamp = meta["timestamp_utc"][:10]

    lines = [
        "| Device | Batch | p50 (ms) | p95 (ms) | Throughput | Checksum |",
        "|--------|-------|----------|----------|------------|----------|",
        f"| {results['device'].upper()} | {config['batch_size']} | {results['p50_batch_ms']:.3f} | {results['p95_batch_ms']:.3f} | {throughput} pos/s | {results['checksum']} |",
        "",
        f"*Updated: {timestamp} | Model: {model_name}*"
    ]
    table = chr(10).join(lines)

    readme_content = readme_path.read_text(encoding="utf-8")

    pattern = r"<!-- AUTO:BENCHMARK_START -->.*?<!-- AUTO:BENCHMARK_END -->"
    replacement = "<!-- AUTO:BENCHMARK_START -->" + chr(10) + table + chr(10) + "<!-- AUTO:BENCHMARK_END -->"

    new_content = re.sub(pattern, replacement, readme_content, flags=re.DOTALL)

    if new_content == readme_content:
        print("No changes needed")
        return 0

    readme_path.write_text(new_content, encoding="utf-8")
    print(f"Updated README.md from {json_path.name}")
    print(f"  Throughput: {throughput} pos/s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())