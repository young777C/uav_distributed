#!/bin/bash
# Generate the final xelatex chapter 5 with filled-in results.
# Run after experiments complete.

set -e
cd "$(git rev-parse --show-toplevel)"

echo "=== Step 1: Analyze TS sensitivity results ==="
PYTHONPATH=src python3 scripts/analyze_ts_results.py --latex-table > /tmp/ts_latex_tables.tex 2>&1 || echo "Results not yet available"
cat /tmp/ts_latex_tables.tex

echo ""
echo "=== Step 2: Check current section structure ==="
grep -n '\\\\section\|\\\\subsection\|\\\\label{sec:' chapter5_xelatex_final.tex | head -30

echo ""
echo "=== Done ==="
echo "To compile: xelatex chapter5_xelatex_final.tex"
