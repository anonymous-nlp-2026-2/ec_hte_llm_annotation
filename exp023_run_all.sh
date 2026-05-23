#!/bin/bash
# exp-023: Run semi-synthetic validation after annotations are complete
set -e

cd "$(dirname "$0")"

echo "=== Checking prerequisites ==="
if [ ! -f results/exp023_confusion_matrices.json ]; then
    echo "ERROR: results/exp023_confusion_matrices.json not found"
    echo "Run exp023_annotate_parallel.py first"
    exit 1
fi

echo "=== Running semi-synthetic DGP ==="
python3 exp023_civil_comments_semisynthetic.py

echo "=== Done ==="
echo "Results:"
ls -la results/exp023_semisynthetic_*.{csv,md}
